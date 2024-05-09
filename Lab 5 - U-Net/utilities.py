import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import copy
import gc
from sklearn.metrics import f1_score, jaccard_score

random.seed(27)
np.random.seed(27)
torch.manual_seed(27)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(27)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_random_data(height, width, count):
    x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)

    return X, Y


def generate_img_and_mask(height, width):
    shape = (height, width)
    # set the seed to 27 so that we can have a reproducible example
    triangle_location = get_random_location(*shape)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    plus_location = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    # Create target masks
    masks = np.asarray(
        [
            add_filled_square(np.zeros(shape, dtype=bool), *square_location),
            add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
            add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
            add_circle(np.zeros(shape, dtype=bool), *circle_location1),
            add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
            # add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
            add_plus(np.zeros(shape, dtype=bool), *plus_location),
        ]
    ).astype(np.float32)

    return arr, masks


def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x - s, y - s : y + s] = True
    arr[x + s, y - s : y + s] = True
    arr[x - s : x + s, y - s] = True
    arr[x - s : x + s, y + s] = True

    return arr


def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]

    return np.logical_or(
        arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s])
    )


def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array


def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]

    return np.logical_or(
        arr,
        logical_and(
            [xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]
        ),
    )


def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x - s : x - s + triangle.shape[0], y - s : y - s + triangle.shape[1]] = triangle

    return arr


def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(
        arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True)
    )

    return new_arr


def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x - 1 : x + 1, y - s : y + s] = True
    arr[x - s : x + s, y - 1 : y + 1] = True

    return arr


def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(
        nrow, ncol, sharex="all", sharey="all", figsize=(ncol * 4, nrow * 4)
    )

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])


def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))


def plot_errors(results_dict, title):
    markers = itertools.cycle(("+", "x", "o"))

    plt.title("{}".format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel("dice_coef")
        plt.xlabel("epoch")
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()


def masks_to_colorimg(masks):
    colors = np.asarray(
        [
            (201, 58, 64),
            (242, 207, 1),
            (0, 152, 75),
            (101, 172, 228),
            (56, 34, 132),
            (160, 194, 56),
        ]
    )

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:, y, x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def generate_images_and_masks_then_plot():
    # Generate some random images
    input_images, target_masks = generate_random_data(192, 192, count=3)

    for x in [input_images, target_masks]:
        print(x.shape)
        print(x.min(), x.max())

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in target_masks]

    # Left: Input image (black and white), Right: Target mask (6ch)
    plot_side_by_side([input_images_rgb, target_masks_rgb])


def reverse_transform(inp):
    if isinstance(inp, np.ndarray):
        inp_copy = inp.copy()  # Make a copy to avoid modifying the original array
    else:
        inp_copy = (
            inp.numpy().copy()
        )  # Convert PyTorch tensor to numpy array and make a copy
    inp_copy = inp_copy.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp_copy = std * inp_copy + mean
    inp_copy = np.clip(inp_copy, 0, 1)
    inp_copy = (inp_copy * 255).astype(np.uint8)
    return inp_copy


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = generate_random_data(
            192, 192, count=count
        )
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # imagenet
        ]
    )

    train_set = SimDataset(100, transform=trans)
    val_set = SimDataset(25, transform=trans)
    test_set = SimDataset(25, transform=trans)
    # image_datasets = {"train": train_set, "val": val_set, "test": test_set}

    batch_size = 25

    dataloaders = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }

    return dataloaders


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()




def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.softmax(pred, dim=1)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics["bce"] += bce.data.cpu().numpy() * target.size(0)
    metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)

    return loss


def calculate_dice_coefficient(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()
    else:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
    return f1_score(y_true, y_pred, average="macro")

def calculate_jaccard_index(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()
    else:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
    return jaccard_score(y_true, y_pred, average="macro")



def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def evaluate_predictions_train(predictions, true_labels):
    # Apply softmax to convert to probabilities
    predictions = F.softmax(predictions, dim=1)
    dice_scores = []
    jaccard_indices = []
    for i in range(len(predictions)):
        pred = predictions[i]
        true_label = true_labels[i]
        pred_class = pred.argmax(dim=0)
        true_label_class = true_label.argmax(dim=0)
        dice_score = calculate_dice_coefficient(pred_class, true_label_class)
        jaccard_index = calculate_jaccard_index(pred_class, true_label_class)
        dice_scores.append(dice_score)
        jaccard_indices.append(jaccard_index)

    return dice_scores, jaccard_indices


def evaluate_predictions_test(predictions, true_labels):
    dice_scores = []
    jaccard_indices = []
    for i in range(len(predictions)):
        pred = predictions[i]
        true_label = true_labels[i]
        pred_class = np.argmax(pred, axis=0)
        true_label_class = np.argmax(true_label, axis=0)
        dice_score = calculate_dice_coefficient(pred_class, true_label_class)
        jaccard_index = calculate_jaccard_index(pred_class, true_label_class)
        dice_scores.append(dice_score)
        jaccard_indices.append(jaccard_index)

    return dice_scores, jaccard_indices


def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_dice = []  # Collect training losses for each epoch
    val_dice = []  # Collect validation losses for each epoch
    train_loss = []  # Collect training losses for each epoch
    val_loss = []  # Collect validation losses for each epoch
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group["lr"])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples
            epoch_dice_loss = metrics["dice"] / epoch_samples

            # Save losses for plotting
            if phase == "train":
                train_dice.append(epoch_dice_loss)
                train_loss.append(epoch_loss)
            else:
                val_dice.append(epoch_dice_loss)
                val_loss.append(epoch_loss)
                dice_scores, jaccard_indices = evaluate_predictions_train(
                    outputs, labels
                )
                print(
                    f"Epoch {epoch}: Average Dice Score for validation:",
                    np.mean(dice_scores),
                )
                print(
                    f"Epoch {epoch}: Average Jaccard Index for validation:",
                    np.mean(jaccard_indices),
                )
            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("Best val loss: {:4f}".format(best_loss))

    # Plot training and validation losses
    dice_dict = {"train": train_dice, "val": val_dice}
    loss_dict = {"train": train_loss, "val": val_loss}
    plot_errors(dice_dict, "Training and Validation Dice Losses")
    plot_errors(loss_dict, "Training and Validation Losses")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    predictions = []
    inputs_list = []
    true_labels = []

    metrics = defaultdict(float)
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(inputs)
            loss = calc_loss(outputs, labels, metrics)
            print("Loss:", loss)

            # Convert predictions to numpy arrays
            predictions.append(outputs.cpu().numpy())
            inputs_list.append(inputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(inputs_list, axis=0),
        np.concatenate(true_labels, axis=0),
    )


def run(UNet, num_epochs=60):
    num_class = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=3, n_classes=num_class).to(device)

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    dataloaders = get_data_loaders()

    model = train_model(
        model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs
    )

    model.eval()  # Set model to the evaluation mode
    # Test set
    test_loader = dataloaders["test"]

    # # Predictions
    predictions, inputs, true_labels = predict(model, test_loader, device)
    print("Predictions shape:", predictions.shape)

    # Jaccard and Dice scores
    print("Evaluating..")
    dice_scores, jaccard_indices = evaluate_predictions_test(predictions, true_labels)
    print("Average Dice Score:", np.mean(dice_scores))
    print("Average Jaccard Index:", np.mean(jaccard_indices))

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(inp) for inp in inputs]
    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in true_labels]
    pred_rgb = [masks_to_colorimg(x) for x in predictions]

    print("Plotting side by side..")
    plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

    # clear the cache
    torch.cuda.empty_cache()
    # clear gpu memory
    del model
    del optimizer_ft
    del exp_lr_scheduler
    del dataloaders
    del test_loader
    del predictions
    del inputs
    del true_labels
    del dice_scores
    del jaccard_indices
    del input_images_rgb
    del target_masks_rgb
    # garbage collection
    gc.collect()
