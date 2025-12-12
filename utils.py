import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


def visualize_mask(masks_path):
    mask = np.array(Image.open(os.path.abspath(masks_path[51])).convert("L"))
    colors = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )

    rgb = colors[mask]
    rgb_img = Image.fromarray(rgb, mode="RGB")
    return rgb_img


COLORS = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)


def mask_to_rgb(mask):
    return COLORS[mask]


def compute_mean_std_gray_from_paths(paths):
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for path in tqdm(paths, desc="Computing mean/std"):
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        pixel_sum += arr.sum()
        pixel_sq_sum += (arr**2).sum()
        pixel_count += arr.size

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean**2)

    return mean, std


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        img_transform=None,
        mask_transform=None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.mask_transform = mask_transform
        self.img_transform = img_transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


def compute_class_weights(loader, num_classes):
    counts = torch.zeros(num_classes)

    for _, masks in tqdm(loader, desc="Counting classes"):
        for c in range(num_classes):
            counts[c] += (masks == c).sum()

    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.sum()

    return weights


def weighted_dice_focal_loss(
    logits,
    targets,
    class_weights,
    dice_weight=0.5,
    gamma=2.0,
    smooth=1.0,
):
    C = logits.shape[1]

    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma * ce).mean()

    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

    probs = probs[:, 1:]
    targets_oh = targets_oh[:, 1:]
    dice_weights = class_weights[1:]

    dims = (0, 2, 3)
    intersection = (probs * targets_oh).sum(dims)
    union = probs.sum(dims) + targets_oh.sum(dims)

    dice = (2 * intersection + smooth) / (union + smooth)
    dice = (dice * dice_weights).sum() / (dice_weights.sum() + 1e-8)

    dice_loss = 1 - dice

    return dice_weight * dice_loss + (1 - dice_weight) * focal


@torch.no_grad()
def visualize_random_sample(model, dataset, device):
    idx = random.randint(0, len(dataset) - 1)

    image, mask_gt = dataset[idx]

    image = image.unsqueeze(0).to(device)
    mask_gt = mask_gt.cpu().numpy()

    logits = model(image)
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    img_np = image[0].cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = img_np[0]
        img_vis = img_np
        cmap = "gray"
    else:
        img_vis = img_np.transpose(1, 2, 0)
        cmap = None

    gt_rgb = mask_to_rgb(mask_gt)
    pred_rgb = mask_to_rgb(pred)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img_vis, cmap=cmap)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT mask")
    plt.imshow(gt_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Pred mask")
    plt.imshow(pred_rgb)
    plt.axis("off")

    plt.show()
