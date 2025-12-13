import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from model import UNetPP


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


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.images[idx])).convert("L")
        mask = Image.open(os.path.join(self.masks_dir, self.masks[idx])).convert("L")

        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


def weighted_dice_focal_loss(logits, targets, class_weights, gamma=2.0, smooth=1.0):
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma * ce).mean()

    probs = F.softmax(logits, dim=1)
    C = logits.shape[1]
    targets_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

    probs = probs[:, 1:]
    targets_oh = targets_oh[:, 1:]
    w = class_weights[1:]

    dims = (0, 2, 3)
    intersection = (probs * targets_oh).sum(dims)
    union = probs.sum(dims) + targets_oh.sum(dims)

    dice = (2 * intersection + smooth) / (union + smooth)
    dice = (dice * w).sum() / (w.sum() + 1e-8)

    return 0.5 * (1 - dice) + 0.5 * focal


def boundary_loss(logits, targets, boundary_class=1):
    probs = torch.softmax(logits, dim=1)[:, boundary_class]
    targets = (targets == boundary_class).float()

    lap = torch.tensor(
        [[0, 1, 0],
         [1,-4, 1],
         [0, 1, 0]],
        device=logits.device,
        dtype=torch.float32
    ).view(1, 1, 3, 3)

    edge_pred = F.conv2d(probs.unsqueeze(1), lap, padding=1)
    edge_gt = F.conv2d(targets.unsqueeze(1), lap, padding=1)

    return F.l1_loss(edge_pred, edge_gt)


def total_loss(logits, targets, class_weights, boundary_weight=0.2):
    region = weighted_dice_focal_loss(logits, targets, class_weights)
    boundary = boundary_loss(logits, targets)
    return region + boundary_weight * boundary


@torch.no_grad()
def visualize_random_sample(model, dataset, device):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    img, gt = dataset[idx]

    img_t = img.unsqueeze(0).to(device)
    logits = model(img_t)
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    img_np = img[0].cpu().numpy()
    gt_rgb = mask_to_rgb(gt.numpy())
    pred_rgb = mask_to_rgb(pred)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_rgb)
    plt.axis("off")
    plt.show()


@torch.no_grad()
def load_weights_and_show_sample(weights_path, dataset, device=None, in_channels=1, num_classes=4):
    device = torch.device(
        device
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    model = UNetPP(in_channels=in_channels, num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    visualize_random_sample(model, dataset, device)
    return model
