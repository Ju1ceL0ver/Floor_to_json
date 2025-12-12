import kagglehub
import os
from pathlib import Path
import utils
import torch
from model import UNet
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import random_split, DataLoader

path = kagglehub.dataset_download("dimka11/segmentation-of-floor-plans")


root = Path(os.path.join(path, "Crimea_dataset"))
imgs_path = [os.path.join(root, "images", i) for i in os.listdir(root / "images")]
masks_path = [
    os.path.join(root, "annotations", i) for i in os.listdir(root / "annotations")
]

# mean, std = utils.compute_mean_std_gray_from_paths(imgs_path)
mean = 0.8954244
std = 0.23227668

print(f"Mean: {mean}, Std: {std}")


img_transform = v2.Compose(
    [
        v2.Resize((512, 512)),
        v2.ToTensor(),
        v2.Normalize(
            mean=[mean],
            std=[std],
        ),
    ]
)

mask_transform = v2.Compose(
    [
        v2.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
    ]
)


dataset = utils.SegmentationDataset(
    root / "images",
    root / "annotations",
    img_transform=img_transform,
    mask_transform=mask_transform,
)
n = len(dataset)

train_size = int(0.9 * n)
test_size = n - train_size

train_ds, test_ds = random_split(
    dataset,
    [train_size, test_size],
)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=2,
    shuffle=False,
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# weights = utils.compute_class_weights(train_loader, num_classes=4)
weights = torch.tensor([0.0060, 0.0632, 0.6359, 0.2949])
print(weights)

class_weights = weights.to(device)

model = UNet(num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
best_val_loss = float("inf")
best_model_path = Path("best_model.pth")

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for imgs, masks in tqdm(train_loader, desc="Training"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = utils.weighted_dice_focal_loss(logits, masks, class_weights)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    train_loss = sum(train_losses) / len(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Validation"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = utils.weighted_dice_focal_loss(logits, masks, class_weights)
            val_losses.append(loss.item())

    val_loss = sum(val_losses) / len(val_losses)

    print(
        f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss:.4f} - "
        f"Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model to {best_model_path} (val loss {val_loss:.4f})")
