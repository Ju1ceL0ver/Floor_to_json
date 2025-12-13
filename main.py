import kagglehub
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import utils
from model import UNetPP

path = kagglehub.dataset_download("dimka11/segmentation-of-floor-plans")
root = Path(os.path.join(path, "Crimea_dataset"))

mean = 0.8954244
std = 0.23227668

img_transform = v2.Compose(
    [
        v2.Resize((320, 320)),
        v2.ToTensor(),
        v2.Normalize(mean=[mean], std=[std]),
    ]
)

mask_transform = v2.Compose(
    [
        v2.Resize((320, 320), interpolation=InterpolationMode.NEAREST),
    ]
)

dataset = utils.SegmentationDataset(
    root / "images",
    root / "annotations",
    img_transform,
    mask_transform,
)

n = len(dataset)
train_ds, val_ds = random_split(dataset, [int(0.9 * n), n - int(0.9 * n)])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class_weights = torch.tensor([0.0060, 0.0632, 0.6359, 0.2949]).to(device)

model = UNetPP(in_channels=1, num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = 1e9

for epoch in range(1, 21):
    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Train {epoch}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = utils.total_loss(model(x), y, class_weights)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Val {epoch}"):
            x, y = x.to(device), y.to(device)
            val_loss += utils.total_loss(model(x), y, class_weights).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
