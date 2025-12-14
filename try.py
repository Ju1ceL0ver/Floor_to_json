import argparse
import json
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.transforms import v2

import utils
import postprocess
from model import UNetPP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Floor plan inference -> JSON + visualization"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image (PNG/JPG)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    weights_path = Path("best_model.pth")
    if not weights_path.exists():
        raise FileNotFoundError("best_model.pth not found")

    mean = 0.8954244
    std = 0.23227668
    size = (320, 320)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    img_transform = v2.Compose([
        v2.Resize(size),
        v2.ToTensor(),
        v2.Normalize(mean=[mean], std=[std]),
    ])

    img = Image.open(img_path).convert("L")
    img_resized = img.resize(size)
    img_np = np.array(img_resized)

    img_t = img_transform(img).unsqueeze(0).to(device)

    model = UNetPP(in_channels=1, num_classes=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(img_t)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    uniq, cnt = np.unique(pred, return_counts=True)
    print("Pred classes:", dict(zip(uniq.tolist(), cnt.tolist())))

    json_data = postprocess.build_json(
        pred,
        image_name=img_path.name,
    )

    json_path = Path.cwd() / f"{img_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to: {json_path}")

    original = img_np
    seg = utils.mask_to_rgb(pred)

    post = img_np.copy()
    if post.ndim == 2:
        post = np.stack([post] * 3, axis=-1)

    for w in json_data.get("walls", []):
        p1, p2 = w["points"]
        cv2.line(post, tuple(p1), tuple(p2), (255, 0, 0), 2)

    for d in json_data.get("doors", []):
        p1, p2 = d["points"]
        cv2.line(post, tuple(p1), tuple(p2), (0, 0, 255), 3)

    for win in json_data.get("windows", []):
        p1, p2 = win["points"]
        cv2.line(post, tuple(p1), tuple(p2), (0, 255, 0), 3)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmentation")
    plt.imshow(seg)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Postprocess (walls/doors/windows)")
    plt.imshow(post)
    plt.axis("off")

    vis_path = Path.cwd() / f"{img_path.stem}_viz.png"
    plt.savefig(vis_path, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Visualization saved to: {vis_path}")

    print("\n--- JSON OUTPUT ---")
    print(json.dumps(json_data, indent=2))


if __name__ == "__main__":
    main()
