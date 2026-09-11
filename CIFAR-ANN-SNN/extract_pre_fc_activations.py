"""Export 40-D activations before the FC classifier for all CIFAR-10 images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model import TinyCifarNet
from train import CIFAR10_MEAN, CIFAR10_STD, DEFAULT_LOCAL_IMAGES, DEFAULT_LOCAL_LABELS


class AllLocalCifar10(Dataset):
    """All 60 000 images in file order (train 0..49999, test 50000..59999)."""

    def __init__(self, images_path: Path, labels_path: Path, transform, channel_order: str = "chw"):
        images = np.fromfile(images_path, dtype=np.uint8)
        if channel_order == "chw":
            images = images.reshape(60000, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            images = images.reshape(60000, 32, 32, 3)
        self.images = images
        self.labels = np.loadtxt(labels_path, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return 60000

    def __getitem__(self, index: int):
        image = self.transform(Image.fromarray(self.images[index], mode="RGB"))
        return image, int(self.labels[index])


@torch.no_grad()
def extract_activations(
    checkpoint: Path,
    images_path: Path,
    labels_path: Path,
    out_csv: Path,
    batch_size: int = 512,
    channel_order: str = "chw",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TinyCifarNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = AllLocalCifar10(images_path, labels_path, transform=tf, channel_order=channel_order)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    rows: list[np.ndarray] = []
    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        feats = model.features(inputs)
        feats = torch.flatten(feats, 1)  # (N, 40)
        rows.append(feats.cpu().numpy())

    activations = np.concatenate(rows, axis=0)
    assert activations.shape == (60000, 40), activations.shape
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"n{i}" for i in range(40))
    np.savetxt(out_csv, activations, delimiter=",", header=header, comments="", fmt="%.8g")
    print(f"wrote {out_csv} shape={activations.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument("--local-images", type=Path, default=DEFAULT_LOCAL_IMAGES)
    parser.add_argument("--local-labels", type=Path, default=DEFAULT_LOCAL_LABELS)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pre_fc_activations.csv"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--channel-order", choices=("hwc", "chw"), default="chw")
    args = parser.parse_args()
    extract_activations(
        args.checkpoint,
        args.local_images,
        args.local_labels,
        args.out,
        batch_size=args.batch_size,
        channel_order=args.channel_order,
    )


if __name__ == "__main__":
    main()
