# Spec: 2026-09-14_precomputed-layer-activations.md
"""Export 40-D pre-FC features and intermediate maps (all layers except conv1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model import TinyCifarNet
from train import CIFAR10_MEAN, CIFAR10_STD, DEFAULT_LOCAL_IMAGES, DEFAULT_LOCAL_LABELS

SAVE_LAYERS = ("conv2", "pool1", "conv3", "conv4", "pool2", "conv5", "gap")
SKIP_LAYERS = ("conv1",)
N_CIFAR = 60000


class AllLocalCifar10(Dataset):
    """All 60 000 images in file order (train 0..49999, test 50000..59999)."""

    def __init__(self, images_path: Path, labels_path: Path, transform, channel_order: str = "chw"):
        images = np.fromfile(images_path, dtype=np.uint8)
        if channel_order == "chw":
            images = images.reshape(N_CIFAR, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            images = images.reshape(N_CIFAR, 32, 32, 3)
        self.images = images
        self.labels = np.loadtxt(labels_path, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return N_CIFAR

    def __getitem__(self, index: int):
        image = self.transform(Image.fromarray(self.images[index], mode="RGB"))
        return image, int(self.labels[index])


@torch.no_grad()
def capture_named_feature_maps(model: TinyCifarNet, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Maps after ReLU (convs), AvgPool, and GAP. Keys match architecture.json."""
    conv_names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    pool_names = ["pool1", "pool2"]
    conv_i = 0
    pool_i = 0
    maps: dict[str, torch.Tensor] = {}
    modules = list(model.features.children())
    y = x
    i = 0
    while i < len(modules):
        mod = modules[i]
        if isinstance(mod, nn.Conv2d):
            y = modules[i + 1](mod(y))
            y = modules[i + 2](y)
            maps[conv_names[conv_i]] = y
            conv_i += 1
            i += 3
            continue
        if isinstance(mod, nn.AvgPool2d):
            y = mod(y)
            maps[pool_names[pool_i]] = y
            pool_i += 1
            i += 1
            continue
        if isinstance(mod, nn.AdaptiveAvgPool2d):
            y = mod(y)
            maps["gap"] = y
            i += 1
            continue
        y = mod(y)
        i += 1
    return maps


@torch.no_grad()
def extract_activations(
    checkpoint: Path,
    images_path: Path,
    labels_path: Path,
    out_csv: Path,
    activations_dir: Path,
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

    activations_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, np.memmap] = {}
    offset = 0
    pre_fc_rows: list[np.ndarray] = []
    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        maps = capture_named_feature_maps(model, inputs)
        n = int(inputs.shape[0])
        gap = maps["gap"]
        pre_fc_rows.append(torch.flatten(gap, 1).cpu().numpy())
        for name in SAVE_LAYERS:
            arr = maps[name].detach().cpu().numpy().astype(np.float32, copy=False)
            if name not in writers:
                path = activations_dir / f"{name}.npy"
                writers[name] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(N_CIFAR,) + arr.shape[1:]
                )
            writers[name][offset : offset + n] = arr
        offset += n
        print(f"extract {offset}/{N_CIFAR}", flush=True)

    assert offset == N_CIFAR, offset
    for mm in writers.values():
        mm.flush()

    activations = np.concatenate(pre_fc_rows, axis=0)
    assert activations.shape == (N_CIFAR, 40), activations.shape
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"n{i}" for i in range(40))
    np.savetxt(out_csv, activations, delimiter=",", header=header, comments="", fmt="%.8g")
    print(f"wrote {out_csv} shape={activations.shape}")

    manifest = {
        "n_images": N_CIFAR,
        "order": "CIFAR10.bin train 0..49999, test 50000..59999",
        "layout": "NCHW",
        "dtype": "float32",
        "skip": list(SKIP_LAYERS),
        "layers": {
            name: {
                "file": f"{name}.npy",
                "shape": list(writers[name].shape),
            }
            for name in SAVE_LAYERS
        },
        "pre_fc_csv": str(out_csv),
    }
    (activations_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {activations_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument("--local-images", type=Path, default=DEFAULT_LOCAL_IMAGES)
    parser.add_argument("--local-labels", type=Path, default=DEFAULT_LOCAL_LABELS)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pre_fc_activations.csv"))
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=Path("artifacts/activations"),
        help="NCHW float32 maps for conv2..gap (not conv1)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--channel-order", choices=("hwc", "chw"), default="chw")
    args = parser.parse_args()
    extract_activations(
        args.checkpoint,
        args.local_images,
        args.local_labels,
        args.out,
        args.activations_dir,
        batch_size=args.batch_size,
        channel_order=args.channel_order,
    )


if __name__ == "__main__":
    main()
