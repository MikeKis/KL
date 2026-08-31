"""Local CIFAR-10 loader for Workplace/CIFAR10.bin + CIFAR10.target.txt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LocalCifar10(Dataset):
    """
    Images: contiguous uint8 frames of size 32*32*3 (HWC, RGB), no label byte.
    Labels: one integer class id per line in target file (60000 lines).
    First 50000 = train, last 10000 = test (project convention).
    """

    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        train: bool,
        transform=None,
        channel_order: str = "hwc",
    ) -> None:
        self.transform = transform
        self.channel_order = channel_order.lower()
        images = np.fromfile(images_path, dtype=np.uint8)
        if images.size != 60000 * 32 * 32 * 3:
            raise ValueError(f"unexpected image file size: {images.size}")
        if self.channel_order == "hwc":
            images = images.reshape(60000, 32, 32, 3)
        elif self.channel_order == "chw":
            images = images.reshape(60000, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            raise ValueError(channel_order)

        labels = np.loadtxt(labels_path, dtype=np.int64)
        if labels.shape != (60000,):
            raise ValueError(f"unexpected label count: {labels.shape}")

        if train:
            self.images = images[:50000]
            self.labels = labels[:50000]
        else:
            self.images = images[50000:]
            self.labels = labels[50000:]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        # PIL-compatible HWC uint8 → torchvision transforms expect PIL or Tensor
        image = self.images[index]
        label = int(self.labels[index])
        if self.transform is not None:
            from PIL import Image

            image = self.transform(Image.fromarray(image, mode="RGB"))
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label
