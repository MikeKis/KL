# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

from pathlib import Path

ANN_DIR = Path(__file__).resolve().parents[2] / "CIFAR-ANN-SNN" / "artifacts"
ARCH = ANN_DIR / "architecture.json"
WEIGHTS = ANN_DIR / "weights_dump.txt"

SHORT_ARCH = {
    "name": "ShortNet",
    "input": {"shape": [3, 16, 16], "layout": "NCHW"},
    "preprocessing": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "layers": [
        {
            "name": "stem",
            "type": "Conv2d",
            "in_channels": 3,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "bias": True,
            "activation": "ReLU",
        },
        {
            "name": "block",
            "type": "Conv2d",
            "in_channels": 4,
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "bias": True,
            "activation": "ReLU",
        },
        {"name": "gap", "type": "AdaptiveAvgPool2d", "output_size": [1, 1]},
        {"name": "flatten", "type": "Flatten"},
        {
            "name": "head",
            "type": "Linear",
            "in_features": 8,
            "out_features": 4,
            "bias": True,
        },
    ],
}

PADDED_ARCH = {
    "name": "PaddedNet",
    "input": {"shape": [3, 32, 32], "layout": "NCHW"},
    "preprocessing": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "layers": [
        {
            "name": "conv1",
            "type": "Conv2d",
            "in_channels": 3,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "bias": True,
        },
        {"name": "flatten", "type": "Flatten"},
        {"name": "fc", "type": "Linear", "in_features": 4, "out_features": 10, "bias": True},
    ],
}

UNTILED_ARCH = {
    "name": "UntiledPool",
    "input": {"shape": [3, 32, 32], "layout": "NCHW"},
    "preprocessing": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "layers": [
        {
            "name": "conv1",
            "type": "Conv2d",
            "in_channels": 3,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "bias": True,
        },
        {"name": "pool", "type": "AvgPool2d", "kernel_size": 3, "stride": 2},
        {"name": "flatten", "type": "Flatten"},
        {"name": "fc", "type": "Linear", "in_features": 4, "out_features": 10, "bias": True},
    ],
}


def write_dump(path: Path, arrays: dict) -> None:
    import numpy as np

    with path.open("w", encoding="utf-8") as f:
        f.write("# test dump\n")
        for key in sorted(arrays):
            arr = np.asarray(arrays[key], dtype=np.float64)
            f.write(f"\n[{key}] shape={list(arr.shape)} dtype=float64\n")
            for v in arr.reshape(-1):
                f.write(f"{v:.8g}\n")
