"""Tiny CIFAR-10 CNN sized for ANN→SNN conversion (~29k folded params)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TinyCifarNet(nn.Module):
    """
    Compact VGG-style stack: ReLU + AvgPool, BatchNorm (folded at export).

    Layout (spatial size after each stage):
      32x32 → [C16, C16, P] → 16x16 → [C32, C32, P] → 8x8 → [C40, P] → 4x4
      → AdaptiveAvgPool → Linear(40, 10)

    Folded parameter budget (weights+bias after absorbing BN into Conv):
      ≈ 28 626 < 30 000.
    """

    NUM_CLASSES = 10

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            # Block 3
            nn.Conv2d(32, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(40, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_folded_parameters(self) -> int:
        """Params of the inference graph after BN is absorbed into Conv."""
        total = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                out_ch = module.out_channels
                total += module.weight.numel() + out_ch  # folded bias
            elif isinstance(module, nn.Linear):
                total += module.weight.numel()
                if module.bias is not None:
                    total += module.bias.numel()
        return total


def architecture_spec() -> dict[str, Any]:
    """Human/machine-readable description of the convertible inference graph."""
    return {
        "name": "TinyCifarNet",
        "dataset": "CIFAR-10",
        "input": {"shape": [3, 32, 32], "layout": "NCHW", "dtype": "float32"},
        "preprocessing": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "note": "Local Workplace/CIFAR10.bin is CHW uint8 planes; normalize with these stats after /255.",
        },
        "notes": [
            "Activations are ReLU — suitable for ANN→SNN conversion.",
            "Pooling is average (AvgPool2d) — friendlier for rate-based conversion than MaxPool.",
            "BatchNorm is used only during training; export folds it into preceding Conv (bias=True).",
            "Classifier logits are returned; apply ArgMax for class id (no Softmax required for inference).",
        ],
        "layers": [
            {"name": "conv1", "type": "Conv2d", "in_channels": 3, "out_channels": 16,
             "kernel_size": 3, "stride": 1, "padding": 1, "bias": True, "activation": "ReLU"},
            {"name": "conv2", "type": "Conv2d", "in_channels": 16, "out_channels": 16,
             "kernel_size": 3, "stride": 1, "padding": 1, "bias": True, "activation": "ReLU"},
            {"name": "pool1", "type": "AvgPool2d", "kernel_size": 2, "stride": 2},
            {"name": "conv3", "type": "Conv2d", "in_channels": 16, "out_channels": 32,
             "kernel_size": 3, "stride": 1, "padding": 1, "bias": True, "activation": "ReLU"},
            {"name": "conv4", "type": "Conv2d", "in_channels": 32, "out_channels": 32,
             "kernel_size": 3, "stride": 1, "padding": 1, "bias": True, "activation": "ReLU"},
            {"name": "pool2", "type": "AvgPool2d", "kernel_size": 2, "stride": 2},
            {"name": "conv5", "type": "Conv2d", "in_channels": 32, "out_channels": 40,
             "kernel_size": 3, "stride": 1, "padding": 1, "bias": True, "activation": "ReLU"},
            {"name": "pool3", "type": "AvgPool2d", "kernel_size": 2, "stride": 2},
            {"name": "gap", "type": "AdaptiveAvgPool2d", "output_size": [1, 1]},
            {"name": "flatten", "type": "Flatten"},
            {"name": "fc", "type": "Linear", "in_features": 40, "out_features": 10, "bias": True},
        ],
        "parameter_budget": {
            "folded_weights_and_biases": 28626,
            "limit": 30000,
        },
    }
