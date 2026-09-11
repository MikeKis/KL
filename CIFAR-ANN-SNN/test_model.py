"""Quick sanity checks: param budget and export round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export import export_bundle, fold_and_extract, param_count_from_arrays
from model import TinyCifarNet


def test_folded_param_budget() -> None:
    model = TinyCifarNet()
    n = model.count_folded_parameters()
    assert n <= 30000, n
    assert n == 28626, n


def test_forward_shape() -> None:
    model = TinyCifarNet()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_valid_spatial_sizes() -> None:
    """32→30→28→14→12→10→5→3→1; every Conv/Pool window tiles (padding=0)."""
    model = TinyCifarNet()
    x = torch.randn(1, 3, 32, 32)
    expected = [30, 28, 14, 12, 10, 5, 3, 1]
    spatial = []
    for mod in model.features:
        x = mod(x)
        if isinstance(mod, (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            spatial.append(x.shape[-1])
    assert spatial == expected, spatial
    assert torch.flatten(x, 1).shape == (1, 40)


def test_export_arrays() -> None:
    model = TinyCifarNet()
    arrays = fold_and_extract(model)
    assert param_count_from_arrays(arrays) == 28626
    assert arrays["conv1.weight"].shape == (16, 3, 3, 3)
    assert arrays["fc.weight"].shape == (10, 40)


def test_export_bundle_files() -> None:
    model = TinyCifarNet()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = tmp_path / "best.pt"
        torch.save({"model_state_dict": model.state_dict(), "metrics": {"dummy": 1}}, ckpt)
        summary = export_bundle(ckpt, tmp_path / "out")
        assert summary["within_budget"]
        with np.load(tmp_path / "out" / "weights.npz") as data:
            assert set(data.files) == {
                "conv1.weight",
                "conv1.bias",
                "conv2.weight",
                "conv2.bias",
                "conv3.weight",
                "conv3.bias",
                "conv4.weight",
                "conv4.bias",
                "conv5.weight",
                "conv5.bias",
                "fc.weight",
                "fc.bias",
            }


if __name__ == "__main__":
    test_folded_param_budget()
    test_forward_shape()
    test_valid_spatial_sizes()
    test_export_arrays()
    test_export_bundle_files()
    print("all tests passed")
