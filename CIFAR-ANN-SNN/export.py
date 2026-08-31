"""Export TinyCifarNet with BN folded into Conv for easy ANN→SNN extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from model import TinyCifarNet, architecture_spec


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Absorb BatchNorm into Conv weights/bias (standard inference folding)."""
    w = conv.weight.detach().cpu().clone()
    if conv.bias is not None:
        b = conv.bias.detach().cpu().clone()
    else:
        b = torch.zeros(conv.out_channels)

    mean = bn.running_mean.detach().cpu()
    var = bn.running_var.detach().cpu()
    eps = bn.eps
    gamma = bn.weight.detach().cpu()
    beta = bn.bias.detach().cpu()

    std = torch.sqrt(var + eps)
    scale = gamma / std
    w_fused = w * scale.reshape(-1, 1, 1, 1)
    b_fused = beta + (b - mean) * scale
    return w_fused, b_fused


def fold_and_extract(model: TinyCifarNet) -> dict[str, np.ndarray]:
    """
    Walk features as Conv→BN→ReLU [→Conv→BN→ReLU] → Pool sequences
    and emit folded weight tensors keyed by architecture_spec layer names.
    """
    model.eval()
    arrays: dict[str, np.ndarray] = {}
    conv_names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    conv_idx = 0
    modules = list(model.features.children())
    i = 0
    while i < len(modules):
        mod = modules[i]
        if isinstance(mod, nn.Conv2d):
            nxt = modules[i + 1] if i + 1 < len(modules) else None
            if not isinstance(nxt, nn.BatchNorm2d):
                raise RuntimeError(f"Expected BatchNorm after Conv at index {i}")
            name = conv_names[conv_idx]
            w, b = _fuse_conv_bn(mod, nxt)
            arrays[f"{name}.weight"] = w.numpy()
            arrays[f"{name}.bias"] = b.numpy()
            conv_idx += 1
            i += 2  # skip BN; ReLU/Pool handled structurally
            continue
        i += 1

    arrays["fc.weight"] = model.classifier.weight.detach().cpu().numpy()
    arrays["fc.bias"] = model.classifier.bias.detach().cpu().numpy()
    return arrays


def param_count_from_arrays(arrays: dict[str, np.ndarray]) -> int:
    return int(sum(v.size for v in arrays.values()))


def export_bundle(
    checkpoint_path: Path,
    out_dir: Path,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyCifarNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    arrays = fold_and_extract(model)
    n_params = param_count_from_arrays(arrays)

    spec = architecture_spec()
    spec["parameter_budget"]["exported_folded_params"] = n_params
    if metrics:
        spec["metrics"] = metrics
    elif "metrics" in ckpt:
        spec["metrics"] = ckpt["metrics"]

    weights_path = out_dir / "weights.npz"
    np.savez(weights_path, **arrays)

    arch_path = out_dir / "architecture.json"
    arch_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # Flat text dump: easy to parse without NumPy if needed
    dump_path = out_dir / "weights_dump.txt"
    with dump_path.open("w", encoding="utf-8") as f:
        f.write(f"# folded parameter tensors, total={n_params}\n")
        for key in sorted(arrays):
            arr = arrays[key]
            f.write(f"\n[{key}] shape={list(arr.shape)} dtype={arr.dtype}\n")
            np.savetxt(f, arr.reshape(-1), fmt="%.8g")

    summary = {
        "folded_params": n_params,
        "within_budget": n_params <= 30000,
        "files": {
            "architecture": str(arch_path),
            "weights_npz": str(weights_path),
            "weights_dump": str(dump_path),
            "checkpoint": str(checkpoint_path),
        },
        "tensor_shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export folded TinyCifarNet artifacts")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
    )
    args = parser.parse_args()
    summary = export_bundle(args.checkpoint, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
