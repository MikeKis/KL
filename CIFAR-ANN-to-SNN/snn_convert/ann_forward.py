# Spec: 2026-09-10_ann-to-arni-snn.md
"""Numpy/torch ops for folded ANN inference and the joint rate surrogate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ann_graph import AnnGraph, ConverterError
from .uint8_fold import fold_first_conv_to_uint8

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


def conv2d_valid(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, stride: int = 1) -> np.ndarray:
    """x, weight NCHW / OIHW, valid padding."""
    if _HAS_TORCH:
        xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        wt = torch.as_tensor(np.ascontiguousarray(weight), dtype=torch.float32)
        bt = None if bias is None else torch.as_tensor(np.ascontiguousarray(bias), dtype=torch.float32)
        y = F.conv2d(xt, wt, bt, stride=int(stride), padding=0)
        return y.detach().cpu().numpy().astype(np.float64)
    w = np.asarray(weight, dtype=np.float64)
    b = None if bias is None else np.asarray(bias, dtype=np.float64)
    n, c, h, width = x.shape
    f, cin, k, _ = w.shape
    if cin != c:
        raise ConverterError(f"conv channel mismatch {cin} vs {c}")
    out_h = (h - k) // stride + 1
    out_w = (width - k) // stride + 1
    y = np.zeros((n, f, out_h, out_w), dtype=np.float64)
    for ni in range(n):
        for fi in range(f):
            for oy in range(out_h):
                for ox in range(out_w):
                    patch = x[ni, :, oy * stride : oy * stride + k, ox * stride : ox * stride + k]
                    acc = float(np.sum(patch * w[fi]))
                    if b is not None:
                        acc += float(b[fi])
                    y[ni, fi, oy, ox] = acc
    return y


def avg_pool2d(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    if _HAS_TORCH:
        xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        y = F.avg_pool2d(xt, kernel_size=int(kernel), stride=int(stride), padding=0)
        return y.detach().cpu().numpy().astype(np.float64)
    n, c, h, w = x.shape
    k = int(kernel)
    s = int(stride)
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    y = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    for oy in range(out_h):
        for ox in range(out_w):
            patch = x[:, :, oy * s : oy * s + k, ox * s : ox * s + k]
            y[:, :, oy, ox] = patch.mean(axis=(2, 3))
    return y


def adaptive_avg_pool_1(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=(2, 3), keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def load_cifar_hwc(images_path, n: int | None = None, *, train: bool = True) -> np.ndarray:
    raw = np.fromfile(images_path, dtype=np.uint8)
    if raw.size != 60000 * 32 * 32 * 3:
        raise ConverterError(f"unexpected CIFAR bin size {raw.size}")
    frames = raw.reshape(60000, 32, 32, 3)
    frames = frames[:50000] if train else frames[50000:]
    if n is not None:
        frames = frames[: int(n)]
    return frames


def load_cifar_labels(labels_path, n: int | None = None, *, train: bool = True) -> np.ndarray:
    labels = np.loadtxt(labels_path, dtype=np.int64)
    if labels.shape != (60000,):
        raise ConverterError(f"unexpected label count {labels.shape}")
    labels = labels[:50000] if train else labels[50000:]
    if n is not None:
        labels = labels[: int(n)]
    return labels


def write_cifar_subset(
    images_path,
    labels_path,
    out_images,
    out_labels,
    *,
    n_train: int,
    n_val: int,
    seed: int = 42,
) -> tuple[int, int]:
    """Train-only subset (no CIFAR test): first n_train of a shuffle, then n_val."""
    frames = load_cifar_hwc(images_path, n=None, train=True)
    labels = load_cifar_labels(labels_path, n=None, train=True)
    n_take = int(n_train) + int(n_val)
    if n_take > len(frames):
        raise ConverterError(f"subset {n_take} larger than train {len(frames)}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))[:n_take]
    chosen = frames[idx]
    labs = labels[idx]
    Path(out_images).parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(chosen).tofile(out_images)
    np.savetxt(out_labels, labs, fmt="%d")
    return int(n_train), int(n_val)


def uint8_to_nchw(frames_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(frames_hwc.astype(np.float64), (0, 3, 1, 2))


def first_conv_uint8_preact(graph: AnnGraph, x_nchw_u8: np.ndarray) -> np.ndarray:
    first = graph.layers[0]
    w, b = fold_first_conv_to_uint8(
        graph.weights[f"{first.name}.weight"],
        graph.weights[f"{first.name}.bias"],
        graph.mean,
        graph.std,
    )
    stride = first.get_int("stride", 1)
    return conv2d_valid(x_nchw_u8, w, b, stride=stride)


def ann_forward_maps(graph: AnnGraph, x_nchw_u8: np.ndarray) -> dict[str, np.ndarray]:
    """
    Folded ANN maps after each Conv/Pool/GAP. Conv1 uses the uint8-equivalent
    convolution (same as fromFile), later layers use exported ANN kernels.
    """
    if graph.mean is None or graph.std is None:
        raise ConverterError("mean/std required for uint8 first conv")
    maps: dict[str, np.ndarray] = {}
    first = True
    x = x_nchw_u8
    for layer in graph.layers:
        if layer.type == "Conv2d":
            if first:
                x = first_conv_uint8_preact(graph, x)
                first = False
            else:
                w = graph.weights[f"{layer.name}.weight"]
                b = graph.weights.get(f"{layer.name}.bias")
                x = conv2d_valid(x, w, b, stride=layer.get_int("stride", 1))
            x = relu(x)
            maps[layer.name] = x
        elif layer.type == "AvgPool2d":
            k = layer.kernel_size()
            x = avg_pool2d(x, k, layer.get_int("stride", k))
            maps[layer.name] = x
        elif layer.type == "AdaptiveAvgPool2d":
            x = adaptive_avg_pool_1(x)
            maps[layer.name] = x
        elif layer.type in {"ReLU", "Flatten"}:
            continue
        elif layer.type == "Linear":
            continue
        else:
            raise ConverterError(f"unsupported layer {layer.type}")
    return maps
