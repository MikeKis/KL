# Spec: 2026-09-10_ann-to-arni-snn.md appendix D
"""Fold CIFAR mean/std into the first Conv so fromFile can convolve raw uint8."""

from __future__ import annotations

import numpy as np

from .ann_graph import ConverterError


def fold_first_conv_to_uint8(
    weight: np.ndarray,
    bias: np.ndarray,
    mean: list[float] | tuple[float, ...] | np.ndarray,
    std: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ANN: y = W * ((x_u8/255 - mean) / std) + b
    Equivalent uint8 conv: y = W' * x_u8 + b'
      W'[f,c,i,j] = W[f,c,i,j] / (255 * std[c])
      b'[f] = b[f] - sum_{c,i,j} W[f,c,i,j] * mean[c] / std[c]
    weight shape: [out, in, kH, kW]
    """
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if w.ndim != 4:
        raise ConverterError(f"first conv weight must be 4D, got {w.shape}")
    n_out, n_in, _, _ = w.shape
    mean_a = np.asarray(mean, dtype=np.float64).reshape(-1)
    std_a = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean_a.size != n_in or std_a.size != n_in:
        raise ConverterError(
            f"mean/std length {mean_a.size}/{std_a.size} != in_channels {n_in}"
        )
    if np.any(std_a == 0):
        raise ConverterError("std contains zeros")
    if b.shape != (n_out,):
        raise ConverterError(f"first conv bias shape {b.shape} != ({n_out},)")
    w_u8 = w / (255.0 * std_a.reshape(1, n_in, 1, 1))
    b_u8 = b - np.einsum("fcij,c->f", w, mean_a / std_a)
    return w_u8, b_u8
