# Spec: 2026-09-10_ann-to-arni-snn.md
"""Write fromFile / ConvolutionPooling convolution_file text (*** LAYER 0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ann_graph import ConverterError


def format_kernel_value(value: float) -> str:
    return f"{float(value):.8g}"


def convolution_file_text(weight: np.ndarray) -> str:
    """
    weight: [nFilters, nChannels, K, K] (PyTorch Conv2d layout).

    Line layout matches ParseConvolutionTensorFile / the fromFile unit test:
    each kernel row is groups `(c0,c1,...)` per x, with a space after every `)`
    including the last group on the line.
    """
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 4 or w.shape[2] != w.shape[3]:
        raise ConverterError(f"convolution_file: expected [F,C,K,K], got {w.shape}")
    n_f, n_c, k, _ = w.shape
    lines = ["*** LAYER 0"]
    for f in range(n_f):
        for y in range(k):
            groups: list[str] = []
            for x in range(k):
                inner = ",".join(format_kernel_value(w[f, c, y, x]) for c in range(n_c))
                groups.append(f"({inner})")
            lines.append(" ".join(groups) + " ")
        if f != n_f - 1:
            # ParseConvolutionTensorFile looks ahead one line after each square
            # kernel, then reads another line (blank or "*** LAYER ").
            lines.append("")
    return "\n".join(lines) + "\n"


def write_convolution_file(path: Path, weight: np.ndarray) -> None:
    path.write_text(convolution_file_text(weight), encoding="utf-8")
