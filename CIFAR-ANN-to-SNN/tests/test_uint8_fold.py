# Spec: 2026-09-10_ann-to-arni-snn.md appendix D
from __future__ import annotations

import numpy as np

from snn_convert.uint8_fold import fold_first_conv_to_uint8


def _conv2d_numpy(x, w, b):
    """Valid conv, NCHW, x float [N,C,H,W], w [F,C,K,K]."""
    n, c, h, w_in = x.shape
    f, _, k, _ = w.shape
    out_h = h - k + 1
    out_w = w_in - k + 1
    y = np.zeros((n, f, out_h, out_w), dtype=np.float64)
    for n_i in range(n):
        for f_i in range(f):
            for y_i in range(out_h):
                for x_i in range(out_w):
                    patch = x[n_i, :, y_i : y_i + k, x_i : x_i + k]
                    y[n_i, f_i, y_i, x_i] = np.sum(patch * w[f_i]) + b[f_i]
    return y


def test_uint8_fold_matches_normalized_conv():
    rng = np.random.default_rng(2)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.2435, 0.2616])
    w = rng.normal(size=(5, 3, 3, 3))
    b = rng.normal(size=(5,))
    x_u8 = rng.integers(0, 256, size=(2, 3, 8, 8), dtype=np.uint8)

    x_norm = (x_u8.astype(np.float64) / 255.0 - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    y_ref = _conv2d_numpy(x_norm, w, b)
    w_u8, b_u8 = fold_first_conv_to_uint8(w, b, mean, std)
    y_u8 = _conv2d_numpy(x_u8.astype(np.float64), w_u8, b_u8)
    np.testing.assert_allclose(y_u8, y_ref, rtol=1e-10, atol=1e-10)
