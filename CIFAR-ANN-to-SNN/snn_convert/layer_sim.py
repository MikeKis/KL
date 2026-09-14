# Spec: 2026-09-13_layerwise-from-colanet.md
"""Python spike-layer sim for Jaccard step 1 (SumPool + LIF conv)."""

from __future__ import annotations

import numpy as np

from .ann_forward import conv2d_valid
from .conversion_formulas import ARNI_SYNAPSE_SCALE, ARNI_THRESHOLD_BASE, DEFAULT_CHARTIME


def trains_to_nchw(trains: np.ndarray, h: int, w: int, c: int) -> np.ndarray:
    """(N, T, H*W*C NHWC) → (N, T, C, H, W)."""
    n, t, d = trains.shape
    if d != h * w * c:
        raise ValueError(f"train width {d} != {h}*{w}*{c}")
    return trains.reshape(n, t, h, w, c).transpose(0, 1, 4, 2, 3)


def nchw_trains_to_rows(trains_nchw: np.ndarray) -> np.ndarray:
    """(N, T, C, H, W) → (N, T, H*W*C NHWC)."""
    n, t, c, h, w = trains_nchw.shape
    return trains_nchw.transpose(0, 1, 3, 4, 2).reshape(n, t, h * w * c)


def sumpool_trains(
    trains: np.ndarray,
    in_h: int,
    in_w: int,
    channels: int,
    kernel: int,
    stride: int,
    *,
    threshold_base: float = ARNI_THRESHOLD_BASE,
) -> np.ndarray:
    """
    SumPool LIF: each window input spike adds THRESHOLD+1; on fire, potential
    is decremented by the threshold (not zeroed). One spike per tact, leftover
    can fire on later tacts of the same presentation. Charge still above
    threshold after the last tact is lost (end of the image).
    """
    x = trains_to_nchw(np.asarray(trains), in_h, in_w, channels)
    n, tpres, c, _h, _w = x.shape
    k = int(kernel)
    s = int(stride)
    out_h = (in_h - k) // s + 1
    out_w = (in_w - k) // s + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("sumpool produced empty map")
    w_syn = float(threshold_base) + 1.0
    thr = float(threshold_base)
    out = np.zeros((n, tpres, c, out_h, out_w), dtype=bool)
    for oy in range(out_h):
        for ox in range(out_w):
            patch = x[:, :, :, oy * s : oy * s + k, ox * s : ox * s + k]
            incoming = patch.reshape(n, tpres, c, -1).sum(axis=3).astype(np.float64)
            potential = np.zeros((n, c), dtype=np.float64)
            for t in range(tpres):
                potential = potential + incoming[:, t] * w_syn
                fired = potential > thr
                out[:, t, :, oy, ox] = fired
                potential = potential - fired.astype(np.float64) * thr
    return nchw_trains_to_rows(out)


def _decay_shift(chartime: int) -> int:
    ct = int(chartime)
    if ct <= 0:
        return 0
    if ct > 1024:
        return -1
    return 1024 - 1024 // ct


def conv_lif_counts(
    trains: np.ndarray,
    in_h: int,
    in_w: int,
    in_c: int,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    weight_scale: float,
    bias_scale: float,
    chartime: int = DEFAULT_CHARTIME,
    threshold_base: float = ARNI_THRESHOLD_BASE,
    synapse_scale: float = ARNI_SYNAPSE_SCALE,
) -> np.ndarray:
    """
    Deterministic LIF: millival synapses, leak as NeuLIF, one spike/tact.
    On fire, potential is decremented by the threshold (not reset to 0).
    Positive bias → mean stochastic current = bias * bias_scale (S/2 with S=2*stim).
    Negative bias → ThresholdExcess = |stim| * chartime.
    """
    w = np.asarray(weight, dtype=np.float64)
    millival = np.rint(w * float(weight_scale) * float(synapse_scale))
    x = trains_to_nchw(np.asarray(trains), in_h, in_w, in_c)
    n, tpres, _c, _h, _w = x.shape
    xt = x.reshape(n * tpres, in_c, in_h, in_w).astype(np.float64)
    current = conv2d_valid(xt, millival, None, stride=1)
    _nt, n_f, out_h, out_w = current.shape
    current = current.reshape(n, tpres, n_f, out_h, out_w)

    stim = np.zeros(n_f, dtype=np.float64)
    excess = np.zeros(n_f, dtype=np.float64)
    if bias is not None:
        b = np.asarray(bias, dtype=np.float64).reshape(-1)
        if b.size != n_f:
            raise ValueError(f"bias {b.size} != filters {n_f}")
        raw = b * float(bias_scale)
        pos = raw >= 0.0
        stim[pos] = raw[pos]  # mean of Uniform[0, 2*stim]
        excess[~pos] = -raw[~pos] * float(chartime)

    thr = float(threshold_base) + excess
    thr_b = thr.reshape(1, n_f, 1, 1)
    stim_b = stim.reshape(1, n_f, 1, 1)
    decay = _decay_shift(chartime)
    potential = np.zeros((n, n_f, out_h, out_w), dtype=np.float64)
    counts = np.zeros((n, n_f, out_h, out_w), dtype=np.int32)
    for t in range(tpres):
        if decay == 0:
            potential = np.zeros_like(potential)
        elif decay > 0:
            potential = np.trunc(potential * decay / 1024.0)
        potential = potential + current[:, t] + stim_b
        fired = potential > thr_b
        counts += fired.astype(np.int32)
        potential = potential - fired.astype(np.float64) * thr_b
    # NHWC flatten
    return counts.transpose(0, 2, 3, 1).reshape(n, out_h * out_w * n_f)


def trains_to_counts(trains: np.ndarray) -> np.ndarray:
    return np.asarray(trains).sum(axis=1).astype(np.int32)
