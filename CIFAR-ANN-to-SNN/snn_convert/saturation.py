# Spec: 2026-09-13_layerwise-from-colanet.md
"""Saturation start: q that maximises spar(N(0,1))/spar(A), as in CIFAR/csv_sparsity_analysis.py."""

from __future__ import annotations

import math

import numpy as np
from statistics import NormalDist

_STANDARD_NORMAL = NormalDist()
_SQRT_TWO = math.sqrt(2.0)
_Q_VALUES = np.arange(0.0001, 0.5 + 1e-12, 0.0001)
_N_LEVELS = 10


def _normal_sparsity(q: float, n: int = _N_LEVELS) -> float:
    if q == 0.5:
        return 0.5
    v = -_STANDARD_NORMAL.inv_cdf(q)
    tail_sum = 0.0
    for k in range(1, n + 1):
        tail_sum += 0.5 * math.erfc((k * v / n) / _SQRT_TWO)
    return tail_sum / n


_NORMAL = np.array([_normal_sparsity(float(q)) for q in _Q_VALUES], dtype=np.float64)


def saturation_from_sparsity_ratio(values: np.ndarray) -> tuple[float, dict]:
    """
    value_A at the q maximising spar(N(0,1); q) / spar(A; q).
    That activation level is the rate-code saturation start.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0, {"q": None, "value_a": 1.0, "ratio": None}
    sorted_values = np.sort(flat)
    n = int(sorted_values.size)
    positive_fraction = float(np.count_nonzero(flat > 0.0) / n)
    tail_counts = np.clip(np.ceil(_Q_VALUES * n).astype(np.int64), 1, n)
    thresholds = sorted_values[n - tail_counts]
    sparsity = np.empty(_Q_VALUES.shape, dtype=np.float64)
    non_positive = thresholds <= 0.0
    sparsity[non_positive] = positive_fraction
    positive_thr = thresholds[~non_positive]
    if positive_thr.size:
        tail_sum = np.zeros(positive_thr.shape, dtype=np.float64)
        for level in range(1, _N_LEVELS + 1):
            left = np.searchsorted(sorted_values, positive_thr * (level / _N_LEVELS), side="left")
            tail_sum += n - left
        sparsity[~non_positive] = tail_sum / (_N_LEVELS * n)
    ratio = np.full(_Q_VALUES.shape, np.inf, dtype=np.float64)
    ok = sparsity > 0.0
    ratio[ok] = _NORMAL[ok] / sparsity[ok]
    best = int(np.argmax(ratio))
    value_a = float(thresholds[best])
    if not math.isfinite(value_a) or value_a <= 0.0:
        pos = flat[flat > 0.0]
        value_a = float(pos.max()) if pos.size else 1.0
    return value_a, {
        "q": float(_Q_VALUES[best]),
        "value_a": value_a,
        "ratio": float(ratio[best]) if math.isfinite(ratio[best]) else None,
        "spar_a": float(sparsity[best]),
    }
