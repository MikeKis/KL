# Spec: 2026-09-13_layerwise-from-colanet.md
"""meanjaccard / discretize from SpikingApproximation-1fba1395.cpp."""

from __future__ import annotations

import numpy as np

RELDIFTHR = 1  # neuron is "active" if spike count > 1


def discretize(activations: np.ndarray, saturation_level: float) -> np.ndarray:
    """Integer spike counts: clip(value/sat * 10, 0, 10) truncated toward zero."""
    act = np.asarray(activations, dtype=np.float64)
    if saturation_level <= 0:
        raise ValueError("saturation_level must be positive")
    scaled = act / float(saturation_level) * 10.0
    counts = np.zeros(act.shape, dtype=np.int32)
    counts[scaled >= 10.0] = 10
    mid = (scaled > 0.0) & (scaled < 10.0)
    counts[mid] = scaled[mid].astype(np.int32)
    return counts


def meanjaccard(left: np.ndarray, right: np.ndarray, *, reldifthr: int = RELDIFTHR) -> float:
    """
    Mean over rows of |A∩B|/|A∪B| where membership is count > reldifthr.
    Empty union contributes 0. Matches C++ meanjaccard.
    """
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        raise ValueError(f"count table shape mismatch {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError("count tables must be 2-D with at least one row")
    total = 0.0
    thr = int(reldifthr)
    for row in range(a.shape[0]):
        has_l = a[row] > thr
        has_r = b[row] > thr
        inter = int(np.count_nonzero(has_l & has_r))
        union = int(np.count_nonzero(has_l | has_r))
        total += 0.0 if union == 0 else inter / union
    return total / float(a.shape[0])


def meanjaccard_as_code(left: np.ndarray, right: np.ndarray) -> int:
    """C++ tool returned int(metric * 10000)."""
    return int(meanjaccard(left, right) * 10000)
