# Spec: 2026-09-13_layerwise-from-colanet.md
"""Replica of fromFile GetSpikesfromTextValuesFileRateCoded (absolute saturation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_TPRES = 10


def maps_to_rows(nchw: np.ndarray) -> np.ndarray:
    """N,C,H,W → N, H*W*C with spatial_index (y * W + x) * C + c."""
    arr = np.asarray(nchw)
    if arr.ndim != 4:
        raise ValueError(f"expected NCHW, got {arr.shape}")
    n, _c, h, w = arr.shape
    return np.transpose(arr, (0, 2, 3, 1)).reshape(n, h * w * arr.shape[1])


def rate_code_trains(
    rows: np.ndarray,
    saturation_level: float,
    *,
    tpres: int = DEFAULT_TPRES,
    reset_state_per_record: bool = False,
) -> np.ndarray:
    """
    Boolean spike trains (N, T, D).

    Per column, state carries across records unless reset_state_per_record.
    If r <= 0 the state is not updated for that record (fromFile skips tacts).
    """
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D rows, got {x.shape}")
    if saturation_level <= 0:
        raise ValueError("saturation_level must be positive")
    n, d = x.shape
    tpres = int(tpres)
    trains = np.zeros((n, tpres, d), dtype=bool)
    state = np.zeros(d, dtype=np.float64)
    sat = float(saturation_level)
    for i in range(n):
        if reset_state_per_record:
            state[:] = 0.0
        r = x[i]
        dval = np.zeros(d, dtype=np.float64)
        pos = r > 0.0
        if np.any(pos):
            below = pos & (r < sat)
            dval[below] = r[below] / sat
            dval[pos & ~below] = 1.0
            for t in range(tpres):
                state += dval
                fired = state >= 1.0
                trains[i, t] = fired
                state[fired] -= 1.0
    return trains


def rate_code_counts(
    rows: np.ndarray,
    saturation_level: float,
    *,
    tpres: int = DEFAULT_TPRES,
    reset_state_per_record: bool = False,
) -> np.ndarray:
    trains = rate_code_trains(
        rows,
        saturation_level,
        tpres=tpres,
        reset_state_per_record=reset_state_per_record,
    )
    return trains.sum(axis=1).astype(np.int32)


def write_activation_csv(path, rows: np.ndarray, *, fmt: str = "%.8g") -> None:
    """DatTab-style CSV: header n0,n1,... then one row per image."""
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D rows, got {arr.shape}")
    header = ",".join(f"n{i}" for i in range(arr.shape[1]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt=fmt)
