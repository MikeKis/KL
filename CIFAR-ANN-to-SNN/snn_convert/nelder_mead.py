# Spec: 2026-09-13_layerwise-from-colanet.md
"""Bounded Nelder–Mead maximizer (log-space for positive parameters)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _to_unit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, lo, hi) / lo) / np.log(hi / lo)


def _from_unit(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return lo * (hi / lo) ** u


def _simplex_collapsed(
    simplex: list[np.ndarray], lo: np.ndarray, hi: np.ndarray, xtol: float
) -> bool:
    """True when every parameter's vertex spread is < xtol times that coordinate's mean."""
    if xtol <= 0:
        return False
    xs = np.vstack([_from_unit(u, lo, hi) for u in simplex])
    mean = xs.mean(axis=0)
    spread = xs.max(axis=0) - xs.min(axis=0)
    return bool(np.all(spread < float(xtol) * mean))


def nelder_mead_max(
    fn: Callable[[np.ndarray], float],
    x0: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    max_iter: int = 40,
    step: float = 0.25,
    ftol: float = 1e-4,
    xtol: float = 0.001,
) -> tuple[np.ndarray, float, int]:
    """
    Maximize fn(x) with x in (lo, hi] via Nelder–Mead on a log-unit box.

    Stops when the score spread is <= ftol, or when for every coordinate the
    spread of simplex vertices is < xtol times the mean of that coordinate
    (measured in the original parameters, not the log-unit box).

    Returns (best_x, best_f, n_eval).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    if np.any(lo <= 0) or np.any(hi <= lo):
        raise ValueError("bounds must be positive and lo < hi")
    n = x0.size
    n_eval = 0

    def f_u(u: np.ndarray) -> float:
        nonlocal n_eval
        n_eval += 1
        return float(fn(_from_unit(u, lo, hi)))

    u0 = _to_unit(np.clip(x0, lo, hi), lo, hi)
    simplex = [u0.copy()]
    for i in range(n):
        u = u0.copy()
        u[i] = float(np.clip(u0[i] + step, 0.0, 1.0))
        if abs(u[i] - u0[i]) < 1e-6:
            u[i] = float(np.clip(u0[i] - step, 0.0, 1.0))
        simplex.append(u)
    scores = [f_u(u) for u in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(int(max_iter)):
        order = np.argsort(scores)[::-1]
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        if max(scores) - min(scores) <= ftol:
            break
        if _simplex_collapsed(simplex, lo, hi, xtol):
            break
        best, worst = simplex[0], simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = np.clip(centroid + alpha * (centroid - worst), 0.0, 1.0)
        fr = f_u(reflected)
        if scores[0] >= fr > scores[-2]:
            simplex[-1], scores[-1] = reflected, fr
            continue
        if fr > scores[0]:
            expanded = np.clip(centroid + gamma * (reflected - centroid), 0.0, 1.0)
            fe = f_u(expanded)
            if fe > fr:
                simplex[-1], scores[-1] = expanded, fe
            else:
                simplex[-1], scores[-1] = reflected, fr
            continue
        contracted = np.clip(centroid + rho * (worst - centroid), 0.0, 1.0)
        fc = f_u(contracted)
        if fc > scores[-1]:
            simplex[-1], scores[-1] = contracted, fc
            continue
        for i in range(1, n + 1):
            simplex[i] = np.clip(best + sigma * (simplex[i] - best), 0.0, 1.0)
            scores[i] = f_u(simplex[i])

    order = np.argsort(scores)[::-1]
    best_u = simplex[order[0]]
    return _from_unit(best_u, lo, hi), float(scores[order[0]]), n_eval
