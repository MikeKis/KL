"""Analytic normal-distribution sparsity helper."""

import math
from statistics import NormalDist


_STANDARD_NORMAL = NormalDist()
_SQRT_TWO = math.sqrt(2.0)


def NormalSparsity(q, n=10):
    """Return quantized E[max(0, min(1, floor(n * X / v) / n))]."""
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise ValueError("q must be a real number in the range (0, 0.5]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    q = float(q)
    if not math.isfinite(q) or q <= 0.0 or q > 0.5:
        raise ValueError("q must be a real number in the range (0, 0.5]")

    if q == 0.5:
        return 0.5

    v = -_STANDARD_NORMAL.inv_cdf(q)
    tail_sum = 0.0
    for k in range(1, n + 1):
        tail_sum += 0.5 * math.erfc((k * v / n) / _SQRT_TWO)
    return float(tail_sum / n)
