"""Analytic normal-distribution sparsity helper."""

import math
from statistics import NormalDist


_STANDARD_NORMAL = NormalDist()
_PDF_AT_ZERO = 1.0 / math.sqrt(2.0 * math.pi)


def NormalSparsity(q):
    """Return E[max(0, min(1, X / v))] for X ~ N(0, 1), P(X > v) = q."""
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise ValueError("q must be a real number in the range (0, 0.5]")

    q = float(q)
    if not math.isfinite(q) or q <= 0.0 or q > 0.5:
        raise ValueError("q must be a real number in the range (0, 0.5]")

    if q == 0.5:
        return 0.5

    v = -_STANDARD_NORMAL.inv_cdf(q)
    numerator = -_PDF_AT_ZERO * math.expm1(-0.5 * v * v)
    return float(q + numerator / v)
