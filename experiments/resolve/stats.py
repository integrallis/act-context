"""Pure statistics for exp-014b: Wilson CIs + exact McNemar (no deps beyond stdlib)."""
from __future__ import annotations

from math import comb, sqrt


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c.

    b = # instances where arm A resolved and arm B did not; c = the reverse.
    Concordant pairs (both resolved / both failed) are irrelevant by construction.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def discordant(resolved_a: set[str], resolved_b: set[str], universe: set[str]) -> tuple[int, int]:
    """(b, c) over `universe`: b = A-only resolved, c = B-only resolved."""
    a = resolved_a & universe
    bb = resolved_b & universe
    return (len(a - bb), len(bb - a))


def paired_mcnemar(resolved_a: set[str], resolved_b: set[str], universe: set[str]) -> dict:
    b, c = discordant(resolved_a, resolved_b, universe)
    return {"b_only_A": b, "c_only_B": c, "discordant": b + c, "p_value": mcnemar_exact(b, c)}
