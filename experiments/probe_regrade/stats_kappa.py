"""Cohen's kappa for binary verdict pairs (stdlib only)."""
from __future__ import annotations


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """pairs: [(judge_a_verdict, judge_b_verdict)]. Returns kappa in [-1, 1].
    If both judges are constant and identical, agreement is perfect by definition -> 1.0."""
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)
