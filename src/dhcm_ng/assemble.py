"""Context assembly + redundancy/token metrics for the H2' ablation.

Assembling the root->locus path: each ancestor contributes its summary (scheme-specific) and the
locus contributes full source. Under CUMULATIVE, ancestor summaries restate overlapping context
(high redundancy, more tokens); under RESIDUAL, the top ancestor carries context once and each
level adds only its delta (low redundancy, fewer tokens) — at IDENTICAL node coverage (same path),
so this is an iso-coverage comparison.
"""

from __future__ import annotations

import re

from .model import Tree


def _tok(text: str, counter=None) -> int:
    if counter is not None:
        return counter(text)
    return max(1, len(text) // 4)


def path_to(tree: Tree, locus_id: str) -> list[str]:
    """Root -> locus node ids (inclusive)."""
    anc = list(reversed([a.id for a in tree.ancestors(locus_id)]))  # root-first
    return anc + [locus_id]


def assemble_path(tree: Tree, locus_id: str, scheme: str):
    """Return (summary_pieces, locus_source) for the root->parent ancestor summaries +
    the locus full source. `scheme` in {residual, cumulative}."""
    pieces = []
    for nid in path_to(tree, locus_id)[:-1]:  # ancestors only get summaries
        n = tree.nodes[nid]
        s = n.residual_summary if scheme == "residual" else n.cumulative_summary
        if s:
            pieces.append(s)
    locus = tree.nodes[locus_id]
    return pieces, (locus.source or locus.cumulative_summary or locus.name)


def _trigrams(text: str) -> list[tuple[str, str, str]]:
    toks = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())
    return list(zip(toks, toks[1:], toks[2:]))


def redundancy(pieces: list[str]) -> float:
    """Fraction of trigram occurrences that are duplicates across the assembled pieces.
    0 = no repeated content; ->1 = highly redundant. Measures cross-piece restatement."""
    grams = []
    for p in pieces:
        grams.extend(_trigrams(p))
    if not grams:
        return 0.0
    seen, dup = set(), 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / len(grams)


def summary_tokens(pieces: list[str], counter=None) -> int:
    return sum(_tok(p, counter) for p in pieces)


def assess_locus(tree: Tree, locus_id: str, scheme: str, counter=None) -> dict:
    pieces, _locus_src = assemble_path(tree, locus_id, scheme)
    return {
        "summary_tokens": summary_tokens(pieces, counter),
        "redundancy": round(redundancy(pieces), 4),
        "n_ancestors": len(pieces),
    }
