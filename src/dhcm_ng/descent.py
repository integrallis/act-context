"""Beam descent over the architecture tree — the FIND mechanism (C1).

The abstraction hierarchy is the navigation hierarchy: at each level, rank the children of the
current frontier by cosine(query, child summary embedding), keep the top-`beam`, descend until
leaves. The descent key is the node's summary in a chosen scheme (residual or cumulative) — so
this is also where C2 (residual vs cumulative) is decided on LOCALIZATION rather than Q&A.

No LLM at query time; cost = beam x depth summary-embedding comparisons (vs flat = all leaves).
"""

from __future__ import annotations

from .embed import cosine
from .model import Tree


def summary_for(node, scheme: str) -> str:
    return node.residual_summary if scheme == "residual" else node.cumulative_summary


def index_tree(tree: Tree, embedder, scheme: str) -> dict[str, list[float]]:
    """Embed every node's descent key (its summary in `scheme`). Leaves use their own
    (scheme-identical) signature summary."""
    ids = list(tree.nodes.keys())
    texts = [summary_for(tree.nodes[i], scheme) or tree.nodes[i].name for i in ids]
    embs = embedder.embed_batch(texts)
    return dict(zip(ids, embs))


def descend(tree: Tree, query_vec, node_embs: dict[str, list[float]], beam: int = 3,
            k: int = 5) -> tuple[list[str], int]:
    """Beam-descend from root to leaves. Returns (top-k leaf ids, #nodes scored) — the latter
    is the descent cost (vs flat = #all leaves)."""
    frontier = [tree.root]
    scored = 0
    leaves_found: list[tuple[float, str]] = []
    while frontier:
        children = []
        for nid in frontier:
            children.extend(tree.nodes[nid].children)
        if not children:
            break
        ranked = []
        for c in children:
            s = cosine(query_vec, node_embs.get(c, []))
            scored += 1
            ranked.append((s, c))
        ranked.sort(reverse=True)
        # separate leaves (keep as candidates) from internal (continue descent)
        kept = ranked[:beam]
        next_frontier = []
        for s, c in kept:
            if tree.nodes[c].children:
                next_frontier.append(c)
            else:
                leaves_found.append((s, c))
        # also let leaves among ALL children compete (not just top-beam) so a high-scoring leaf
        # sibling of a chosen internal node is not lost:
        for s, c in ranked:
            if not tree.nodes[c].children:
                leaves_found.append((s, c))
        frontier = next_frontier
    leaves_found.sort(reverse=True)
    seen, out = set(), []
    for s, c in leaves_found:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= k:
            break
    return out, scored


def localize(tree: Tree, query: str, embedder, node_embs, beam: int = 3, k: int = 5):
    q = embedder.embed(query)
    return descend(tree, q, node_embs, beam=beam, k=k)


def flat_localize(tree: Tree, query: str, embedder, leaf_embs: dict[str, list[float]], k: int = 5):
    """Baseline: cosine of query vs ALL leaf embeddings (no hierarchy)."""
    q = embedder.embed(query)
    ranked = sorted(((cosine(q, v), i) for i, v in leaf_embs.items()), reverse=True)
    return [i for _, i in ranked[:k]], len(leaf_embs)
