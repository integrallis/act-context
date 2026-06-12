"""H-find: localization via beam descent — residual vs cumulative descent keys, vs flat retrieval,
stratified by lexical overlap (the test of whether the finder is novel where flat search fails).

Self-supervised localization: query = a function's docstring; gold = that function. ANTI-LEAKAGE:
leaf descent/flat keys = the function SIGNATURE ONLY (def line, no docstring), and internal-node
keys are LLM summaries (no docstring) — so matching requires semantic alignment, not exact text.

Stratify by lexical overlap between the query and the function NAME tokens: LOW overlap =
conceptual (where structure-guided descent should help); HIGH = lexical (flat should already win).
Reports recall@k for residual-descent / cumulative-descent / flat, overall + per stratum, + cost
(nodes scored: descent = beam*depth vs flat = all leaves).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..extract import extract_tree
from ..embed import STEmbedder, cosine
from ..llm import cheap_client
from ..model import Kind, Tree
from ..summarize import summarize_tree
from ..descent import descend


def _subtokens(name: str) -> set[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower().split()
    return {p for p in parts if len(p) > 2}


def _qtokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9]+", text.lower()) if len(t) > 2}


def _descent_embs(tree: Tree, embedder, scheme: str) -> dict[str, list[float]]:
    """Descent keys: internal = LLM summary (scheme); leaf = SIGNATURE ONLY (anti-leakage)."""
    ids, texts = [], []
    for nid, n in tree.nodes.items():
        if n.children:
            key = (n.residual_summary if scheme == "residual" else n.cumulative_summary) or n.name
        else:
            key = n.signature or n.name           # NO docstring -> no leakage from the query
        ids.append(nid); texts.append(key)
    return dict(zip(ids, embedder.embed_batch(texts)))


def run(repo_path, cheap_model="qwen2.5-coder:3b", beam=3, k=5, n_queries=120, out_dir="results"):
    tree = extract_tree(repo_path)
    print(f"tree {len(tree.nodes)} nodes, {len(tree.internal_nodes())} internal; summarizing all...")
    summarize_tree(tree, cheap_client(cheap_model))  # all internal nodes (no cap)
    emb = STEmbedder()

    res_embs = _descent_embs(tree, emb, "residual")
    cum_embs = _descent_embs(tree, emb, "cumulative")
    leaf_ids = [n.id for n in tree.nodes.values() if not n.children]
    leaf_embs = {i: res_embs[i] for i in leaf_ids}  # leaf keys identical across schemes (signature)

    # queries: leaves with a real docstring
    queries = [n for n in tree.by_kind(Kind.FUNCTION)
               if not n.children and n.docstring and len(n.docstring) > 25]
    queries.sort(key=lambda n: n.id)
    queries = queries[:n_queries]
    print(f"{len(queries)} docstring queries; beam={beam}, k={k}")

    def overlap(n):
        nt = _subtokens(n.name)
        return (len(nt & _qtokens(n.docstring)) / len(nt)) if nt else 0.0

    buckets = {"all": [], "conceptual": [], "lexical": []}  # each: list of (res_hit, cum_hit, flat_hit)
    res_cost, flat_cost = [], []
    qvecs = emb.embed_batch([q.docstring[:400] for q in queries])
    for q, qv in zip(queries, qvecs):
        res_top, sc = descend(tree, qv, res_embs, beam=beam, k=k); res_cost.append(sc)
        cum_top, _ = descend(tree, qv, cum_embs, beam=beam, k=k)
        flat_ranked = sorted(((cosine(qv, v), i) for i, v in leaf_embs.items()), reverse=True)
        flat_top = [i for _, i in flat_ranked[:k]]; flat_cost.append(len(leaf_embs))
        rec = (int(q.id in res_top), int(q.id in cum_top), int(q.id in flat_top))
        buckets["all"].append(rec)
        buckets["conceptual" if overlap(q) < 0.34 else "lexical"].append(rec)

    def recalls(rows):
        if not rows:
            return {"n": 0}
        n = len(rows)
        return {"n": n,
                "residual": round(sum(r[0] for r in rows) / n, 3),
                "cumulative": round(sum(r[1] for r in rows) / n, 3),
                "flat": round(sum(r[2] for r in rows) / n, 3)}

    summary = {
        "repo": str(repo_path), "cheap_model": cheap_model, "beam": beam, "k": k,
        "recall_at_k": {b: recalls(rows) for b, rows in buckets.items()},
        "mean_nodes_scored": {"descent": round(sum(res_cost) / len(res_cost), 1) if res_cost else 0,
                              "flat": round(sum(flat_cost) / len(flat_cost), 1) if flat_cost else 0},
    }
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "h_find_localization.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== FIND localization recall@{k} ===")
    for b in ("all", "conceptual", "lexical"):
        r = summary["recall_at_k"][b]
        if r["n"]:
            print(f"  {b:11s} (n={r['n']:3d}): residual={r['residual']:.3f}  "
                  f"cumulative={r['cumulative']:.3f}  flat={r['flat']:.3f}")
    c = summary["mean_nodes_scored"]
    print(f"  cost (nodes scored): descent={c['descent']:.0f} vs flat={c['flat']:.0f}")
    print(f"  C2(localization): residual ~>= cumulative & cheaper?  C1: descent > flat on CONCEPTUAL?")
    print(f"Wrote {out / 'h_find_localization.json'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--cheap-model", default="qwen2.5-coder:3b")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-queries", type=int, default=120)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    run(a.repo, a.cheap_model, a.beam, a.k, a.n_queries, a.out_dir)
