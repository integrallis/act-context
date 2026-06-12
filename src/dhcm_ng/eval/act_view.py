"""ACT-view representation experiment (the user's question, made empirical).

The earlier control proved: for behavioral questions, summaries=0%, source=100%. But "source"
need not mean "the WHOLE class." Here we ask: what is the MINIMAL representation that still lets
the client act correctly on a locus method M?

Representations tested for a locus method M (enclosing class C):
  - full_class     : C's entire source (the naive act-view)
  - method_only    : only M's source
  - uml_plus_method: UML skeleton of C (signatures/fields, NO bodies) + M's full body  [the GATE]
  - signature_only : M's signature + docstring

Probes are behavioral, generated from M's ground-truth source. Frontier (Anthropic) answers from
each representation and judges. The minimal representation that matches full_class accuracy at the
lowest tokens is the act-view target — and answers "whole class vs just the method?" and "can UML
carry equal info at lower cognitive load?" with data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..extract import extract_tree
from ..llm import frontier_client
from ..model import Kind, Tree
from ..reduce import uml_class_skeleton, function_signature
from .info_preservation import gen_probes, answer, grade, _tok


def _representations(tree: Tree, m) -> dict[str, str]:
    parent = tree.nodes[m.parent] if m.parent else None
    enclosing_class = parent if (parent and parent.kind == Kind.CLASS) else None
    reps = {"method_only": m.source, "signature_only": function_signature(m.source)}
    if enclosing_class and enclosing_class.source:
        reps["full_class"] = enclosing_class.source
        reps["uml_plus_method"] = uml_class_skeleton(enclosing_class.source) + \
            f"\n\n# --- locus method (full body) ---\n{m.source}"
    else:
        reps["full_class"] = m.source       # module-level fn: no class context
        reps["uml_plus_method"] = m.source
    return reps


def _sample_methods(tree: Tree, n: int):
    cands = [m for m in tree.by_kind(Kind.FUNCTION)
             if not m.children and m.parent and tree.nodes[m.parent].kind == Kind.CLASS
             and m.source and len(m.source) > 120 and (m.docstring or "return" in m.source)]
    cands.sort(key=lambda m: -len(m.source))
    return cands[:n]


def run(repo_path, frontier_model="claude-sonnet-4-6", n_methods=6, k_probes=3, out_dir="results"):
    tree = extract_tree(repo_path)
    fr = frontier_client(frontier_model)
    counter = _tok()
    methods = _sample_methods(tree, n_methods)
    print(f"{len(methods)} locus methods; {k_probes} probes each; frontier={frontier_model}")

    reps_order = ["full_class", "uml_plus_method", "method_only", "signature_only"]
    acc = {r: [] for r in reps_order}
    tok = {r: [] for r in reps_order}
    rows = []
    for m in methods:
        probes = gen_probes(fr, m.name, m.source, k_probes)
        if not probes:
            continue
        reps = _representations(tree, m)
        for r in reps_order:
            tok[r].append(counter(reps[r]))
        for pr in probes:
            for r in reps_order:
                a = answer(fr, reps[r], pr["q"])
                g = grade(fr, pr["q"], pr["ref"], a)
                acc[r].append(g)
                rows.append({"method": m.name, "rep": r, "correct": g})

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    summary = {
        "repo": str(repo_path), "frontier_model": frontier_model,
        "n_methods": len(methods), "k_probes": k_probes, "n_graded_per_rep": len(acc["full_class"]),
        "by_representation": {r: {"accuracy": mean(acc[r]), "mean_tokens": mean(tok[r])} for r in reps_order},
        "rows": rows,
    }
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "act_view_representations.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ACT-view: minimal sufficient representation ===")
    base = summary["by_representation"]["full_class"]
    for r in reps_order:
        b = summary["by_representation"][r]
        ratio = round(b["mean_tokens"] / base["mean_tokens"], 3) if base["mean_tokens"] else None
        print(f"  {r:16s}: accuracy={b['accuracy']:.3f}  tokens={b['mean_tokens']:.0f}  ({ratio}x full_class)")
    print("  -> the rep matching full_class accuracy at the FEWEST tokens is the act-view target.")
    print(f"Wrote {out / 'act_view_representations.json'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--frontier-model", default="claude-sonnet-4-6")
    p.add_argument("--n-methods", type=int, default=6)
    p.add_argument("--k-probes", type=int, default=3)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    run(a.repo, a.frontier_model, a.n_methods, a.k_probes, a.out_dir)
