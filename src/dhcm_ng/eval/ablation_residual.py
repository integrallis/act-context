"""H2' ablation: RESIDUAL vs CUMULATIVE summaries (the lead claim, C2).

Two metrics, both at iso-coverage (same root->locus paths):
  1. Assembly redundancy + summary tokens of the root->locus path.
  2. Maintenance cost: #nodes to re-summarize when a leaf is edited.

Real cheap model (Ollama) for summaries — NO mock. Real tokenizer (tiktoken).
Prediction (C2): residual has LOWER tokens AND LOWER redundancy at equal coverage, and
O(1) vs O(depth) maintenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..extract import extract_tree
from ..llm import cheap_client
from ..model import Kind, Tree
from ..summarize import summarize_tree, resummarize_set
from ..assemble import assess_locus


def _tiktoken_counter():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda t: len(enc.encode(t or ""))


def _sample_loci(tree: Tree, n: int) -> list[str]:
    # function leaves that have at least 2 ancestors (so the path is non-trivial)
    leaves = [x.id for x in tree.by_kind(Kind.FUNCTION)
              if not x.children and len(tree.ancestors(x.id)) >= 2]
    leaves.sort()
    if n >= len(leaves):
        return leaves
    step = max(1, len(leaves) // n)
    return leaves[::step][:n]


def run(repo_path, model="qwen2.5-coder:3b", max_internal=60, n_loci=40, out_dir="results"):
    tree = extract_tree(repo_path)
    n_internal = len(tree.internal_nodes())
    print(f"tree: {len(tree.nodes)} nodes, {n_internal} internal "
          f"(summarizing up to {max_internal}); model={model}")
    llm = cheap_client(model)
    rpt = summarize_tree(tree, llm, max_internal=max_internal)
    print(f"summarized: {rpt['llm_calls']} LLM calls over {rpt['internal']} internal nodes")

    counter = _tiktoken_counter()
    loci = _sample_loci(tree, n_loci)

    agg = {s: {"summary_tokens": [], "redundancy": []} for s in ("residual", "cumulative")}
    for lid in loci:
        for s in ("residual", "cumulative"):
            a = assess_locus(tree, lid, s, counter)
            agg[s]["summary_tokens"].append(a["summary_tokens"])
            agg[s]["redundancy"].append(a["redundancy"])

    # maintenance: re-summarize set size per edited leaf
    maint = {"residual": [], "cumulative": []}
    for lid in loci:
        for s in ("residual", "cumulative"):
            maint[s].append(len(resummarize_set(tree, lid, s)))

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    summary = {
        "repo": str(repo_path), "model": model, "n_loci": len(loci),
        "tree_nodes": len(tree.nodes), "internal_nodes": n_internal, "llm_calls": rpt["llm_calls"],
        "assembly": {s: {"mean_summary_tokens": mean(agg[s]["summary_tokens"]),
                         "mean_redundancy": mean(agg[s]["redundancy"])}
                     for s in ("residual", "cumulative")},
        "maintenance": {s: {"mean_resummarize_nodes": mean(maint[s])}
                        for s in ("residual", "cumulative")},
    }
    # deltas (residual relative to cumulative)
    cu, re_ = summary["assembly"]["cumulative"], summary["assembly"]["residual"]
    summary["delta"] = {
        "token_ratio_residual_over_cumulative":
            round(re_["mean_summary_tokens"] / cu["mean_summary_tokens"], 3) if cu["mean_summary_tokens"] else None,
        "redundancy_abs_drop": round(cu["mean_redundancy"] - re_["mean_redundancy"], 4),
        "maint_ratio_residual_over_cumulative":
            round(summary["maintenance"]["residual"]["mean_resummarize_nodes"] /
                  summary["maintenance"]["cumulative"]["mean_resummarize_nodes"], 3)
            if summary["maintenance"]["cumulative"]["mean_resummarize_nodes"] else None,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "h2_residual_vs_cumulative.json").write_text(json.dumps(summary, indent=2))

    print("\n=== H2' residual vs cumulative ===")
    for s in ("residual", "cumulative"):
        a, m = summary["assembly"][s], summary["maintenance"][s]
        print(f"  {s:11s}: summary_tokens={a['mean_summary_tokens']:.1f}  "
              f"redundancy={a['mean_redundancy']:.3f}  resummarize_nodes={m['mean_resummarize_nodes']:.2f}")
    print(f"  delta: residual uses {summary['delta']['token_ratio_residual_over_cumulative']}x tokens, "
          f"redundancy -{summary['delta']['redundancy_abs_drop']:.3f}, "
          f"maintenance {summary['delta']['maint_ratio_residual_over_cumulative']}x nodes")
    print(f"Wrote {out / 'h2_residual_vs_cumulative.json'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", default="qwen2.5-coder:3b")
    p.add_argument("--max-internal", type=int, default=60)
    p.add_argument("--n-loci", type=int, default=40)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    run(a.repo, model=a.model, max_internal=a.max_internal, n_loci=a.n_loci, out_dir=a.out_dir)
