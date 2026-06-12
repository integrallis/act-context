"""H2' decisive sub-test: does RESIDUAL preserve information at its lower token cost?

The token/maintenance run shows residual is ~0.40x tokens at iso-NODE coverage — but that is
only a win if residual still conveys the same INFORMATION (vs dropping content). Here we test it.

For sampled "map nodes" P (internal nodes with >=3 children — a realistic neighborhood), we build
two context maps of the SAME nodes (ancestors + P + P's children):
  - RESIDUAL map:   ancestors' residual summaries (shared, once) + P + each child's residual delta.
  - CUMULATIVE map: P + each child's self-contained cumulative summary (ancestor context restated
                    per child => duplication across children).
We generate probe questions FROM GROUND-TRUTH SOURCE (not from either summary), answer each probe
given each map, and grade vs a reference answer. C2 holds iff residual preserves answer accuracy
at materially fewer tokens. Answerer + judge = frontier Anthropic (the realistic client).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..extract import extract_tree
from ..llm import cheap_client, frontier_client
from ..model import Tree
from ..summarize import summarize_tree


def _tok():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda t: len(enc.encode(t or ""))


def build_map(tree: Tree, pid: str, scheme: str) -> str:
    parts = []
    if scheme == "source":
        # CONTROL (act-view): full source of children. Validates the harness and shows
        # whether behavioral probes are answerable from CODE (expected yes) vs summaries.
        for c in tree.children(pid):
            if c.source:
                parts.append(c.source)
        return "\n\n".join(parts)
    if scheme == "residual":
        for a in reversed(tree.ancestors(pid)):  # root-first, shared context once
            parts.append(f"[{a.kind.value} {a.name}] {a.residual_summary}")
        p = tree.nodes[pid]
        parts.append(f"[{p.kind.value} {p.name}] {p.residual_summary}")
        for c in tree.children(pid):
            parts.append(f"  - [{c.kind.value} {c.name}] {c.residual_summary}")
    else:  # cumulative: self-contained P + children (no ancestors; context restated per node)
        p = tree.nodes[pid]
        parts.append(f"[{p.kind.value} {p.name}] {p.cumulative_summary}")
        for c in tree.children(pid):
            parts.append(f"  - [{c.kind.value} {c.name}] {c.cumulative_summary}")
    return "\n".join(parts)


def _children_source(tree: Tree, pid: str, limit_chars: int = 4000) -> str:
    out = []
    for c in tree.children(pid):
        if c.source:
            out.append(c.source)
    return "\n\n".join(out)[:limit_chars]


def gen_probes(frontier, name: str, source: str, k: int) -> list[dict]:
    prompt = (
        f"Here is the source of the parts of a code unit named '{name}':\n\n{source}\n\n"
        f"Write {k} specific factual questions about what these parts DO (behavior, return values, "
        f"responsibilities), each answerable ONLY from this source. Give a short reference answer "
        f"for each. Reply as STRICT JSON: a list of objects with keys \"q\" and \"ref\". No prose.")
    raw = frontier.generate(prompt, max_tokens=700)
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        items = json.loads(m.group(0) if m else raw)
        return [{"q": str(i["q"]), "ref": str(i["ref"])} for i in items][:k]
    except Exception:
        return []


def answer(frontier, context: str, q: str) -> str:
    prompt = (f"Use ONLY this context to answer; if it is not answerable, say \"UNKNOWN\".\n\n"
              f"CONTEXT:\n{context}\n\nQUESTION: {q}\nAnswer concisely:")
    return frontier.generate(prompt, max_tokens=160)


def grade(frontier, q: str, ref: str, ans: str) -> int:
    prompt = (f"Question: {q}\nReference answer: {ref}\nCandidate answer: {ans}\n\n"
              f"Is the candidate answer correct and consistent with the reference? "
              f"Reply with ONE word: CORRECT or INCORRECT.")
    v = frontier.generate(prompt, max_tokens=8).strip().upper()
    return 1 if "CORRECT" in v and "INCORRECT" not in v else 0


def run(repo_path, cheap_model="qwen2.5-coder:3b", frontier_model="claude-sonnet-4-6",
        max_internal=40, n_maps=5, k_probes=3, out_dir="results"):
    tree = extract_tree(repo_path)
    print(f"tree {len(tree.nodes)} nodes; summarizing (cheap={cheap_model})...")
    summarize_tree(tree, cheap_client(cheap_model), max_internal=max_internal)
    fr = frontier_client(frontier_model)
    counter = _tok()

    # map nodes: internal nodes with >=3 children that have source (classes/files)
    cands = [n for n in tree.internal_nodes()
             if len([c for c in tree.children(n.id) if c.source]) >= 3]
    cands.sort(key=lambda n: -len(n.children))
    maps = cands[:n_maps]
    print(f"{len(maps)} map nodes; {k_probes} probes each; frontier={frontier_model}")

    schemes = ("residual", "cumulative", "source")  # source = act-view control
    rows, tok = [], {s: [] for s in schemes}
    acc = {s: [] for s in schemes}
    for p in maps:
        src = _children_source(tree, p.id)
        probes = gen_probes(fr, p.name, src, k_probes)
        if not probes:
            continue
        ctx = {s: build_map(tree, p.id, s) for s in schemes}
        for s in schemes:
            tok[s].append(counter(ctx[s]))
        for pr in probes:
            for s in schemes:
                a = answer(fr, ctx[s], pr["q"])
                g = grade(fr, pr["q"], pr["ref"], a)
                acc[s].append(g)
                rows.append({"node": p.name, "scheme": s, "q": pr["q"][:80], "correct": g})

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    summary = {
        "repo": str(repo_path), "cheap_model": cheap_model, "frontier_model": frontier_model,
        "n_maps": len(maps), "k_probes": k_probes, "n_graded": len(acc["residual"]),
        "accuracy": {s: mean(acc[s]) for s in schemes},
        "map_tokens": {s: mean(tok[s]) for s in schemes},
        "token_ratio_residual_over_cumulative":
            round(mean(tok["residual"]) / mean(tok["cumulative"]), 3) if mean(tok["cumulative"]) else None,
        "rows": rows,
    }
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "h2_info_preservation.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H2' information preservation (source = act-view control) ===")
    for s in schemes:
        print(f"  {s:11s}: accuracy={summary['accuracy'][s]:.3f}  map_tokens={summary['map_tokens'][s]:.0f}")
    print(f"  residual uses {summary['token_ratio_residual_over_cumulative']}x tokens at "
          f"accuracy {summary['accuracy']['residual']} vs {summary['accuracy']['cumulative']}")
    print(f"  VERDICT: C2 holds iff residual accuracy ~>= cumulative AND token ratio < 1.")
    print(f"Wrote {out / 'h2_info_preservation.json'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--cheap-model", default="qwen2.5-coder:3b")
    p.add_argument("--frontier-model", default="claude-sonnet-4-6")
    p.add_argument("--max-internal", type=int, default=40)
    p.add_argument("--n-maps", type=int, default=5)
    p.add_argument("--k-probes", type=int, default=3)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    run(a.repo, a.cheap_model, a.frontier_model, a.max_internal, a.n_maps, a.k_probes, a.out_dir)
