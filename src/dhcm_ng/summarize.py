"""Bottom-up summarization in TWO schemes — the C2 ablation (residual vs cumulative).

CUMULATIVE (RAPTOR/HCGS-style, the baseline): each node's summary is SELF-CONTAINED — it
restates the essence of its subtree. Including it gives full context but DUPLICATES ancestor/
descendant content, and editing any descendant invalidates the whole ancestor chain.

RESIDUAL (the contribution): each node's summary is the LOCAL DELTA only — what this node adds
beyond what its parent's summary already conveys. Assembling a root->locus path yields the
ancestor's context once + each level's delta => duplication-free; and editing a deep leaf does
NOT change ancestors' deltas => O(1) maintenance instead of O(depth).

Leaves (functions) get a deterministic signature+docstring "summary" (no LLM, identical in both
schemes). Only internal nodes (class/file/component/subsystem) cost an LLM call per scheme.
"""

from __future__ import annotations

from .model import Kind, LLMClient, Node, Tree

# Cumulative summary budget grows with abstraction level (∝ subtree); residual is terse.
_CUMULATIVE_TOK = {Kind.CLASS: 180, Kind.FILE: 260, Kind.COMPONENT: 360, Kind.SUBSYSTEM: 480}
_RESIDUAL_TOK = {Kind.CLASS: 90, Kind.FILE: 110, Kind.COMPONENT: 140, Kind.SUBSYSTEM: 180}


def _leaf_summary(n: Node) -> str:
    doc = (n.docstring or "").strip().splitlines()
    first = doc[0] if doc else ""
    sig = n.signature or n.name
    return f"{sig}{(' — ' + first) if first else ''}"


def _children_digest(tree: Tree, node: Node, scheme: str, limit: int = 40) -> str:
    lines = []
    for c in tree.children(node.id):
        s = (c.cumulative_summary if scheme == "cumulative" else c.residual_summary) or c.name
        lines.append(f"- {c.kind.value} {c.name}: {s}")
    return "\n".join(lines[:limit])


def _cumulative_prompt(node: Node, digest: str) -> str:
    return (
        f"You are documenting a software {node.kind.value} named '{node.name}'.\n"
        f"Its parts:\n{digest}\n\n"
        f"Write a SELF-CONTAINED summary describing what this {node.kind.value} does and the "
        f"roles of its parts, so a reader needs nothing else. Plain prose, no preamble.")


def _residual_prompt(node: Node, parent_summary: str, digest: str) -> str:
    return (
        f"Parent context (already known to the reader):\n{parent_summary or '(top level)'}\n\n"
        f"This is a {node.kind.value} named '{node.name}' inside that parent. Its parts:\n{digest}\n\n"
        f"Describe ONLY what is specific to '{node.name}' that the parent context does NOT already "
        f"convey. Do NOT restate the parent or the overall system. Be terse. Plain prose, no preamble.")


def summarize_tree(tree: Tree, llm: LLMClient, max_internal: int | None = None) -> dict:
    """Fill residual_summary + cumulative_summary on every node. Returns a small report
    ({'llm_calls': n, 'internal': n}). Leaves are deterministic; internal nodes call the LLM
    twice (once per scheme)."""
    # 1) leaves: deterministic, identical in both schemes
    for n in tree.nodes.values():
        if not n.children:
            s = _leaf_summary(n)
            n.cumulative_summary = s
            n.residual_summary = s

    internal = sorted(tree.internal_nodes(), key=lambda n: _depth(tree, n.id))
    if max_internal is not None:
        # keep the deepest `max_internal` internal nodes (classes/files) for bounded runs
        internal = sorted(internal, key=lambda n: _depth(tree, n.id), reverse=True)[:max_internal]
        internal = sorted(internal, key=lambda n: _depth(tree, n.id))
    deep_first = sorted(internal, key=lambda n: _depth(tree, n.id), reverse=True)
    shallow_first = sorted(internal, key=lambda n: _depth(tree, n.id))

    calls = 0
    # 2) CUMULATIVE: bottom-up (children cumulative summaries must exist first)
    for n in deep_first:
        digest = _children_digest(tree, n, "cumulative")
        n.cumulative_summary = llm.generate(_cumulative_prompt(n, digest),
                                            max_tokens=_CUMULATIVE_TOK.get(n.kind, 200)) or n.name
        calls += 1
    # 3) RESIDUAL: top-down (needs parent's cumulative summary as the "already known" context)
    for n in shallow_first:
        parent = tree.nodes[n.parent] if n.parent else None
        parent_ctx = parent.cumulative_summary if parent else ""
        digest = _children_digest(tree, n, "residual")
        n.residual_summary = llm.generate(_residual_prompt(n, parent_ctx, digest),
                                          max_tokens=_RESIDUAL_TOK.get(n.kind, 110)) or n.name
        calls += 1

    return {"llm_calls": calls, "internal": len(internal)}


def _depth(tree: Tree, nid: str) -> int:
    d, cur = 0, tree.nodes[nid].parent
    while cur is not None:
        d += 1
        cur = tree.nodes[cur].parent
    return d


# --- maintenance cost (the second H2' metric) ----------------------------------
def resummarize_set(tree: Tree, edited_leaf_id: str, scheme: str) -> set[str]:
    """Nodes whose summary must be regenerated if `edited_leaf_id`'s code changes.

    CUMULATIVE: the leaf + ALL ancestors (each ancestor's self-contained summary restates
    descendant content). RESIDUAL: the leaf + its DIRECT parent only (ancestors' deltas describe
    what THEY add beyond THEIR parent and do not encode the deep leaf's internals)."""
    if scheme == "cumulative":
        return {edited_leaf_id} | {a.id for a in tree.ancestors(edited_leaf_id)}
    parent = tree.nodes[edited_leaf_id].parent
    return {edited_leaf_id} | ({parent} if parent else set())
