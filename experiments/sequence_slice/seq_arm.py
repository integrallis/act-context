"""exp-014c: the sequence-slice arm — the first INTERACTION-level artifact in the rig.

Tests the owner's original hypothesis that the H2 null (skeletons/signatures vs keep/drop) did NOT
cover: that UML *sequence/interaction* information — who calls whom, in what order, around the edit
site — carries resolve signal that flat text and placeholders do not.

Arm definition (registered): `oracle_seq` = oracle_keep_drop's EXACT content (gold units verbatim,
module preamble, non-.py gold files, drop placeholders) PLUS a deterministic, syntactically derived
Mermaid sequence slice of the gold functions' local interactions. The paired comparison
oracle_seq-vs-oracle_keep_drop therefore isolates exactly one variable: the added interaction
artifact.

Slice derivation (pure AST, no LLM, no call-graph infrastructure needed at oracle scope):
  - name index over FUNCTION/CLASS nodes defined in the gold files (same-file resolution preferred)
  - outgoing: every ast.Call inside a gold function whose callee name resolves in the index,
    in source-line order
  - incoming: calls from non-gold functions in the gold files TO gold-function names
  - rendered as a Mermaid sequenceDiagram, capped at MAX_MESSAGES (overflow noted, never silent)
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "resolve"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import arms_v2 as av  # noqa: E402
from dhcm_ng.context_arms import _file_full_source, ntok  # noqa: E402
from dhcm_ng.model import Kind, Tree  # noqa: E402

MAX_MESSAGES = 40


def _callee_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _functions_in_file(src: str):
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node)
    return out


def call_slice(repo_path, gold_files: list[str], tree: Tree, gold_units: set[str]) -> str:
    """Deterministic Mermaid sequence slice of gold-function interactions within the gold files."""
    repo_path = Path(repo_path)
    py_files = [f for f in gold_files if f.endswith(".py")]
    gold_nodes = [tree.nodes[i] for i in gold_units if i in tree.nodes]
    gold_fn_names = {n.name for n in gold_nodes if n.kind == Kind.FUNCTION}
    gold_spans = {}  # file -> [(start, end, name)]
    for n in gold_nodes:
        if n.kind == Kind.FUNCTION:
            gold_spans.setdefault(n.path.replace("\\", "/"), []).append(
                (n.start_line, n.end_line, n.name))

    # name index: entities defined in the gold files (functions + classes)
    defined: dict[str, set[str]] = {}
    for f in py_files:
        for n in tree.nodes.values():
            if n.kind in (Kind.FUNCTION, Kind.CLASS) and n.path.replace("\\", "/").endswith(f):
                defined.setdefault(n.name, set()).add(f)

    messages: list[tuple[int, str, str, str]] = []  # (line, caller, callee, label)
    for f in py_files:
        src = _file_full_source(repo_path, f)
        spans = gold_spans.get(f, []) or [s for ff, ss in gold_spans.items()
                                          if f.endswith(ff) for s in ss]
        for fn in _functions_in_file(src):
            start, end = fn.lineno, getattr(fn, "end_lineno", fn.lineno)
            owner = next((nm for a, b, nm in spans if a <= start and end <= b + 1
                          and fn.name == nm), None)
            is_gold = owner is not None
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                cal = _callee_name(node)
                if not cal or cal == fn.name:
                    continue
                if is_gold and cal in defined:
                    messages.append((node.lineno, fn.name, cal, f))
                elif not is_gold and cal in gold_fn_names:
                    messages.append((node.lineno, fn.name, cal, f))
    if not messages:
        return ""
    messages.sort()
    seen, ordered = set(), []
    for line, a, b, f in messages:
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((a, b))
    overflow = max(0, len(ordered) - MAX_MESSAGES)
    ordered = ordered[:MAX_MESSAGES]
    participants = []
    for a, b in ordered:
        for p in (a, b):
            if p not in participants:
                participants.append(p)
    lines = ["```mermaid", "sequenceDiagram"]
    lines += [f"    participant {p}" for p in participants]
    lines += [f"    {a}->>{b}: {b}()" for a, b in ordered]
    if overflow:
        lines.append(f"    Note over {participants[0]}: ... {overflow} more call edges omitted")
    lines.append("```")
    return "\n".join(lines)


def build_seq_arm(tree: Tree, repo_path, gold_files: list[str],
                  edited_lines: dict[str, set[int]]) -> dict:
    """oracle_seq = oracle_keep_drop content + the Mermaid interaction slice (appended section).
    Returns {text, tokens, slice_tokens, n_messages>0}."""
    base = av.build_oracle_arms_v2(tree, repo_path, gold_files, edited_lines)
    kd = base["oracle_keep_drop"]["text"]
    gold_units = set(base["_gold_units"])
    sl = call_slice(repo_path, gold_files, tree, gold_units)
    if sl:
        text = kd + "\n\n### Interaction slice (derived from the code above; who calls whom, in order)\n" + sl
    else:
        text = kd
    return {"text": text, "tokens": ntok(text), "slice_tokens": ntok(sl),
            "has_slice": bool(sl),
            "keep_drop_text": kd, "keep_drop_tokens": base["oracle_keep_drop"]["tokens"]}
