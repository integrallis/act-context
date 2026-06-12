"""exp-014: representation-tier context arms for the resolve@cost study.

Builds, for a SWE-bench instance, the context under 5 arms that differ ONLY in how retrieved
elements are rendered (same retrieval scope):
  (a) full_file      : full source of files containing retrieved elements (baseline to beat)
  (b) tier_selector  : per-role rungs (locus->body, enclosing class->UML skeleton, relevant->sig)
  (c) uniform_r3     : EVERY retrieved element at signature+doc (control: cheaper tiers, no selection)
  (d) oracle_minimal : gold-edited functions->full + enclosing UML, else exclude (ceiling; analysis-only)
  (e) keep_drop      : locus->full, non-locus DROPPED with placeholder (SWEzze-style binary)

The decisive comparisons: (b) vs (e) = "do tiers carry signal keep/drop discards?"; (b) vs (c) =
"does per-element SELECTION beat uniform cheap tiers?". Real tiktoken token counts. No mocks.
Retrieval is issue-text-only (no gold leakage); gold is used ONLY for arm (d) and analysis.
"""

from __future__ import annotations

import re

from dhcm_ng.extract import extract_tree            # noqa: E402
from dhcm_ng.embed import STEmbedder, cosine        # noqa: E402
from dhcm_ng.reduce import uml_class_skeleton, function_signature  # noqa: E402
from dhcm_ng.model import Kind, Tree                # noqa: E402

import tiktoken                                      # noqa: E402
_ENC = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
    return len(_ENC.encode(text or ""))


# --- gold ground truth (analysis / oracle only; never fed to retrieval) ---------
def gold_files_from_patch(patch: str) -> list[str]:
    return sorted({l[6:].strip() for l in patch.split("\n") if l.startswith("+++ b/")} - {"/dev/null"})


def gold_hunks_from_patch(patch: str) -> dict[str, list[tuple[int, int]]]:
    out, cur = {}, None
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
        elif line.startswith("@@") and cur:
            m = re.search(r"@@ -(\d+)(?:,(\d+))?", line)
            if m:
                s, c = int(m.group(1)), int(m.group(2) or 1)
                out.setdefault(cur, []).append((s, s + max(c, 1) - 1))
    return out


def gold_function_units(tree: Tree, gold_hunks: dict[str, list[tuple[int, int]]]) -> set[str]:
    units = set()
    funcs = [n for n in tree.by_kind(Kind.FUNCTION) if not n.children]
    for gfile, ranges in gold_hunks.items():
        gf = gfile.replace("\\", "/")
        for fn in funcs:
            if fn.path.replace("\\", "/").endswith(gf):
                if any(not (fn.end_line < a or fn.start_line > b) for a, b in ranges):
                    units.add(fn.id)
    return units


# --- retrieval + role classification (issue-text only) --------------------------
def _node_text(n) -> str:
    return f"{n.name} {n.signature} {(n.docstring or '')[:200]}"


def _is_test_path(path: str) -> bool:
    """Careful test detection (component/prefix/suffix, NOT the `_pytest` substring bug)."""
    p = path.replace("\\", "/")
    parts = p.split("/")
    if any(part in ("test", "tests", "testing") for part in parts[:-1]):
        return True
    base = parts[-1]
    return base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"


def retrieve(tree: Tree, embedder, issue: str, top_k: int = 3, n_relevant: int = 8,
             exclude_tests: bool = True):
    cands = [n for n in tree.nodes.values()
             if ((n.kind == Kind.FUNCTION and not n.children) or n.kind == Kind.CLASS)
             and not (exclude_tests and _is_test_path(n.path))]
    embs = embedder.embed_batch([_node_text(n) for n in cands])
    q = embedder.embed(issue[:1000])
    ranked = sorted(zip((cosine(q, e) for e in embs), cands), key=lambda x: -x[0])
    ranked_nodes = [n for _, n in ranked]
    locus = ranked_nodes[:top_k]
    locus_ids = {n.id for n in locus}
    enclosing = []
    for n in locus:
        if n.kind == Kind.FUNCTION and n.parent and tree.nodes[n.parent].kind == Kind.CLASS:
            enclosing.append(tree.nodes[n.parent])
    enc_ids = {n.id for n in enclosing}
    relevant = [n for n in ranked_nodes if n.id not in locus_ids and n.id not in enc_ids][:n_relevant]
    return {"ranked": ranked_nodes, "locus": locus, "enclosing": enclosing, "relevant": relevant}


# --- rung rendering -------------------------------------------------------------
def render(node, rung: str) -> str:
    if rung == "exclude":
        return ""
    if rung == "drop":
        return f"# {node.kind.value} {node.name} (omitted)"
    if rung == "name":
        return f"{node.kind.value} {node.name}"
    if rung == "full":
        return node.source or node.name
    if rung == "uml":
        return uml_class_skeleton(node.source) if node.kind == Kind.CLASS else (node.signature or node.name)
    if rung == "signature":
        if node.kind == Kind.CLASS:
            return uml_class_skeleton(node.source)
        return function_signature(node.source) if node.source else (node.signature or node.name)
    return node.source or node.name


def _file_full_source(repo_path: Path, rel: str) -> str:
    p = repo_path / rel
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _assemble(items: list[tuple[str, str]], budget: int | None) -> tuple[str, int]:
    """items: (header, body). Pack until budget (None = no cap)."""
    parts, used = [], 0
    for header, body in items:
        if not body:
            continue
        chunk = f"### {header}\n{body}"
        t = ntok(chunk)
        if budget is not None and used + t > budget and parts:
            continue
        parts.append(chunk)
        used += t
    text = "\n\n".join(parts)
    return text, ntok(text)


def build_arm(tree, repo_path: Path, ret: dict, arm: str, gold_units: set[str],
              budget: int | None = None) -> tuple[str, int]:
    locus, enclosing, relevant = ret["locus"], ret["enclosing"], ret["relevant"]

    if arm == "full_file":
        files = []
        seen = set()
        for n in locus + relevant:
            if n.path not in seen:
                seen.add(n.path)
                files.append((n.path, _file_full_source(repo_path, n.path)))
        return _assemble(files, budget)

    if arm == "tier_selector":
        items = [(f"{n.path}:{n.name}", render(n, "full")) for n in locus]
        items += [(f"{n.path}:{n.name}", render(n, "uml")) for n in enclosing]
        items += [(f"{n.path}:{n.name}", render(n, "signature")) for n in relevant]
        return _assemble(items, budget)

    if arm == "uniform_r3":
        items = [(f"{n.path}:{n.name}", render(n, "signature")) for n in (locus + enclosing + relevant)]
        return _assemble(items, budget)

    if arm == "oracle_minimal":
        gold_nodes = [tree.nodes[i] for i in gold_units]
        items = [(f"{n.path}:{n.name}", render(n, "full")) for n in gold_nodes]
        encl = {tree.nodes[n.parent].id for n in gold_nodes
                if n.parent and tree.nodes[n.parent].kind == Kind.CLASS}
        items += [(f"{tree.nodes[i].path}:{tree.nodes[i].name}", render(tree.nodes[i], "uml")) for i in encl]
        return _assemble(items, budget)

    if arm == "keep_drop":
        items = [(f"{n.path}:{n.name}", render(n, "full")) for n in locus]
        items += [(f"{n.path}:{n.name}", render(n, "drop")) for n in (enclosing + relevant)]
        return _assemble(items, budget)

    raise ValueError(f"unknown arm: {arm}")


ARMS = ["full_file", "tier_selector", "uniform_r3", "oracle_minimal", "keep_drop"]

# --- ORACLE-localized arms: localization FIXED (gold files known), vary ONLY representation.
# These isolate the representation-tier thesis from the retrieval confound (the smoke finding).
ORACLE_ARMS = ["oracle_full", "oracle_tier", "oracle_keep_drop"]


def _elements_in_files(tree, files) -> list:
    fs = [f.replace("\\", "/") for f in files]
    return [n for n in tree.nodes.values()
            if n.kind in (Kind.FUNCTION, Kind.CLASS)
            and any(n.path.replace("\\", "/").endswith(f) for f in fs)]


def build_oracle_arms(tree, repo_path, gold_files: list[str], gold_units: set[str],
                      budget: int | None = None) -> dict:
    """Three representation variants of the SAME (gold) localization:
      oracle_full      : gold files at full source (perfect localization, no compression)
      oracle_tier      : gold-edited fns FULL + enclosing class UML + other file elements at signature
      oracle_keep_drop : gold-edited fns FULL + everything else in the gold files DROPPED (SWEzze-style)
    The (tier vs keep_drop) gap at iso-resolve is the centerpiece; (tier vs full) is the cost win."""
    repo_path = Path(repo_path)
    gold_nodes = [tree.nodes[i] for i in gold_units]
    gold_ids = {n.id for n in gold_nodes}
    encl_ids = {n.parent for n in gold_nodes if n.parent and tree.nodes[n.parent].kind == Kind.CLASS}
    els = _elements_in_files(tree, gold_files)

    of = _assemble([(f, _file_full_source(repo_path, f)) for f in gold_files], budget)

    tier_items = [(f"{n.path}:{n.name}", render(n, "full")) for n in gold_nodes]
    tier_items += [(f"{tree.nodes[i].path}:{tree.nodes[i].name}", render(tree.nodes[i], "uml")) for i in encl_ids]
    tier_items += [(f"{n.path}:{n.name}", render(n, "signature")) for n in els
                   if n.id not in gold_ids and n.id not in encl_ids and n.parent not in encl_ids]
    ot = _assemble(tier_items, budget)

    kd_items = [(f"{n.path}:{n.name}", render(n, "full")) for n in gold_nodes]
    kd_items += [(f"{n.path}:{n.name}", render(n, "drop")) for n in els if n.id not in gold_ids]
    okd = _assemble(kd_items, budget)

    return {"oracle_full": {"tokens": of[1], "text": of[0]},
            "oracle_tier": {"tokens": ot[1], "text": ot[0]},
            "oracle_keep_drop": {"tokens": okd[1], "text": okd[0]}}


def build_all_arms(repo_path, issue: str, patch: str, embedder=None, top_k=3, n_relevant=8,
                   budget: int | None = None) -> dict:
    repo_path = Path(repo_path)
    tree = extract_tree(repo_path)
    embedder = embedder or STEmbedder()
    ret = retrieve(tree, embedder, issue, top_k=top_k, n_relevant=n_relevant)
    gold_units = gold_function_units(tree, gold_hunks_from_patch(patch))
    out = {}
    for arm in ARMS:
        text, toks = build_arm(tree, repo_path, ret, arm, gold_units, budget=budget)
        out[arm] = {"tokens": toks, "text": text}
    out["_meta"] = {"gold_files": gold_files_from_patch(patch), "gold_units": len(gold_units),
                    "locus": [n.name for n in ret["locus"]]}
    return out
