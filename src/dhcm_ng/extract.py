"""Deterministic AST -> structural tree (Subsystem -> Component -> File -> Class -> Function).

Pure `ast`, no LLM, no regex for structure. This is the only "extraction" needed for the
H1'/H2' ablations; richer edge resolution (CALLS/IMPORTS) can be added later if needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .model import Kind, Node, Tree, content_hash

_DEFAULT_EXCLUDES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox",
                     "build", "dist", ".mypy_cache", ".pytest_cache"}


def _add(nodes: dict[str, Node], node: Node, parent: Node | None) -> None:
    nodes[node.id] = node
    if parent is not None:
        node.parent = parent.id
        parent.children.append(node.id)


def _functions_and_classes(src: str, rel: str, file_node: Node, nodes: dict[str, Node]) -> None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return

    def handle(stmt, parent: Node, qual: str) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src, stmt) or ""
            q = f"{qual}.{stmt.name}" if qual else stmt.name
            n = Node(
                id=f"function:{rel}:{q}:{stmt.lineno}", kind=Kind.FUNCTION, name=stmt.name,
                path=rel, source=seg, docstring=ast.get_docstring(stmt) or "",
                signature=seg.split("\n", 1)[0] if seg else f"def {stmt.name}(...)",
                start_line=stmt.lineno, end_line=getattr(stmt, "end_lineno", stmt.lineno),
                content_hash=content_hash(seg),
            )
            _add(nodes, n, parent)
        elif isinstance(stmt, ast.ClassDef):
            seg = ast.get_source_segment(src, stmt) or ""
            q = f"{qual}.{stmt.name}" if qual else stmt.name
            cls = Node(
                id=f"class:{rel}:{q}", kind=Kind.CLASS, name=stmt.name, path=rel, source=seg,
                docstring=ast.get_docstring(stmt) or "",
                signature=f"class {stmt.name}",
                start_line=stmt.lineno, end_line=getattr(stmt, "end_lineno", stmt.lineno),
                content_hash=content_hash(seg),
            )
            _add(nodes, cls, parent)
            for sub in stmt.body:
                handle(sub, cls, q)

    for stmt in tree.body:
        handle(stmt, file_node, "")


def extract_tree(repo_path: str | Path, exclude_dirs: set[str] | None = None) -> Tree:
    repo = Path(repo_path).resolve()
    excludes = _DEFAULT_EXCLUDES | (exclude_dirs or set())
    nodes: dict[str, Node] = {}
    root = Node(id=f"subsystem:{repo.name}", kind=Kind.SUBSYSTEM, name=repo.name, path=str(repo))
    nodes[root.id] = root

    components: dict[str, Node] = {}
    for py in sorted(repo.rglob("*.py")):
        rel = py.relative_to(repo).as_posix()
        if any(part in excludes for part in py.relative_to(repo).parts):
            continue
        comp_key = py.relative_to(repo).parts[0] if len(py.relative_to(repo).parts) > 1 else "."
        if comp_key not in components:
            comp = Node(id=f"component:{comp_key}", kind=Kind.COMPONENT, name=comp_key, path=comp_key)
            _add(nodes, comp, root)
            components[comp_key] = comp
        comp = components[comp_key]
        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fnode = Node(id=f"file:{rel}", kind=Kind.FILE, name=py.name, path=rel,
                     content_hash=content_hash(src))
        _add(nodes, fnode, comp)
        _functions_and_classes(src, rel, fnode, nodes)

    _compute_subtree_hashes(Tree(nodes, root.id))
    return Tree(nodes, root.id)


def _compute_subtree_hashes(tree: Tree) -> None:
    """Bottom-up subtree hash: own content + sorted child subtree hashes. Enables
    content-hash incremental re-summarization (only changed subtrees re-run)."""
    def visit(nid: str) -> str:
        n = tree.nodes[nid]
        if not n.children:
            n.subtree_hash = n.content_hash or content_hash(n.id)
            return n.subtree_hash
        child_hashes = sorted(visit(c) for c in n.children)
        n.subtree_hash = content_hash(n.content_hash + "|" + "|".join(child_hashes))
        return n.subtree_hash

    visit(tree.root)
