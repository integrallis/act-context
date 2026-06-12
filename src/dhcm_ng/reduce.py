"""Deterministic reduction operators over standard models (UML/C4 style).

These are the act-view "rungs": full code -> UML class skeleton (signatures, no bodies) ->
signatures -> name. AST-based, no LLM. Used to test whether a UML/structured representation
carries equal task-relevant information at lower token cost than raw code (the act-view gate).
"""

from __future__ import annotations

import ast


def uml_class_skeleton(class_source: str) -> str:
    """Class -> UML-style skeleton: class header + attributes + method signatures (NO bodies).

    The 'cognitive reduction' of a class diagram: keep structure (names, fields, signatures,
    bases), drop implementation."""
    try:
        mod = ast.parse(class_source)
        cls = next(n for n in mod.body if isinstance(n, ast.ClassDef))
    except (SyntaxError, StopIteration):
        return class_source.split("\n", 1)[0] if class_source else ""

    bases = ", ".join(ast.unparse(b) for b in cls.bases) if cls.bases else ""
    head = f"class {cls.name}({bases}):" if bases else f"class {cls.name}:"
    lines = [head]

    # attributes: annotated/assigned names at class body level
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            ann = f": {ast.unparse(stmt.annotation)}" if stmt.annotation else ""
            lines.append(f"    {stmt.target.id}{ann}")
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    lines.append(f"    {t.id}")

    # method signatures (no bodies)
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                args = ast.unparse(stmt.args)
            except Exception:
                args = "..."
            ret = f" -> {ast.unparse(stmt.returns)}" if stmt.returns else ""
            prefix = "async def" if isinstance(stmt, ast.AsyncFunctionDef) else "def"
            doc = ast.get_docstring(stmt)
            first = (doc.strip().splitlines()[0] if doc and doc.strip() else "")
            tail = f"  # {first}" if first else ""
            lines.append(f"    {prefix} {stmt.name}({args}){ret}{tail}")

    if len(lines) == 1:
        lines.append("    ...")
    return "\n".join(lines)


def function_signature(func_source: str) -> str:
    """Function -> signature + docstring first line (no body)."""
    try:
        mod = ast.parse(func_source)
        fn = next(n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except (SyntaxError, StopIteration):
        return func_source.split("\n", 1)[0] if func_source else ""
    args = ast.unparse(fn.args)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    doc = ast.get_docstring(fn)
    first = (doc.strip().splitlines()[0] if doc and doc.strip() else "")
    return f"def {fn.name}({args}){ret}" + (f'\n    """{first}"""' if first else "")
