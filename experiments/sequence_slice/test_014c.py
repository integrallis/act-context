"""Tests for exp-014c: call-slice extraction + arm construction. No API, no Docker."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resolve"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import arms_v2 as av  # noqa: E402
import seq_arm  # noqa: E402

_SRC = '''import os

def helper(x):
    return x + 1


def gold_fn(a):
    b = helper(a)
    c = other(b)
    return Widget(c)


def other(b):
    return b * 2


def caller():
    return gold_fn(3)


class Widget:
    def __init__(self, c):
        self.c = c

    def render(self):
        return str(self.c)
'''


def _tree(tmp_path):
    from dhcm_ng.extract import extract_tree
    (tmp_path / "mod.py").write_text(_SRC)
    return extract_tree(tmp_path)


def test_call_slice_outgoing_and_incoming(tmp_path):
    tree = _tree(tmp_path)
    gold_line = _SRC.split("\n").index("def gold_fn(a):") + 2  # a line inside gold_fn
    units = av.gold_units_v2(tree, {"mod.py": {gold_line}})
    assert any(":gold_fn:" in u for u in units)
    sl = seq_arm.call_slice(tmp_path, ["mod.py"], tree, units)
    assert "sequenceDiagram" in sl
    assert "gold_fn->>helper" in sl and "gold_fn->>other" in sl    # outgoing, resolved in-file
    assert "gold_fn->>Widget" in sl                                # class constructor call
    assert "caller->>gold_fn" in sl                                # incoming
    # unresolvable names (str) and self-calls never appear
    assert "->>str" not in sl


def test_seq_arm_is_keep_drop_plus_slice(tmp_path):
    tree = _tree(tmp_path)
    gold_line = _SRC.split("\n").index("    b = helper(a)") + 1
    built = seq_arm.build_seq_arm(tree, tmp_path, ["mod.py"], {"mod.py": {gold_line}})
    assert built["has_slice"] and built["slice_tokens"] > 0
    assert built["text"].startswith(built["keep_drop_text"])       # exact superset: kd + slice only
    assert "Interaction slice" in built["text"]
    assert "Interaction slice" not in built["keep_drop_text"]


def test_slice_cap_not_silent(tmp_path):
    # many distinct callees -> overflow note appears instead of silent truncation
    body = "\n".join(f"    f{i}(x)" for i in range(60))
    defs = "\n\n".join(f"def f{i}(x):\n    return x" for i in range(60))
    src = f"def gold_fn(x):\n{body}\n\n{defs}\n"
    (tmp_path / "big.py").write_text(src)
    from dhcm_ng.extract import extract_tree
    tree = extract_tree(tmp_path)
    units = av.gold_units_v2(tree, {"big.py": {2}})
    sl = seq_arm.call_slice(tmp_path, ["big.py"], tree, units)
    assert "more call edges omitted" in sl
    assert sl.count("->>") == seq_arm.MAX_MESSAGES
