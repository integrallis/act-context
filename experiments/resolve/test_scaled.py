"""Tests for exp-014b v2: stats, fixed patch builder, coverage-fair arms, frozen sample.
No API, no Docker."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from stats import mcnemar_exact, paired_mcnemar, wilson  # noqa: E402
import edit_patch_v2 as ep  # noqa: E402
import arms_v2 as av  # noqa: E402


# --- stats ----------------------------------------------------------------------
def test_wilson_matches_paper_cells():
    assert tuple(round(x, 2) for x in wilson(15, 15)) == (0.80, 1.00)
    assert tuple(round(x, 2) for x in wilson(0, 15)) == (0.00, 0.20)
    assert tuple(round(x, 2) for x in wilson(1, 3)) == (0.06, 0.79)
    assert wilson(0, 0) == (0.0, 0.0)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    r = paired_mcnemar({"i1", "i2", "i3"}, {"i2", "i4"}, {"i1", "i2", "i3", "i4", "i5"})
    assert (r["b_only_A"], r["c_only_B"], r["discordant"]) == (2, 1, 3)


# --- frozen sample (v2: whole pool) -----------------------------------------------
@pytest.mark.skipif(not (HERE / "scaled_instances.json").exists(),
                    reason="run select_instances.py first")
def test_frozen_sample_invariants():
    spec = json.loads((HERE / "scaled_instances.json").read_text())
    assert spec["protocol_version"] == 2
    assert spec["N"] == len(spec["instances"]) == spec["multi_file_pool_size"]
    assert len(set(spec["instances"])) == spec["N"]
    assert spec["N"] >= 70  # the whole multi-file pool, not a subset


# --- edit_patch_v2 ----------------------------------------------------------------
def test_parse_tolerates_prose_and_strips_ab_prefix():
    raw = """The fix is to rename the variable.

*** FILE: a/astropy/io/votable/x.py
*** SEARCH
old line
*** REPLACE
new line
*** END

Done."""
    edits = ep.parse_edits(raw)
    assert len(edits) == 1
    assert edits[0]["file"] == "astropy/io/votable/x.py"  # the old lstrip('ab/') gave 'stropy/...'
    assert edits[0]["search"] == "old line"


def test_parse_empty_search_block():
    raw = "*** FILE: pkg/new_module.py\n*** SEARCH\n*** REPLACE\nx = 1\n*** END"
    edits = ep.parse_edits(raw)
    assert len(edits) == 1 and edits[0]["search"].strip() == "" and edits[0]["replace"] == "x = 1"


def test_apply_statuses(tmp_path):
    (tmp_path / "m.py").write_text("a = 1\nb = 2\na = 1\n")
    # ambiguous: 'a = 1' occurs twice
    state, res = ep.apply_edits_detailed(tmp_path, [{"file": "m.py", "search": "a = 1", "replace": "a = 9"}])
    assert res[0]["status"] == "ambiguous" and not state
    # unique match applies
    state, res = ep.apply_edits_detailed(tmp_path, [{"file": "m.py", "search": "b = 2", "replace": "b = 7"}])
    assert res[0]["status"] == "matched" and "b = 7" in state["m.py"]
    # second round applies on top of first-round state
    state, res = ep.apply_edits_detailed(tmp_path, [{"file": "m.py", "search": "b = 7", "replace": "b = 8"}], state)
    assert res[0]["status"] == "matched" and "b = 8" in state["m.py"]
    # missing file -> no_file; missing text -> no_match
    _, res = ep.apply_edits_detailed(tmp_path, [{"file": "nope.py", "search": "x", "replace": "y"}])
    assert res[0]["status"] == "no_file"
    _, res = ep.apply_edits_detailed(tmp_path, [{"file": "m.py", "search": "zzz", "replace": "y"}])
    assert res[0]["status"] == "no_match"


def test_new_file_diff_git_applies(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text("a = 1\nb = 2\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    edits = [{"file": "m.py", "search": "b = 2", "replace": "b = 3"},
             {"file": "pkg/new.py", "search": "", "replace": "x = 1"}]
    state, res = ep.apply_edits_detailed(tmp_path, edits)
    assert [r["status"] for r in res] == ["matched", "created"]
    patch = ep.make_diff(tmp_path, state)
    assert "new file mode 100644" in patch and "+++ b/pkg/new.py" in patch
    p = tmp_path / "fix.patch"
    p.write_text(patch)
    subprocess.run(["git", "apply", "--check", str(p)], cwd=tmp_path, check=True)


# --- arms_v2: preamble, gold units, coverage --------------------------------------
_SRC = '''"""Mod doc."""
import os

LIMIT = 10


class C:
    attr = 1

    def m(self):
        return LIMIT


def f():
    return os.sep


if __name__ == "__main__":
    f()
'''


def test_module_preamble_segments():
    segs = av.module_preamble_segments(_SRC)
    text = "\n".join(t for _, _, t in segs)
    assert "import os" in text and "LIMIT = 10" in text and '__main__' in text
    assert "def m" not in text and "def f" not in text and "class C" not in text


def test_gold_units_v2_class_attribute_scope(tmp_path):
    from dhcm_ng.extract import extract_tree
    (tmp_path / "mod.py").write_text(_SRC)
    tree = extract_tree(tmp_path)
    attr_line = _SRC.split("\n").index("    attr = 1") + 1
    units = av.gold_units_v2(tree, {"mod.py": {attr_line}})
    assert any(i.startswith("class:") for i in units)  # class promoted, not silently dropped
    m_line = _SRC.split("\n").index("        return LIMIT") + 1
    units = av.gold_units_v2(tree, {"mod.py": {m_line}})
    assert any(i.startswith("function:") and ":C.m:" in i for i in units)


def test_arms_v2_compressed_arms_cover_module_scope_and_non_py(tmp_path):
    from dhcm_ng.extract import extract_tree
    (tmp_path / "mod.py").write_text(_SRC)
    (tmp_path / "setup.cfg").write_text("[flake8]\nmax-line-length = 100\n")
    tree = extract_tree(tmp_path)
    arms = av.build_oracle_arms_v2(tree, tmp_path, ["mod.py", "setup.cfg"],
                                   {"mod.py": {4}})  # LIMIT line: module scope
    for a in ("oracle_tier", "oracle_keep_drop"):
        assert "LIMIT = 10" in arms[a]["text"]          # module preamble rendered
        assert "max-line-length" in arms[a]["text"]     # non-.py gold file rendered


def test_hunk_preimages_and_coverage():
    patch = """diff --git a/m.py b/m.py
--- a/m.py
+++ b/m.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 3
 c = 4
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1 @@
+x = 1
"""
    pre = av.hunk_preimages(patch)
    assert len(pre) == 2
    assert pre[0]["requirements"] == ["b = 2"] and not pre[0]["new_file"]  # the edit run's '-' lines
    assert pre[1]["new_file"]
    rep = av.coverage_report("### m.py (excerpt)\nb = 2\n", pre)  # only the edited line needed
    assert rep["full_coverage"] and rep["hunks"] == 1 and rep["new_file_hunks"] == 1
    rep2 = av.coverage_report("### m.py\nsomething else\n", pre)
    assert not rep2["full_coverage"] and rep2["uncovered"]


def test_hunk_requirements_insertion_anchor_and_boundary_context():
    # pure insertion -> anchored on nearest non-blank context line, NOT the whole hunk pre-image
    patch = """diff --git a/m.py b/m.py
--- a/m.py
+++ b/m.py
@@ -10,5 +10,7 @@
     def tail(self):
         return 1
+
+    def new_method(self):
+        return 2

     def next_method(self):
"""
    pre = av.hunk_preimages(patch)
    assert len(pre) == 1
    assert pre[0]["requirements"] == ["        return 1"]  # anchor line only
    # context that shows ONLY the edited method (per-element rendering) still counts as covered
    rep = av.coverage_report("### m.py:tail\n    def tail(self):\n        return 1\n", pre)
    assert rep["full_coverage"]
