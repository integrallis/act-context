"""Tests for exp-015: kappa, probe parsing, grade parsing, frozen-selection invariants.
No API, no network."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from regrade_run import _parse_probe_json, ARMS  # noqa: E402
from stats_kappa import cohen_kappa  # noqa: E402


def test_kappa_perfect_and_chance():
    assert cohen_kappa([(True, True), (False, False)] * 5) == pytest.approx(1.0)
    # independent coin-flip-like marginals with 50% agreement -> kappa ~ 0
    pairs = [(True, True), (True, False), (False, True), (False, False)] * 5
    assert abs(cohen_kappa(pairs)) < 1e-9
    assert cohen_kappa([]) == 0.0
    assert cohen_kappa([(True, True)] * 4) == 1.0          # constant-and-identical edge


def test_kappa_known_value():
    # 2x2 table: a=20 both-yes, b=5, c=10, d=15 -> po=.7, pe=.5, kappa=.4
    pairs = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert cohen_kappa(pairs) == pytest.approx(0.4)


def test_parse_probe_json_strict_and_wrapped():
    raw = '[{"question": "What does X return?", "reference": "An int."}]'
    assert _parse_probe_json(raw, 3) == [{"question": "What does X return?", "reference": "An int."}]
    wrapped = "Here you go:\n```json\n" + raw + "\n```\nDone."
    assert len(_parse_probe_json(wrapped, 3)) == 1
    assert _parse_probe_json("no json here", 3) == []
    many = json.dumps([{"question": f"q{i}", "reference": f"r{i}"} for i in range(6)])
    assert len(_parse_probe_json(many, 3)) == 3            # capped at k


def test_grade_verdict_parse():
    # mirrors the regex used in stage_grade
    ok = lambda v: bool(re.fullmatch(r"\W*CORRECT\W*", v.strip().upper()))
    assert ok("CORRECT") and ok(" correct.") and ok("**CORRECT**")
    assert not ok("INCORRECT") and not ok("CORRECT-ish") and not ok("the answer is CORRECT")


@pytest.mark.skipif(not (HERE / "regrade_nodes.json").exists(),
                    reason="run select_nodes.py first")
def test_frozen_selection_invariants():
    spec = json.loads((HERE / "regrade_nodes.json").read_text())
    assert len(spec["repos"]) == 3
    for r in spec["repos"]:
        assert 1 <= len(r["nodes"]) <= spec["n_per_repo"]
        for n in r["nodes"]:
            assert n["test_file"].endswith(".py")
            assert not n["name"].startswith("_")
            assert n["n_methods"] >= 3
    ids = [n["id"] for r in spec["repos"] for n in r["nodes"]]
    assert len(ids) == len(set(ids))
    assert set(ARMS) == {"source", "summary_qwen3b", "summary_frontier", "signature"}
