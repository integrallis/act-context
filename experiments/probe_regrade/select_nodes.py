"""exp-015: deterministic, pre-registered node selection for the de-circularized probe re-grade.

Fixes audit D1 (probe circularity / self-judging / authors'-own-repos) by selecting HELD-OUT
classes from three real repos (pinned to the SWE-bench base commits already used in this project)
such that each class has a matching TEST FILE — probes will be minted from the tests (independent
behavioral ground truth), never from the representation under test.

Selection is a fixed function of the repos (no randomness, no API):
  - candidate classes: Kind.CLASS, >=3 method children, 40 <= line span <= 250, non-test path,
    public name (no leading underscore)
  - must have a matching test file: a test_*.py / *_test.py file in the repo containing the class
    name as a word, 30..2500 lines; smallest such file wins (most focused tests)
  - sort candidates by (path, name); take the first N_PER_REPO
Re-running yields byte-identical output.

Run:  uv run python experiments/015-probe-regrade/select_nodes.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "resolve"))

N_PER_REPO = 5
OUT = HERE / "regrade_nodes.json"

# pinned via the SWE-bench Verified instances this project already uses (base_commit lookup)
REPO_PINS = ["mwaskom__seaborn-3187", "pylint-dev__pylint-6528", "pytest-dev__pytest-8399"]


def _is_test_path(p: str) -> bool:
    parts = p.replace("\\", "/").split("/")
    if any(x in ("test", "tests", "testing") for x in parts[:-1]):
        return True
    b = parts[-1]
    return b.startswith("test_") or b.endswith("_test.py") or b == "conftest.py"


def _test_files(repo: Path) -> list[Path]:
    out = []
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        if _is_test_path(rel):
            try:
                n = p.read_text(encoding="utf-8", errors="ignore").count("\n")
            except OSError:
                continue
            if 30 <= n <= 2500:
                out.append(p)
    return out


def select_for_repo(repo_path: Path) -> list[dict]:
    from dhcm_ng.extract import extract_tree
    from dhcm_ng.model import Kind

    tree = extract_tree(repo_path)
    tests = _test_files(repo_path)
    test_text = {t: t.read_text(encoding="utf-8", errors="ignore") for t in tests}

    chosen = []
    classes = sorted((n for n in tree.by_kind(Kind.CLASS)), key=lambda n: (n.path, n.name))
    for n in classes:
        if len(chosen) >= N_PER_REPO:
            break
        if _is_test_path(n.path) or n.name.startswith("_"):
            continue
        methods = [c for c in n.children if c.startswith("function:")]
        span = (n.end_line or 0) - (n.start_line or 0)
        if len(methods) < 3 or not (40 <= span <= 250):
            continue
        pat = re.compile(rf"\b{re.escape(n.name)}\b")
        matches = [t for t in tests if pat.search(test_text[t])]
        if not matches:
            continue
        tf = min(matches, key=lambda t: (len(test_text[t]), t.as_posix()))
        chosen.append({"id": n.id, "name": n.name, "path": n.path,
                       "start_line": n.start_line, "end_line": n.end_line,
                       "n_methods": len(methods),
                       "test_file": tf.relative_to(repo_path).as_posix()})
    return chosen


def main():
    from datasets import load_dataset
    from scaled_run import _checkout  # exp-014b's pinned checkout helper

    ds = {x["instance_id"]: x for x in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    spec = {"experiment": "exp-015-probe-regrade", "n_per_repo": N_PER_REPO, "repos": []}
    for iid in REPO_PINS:
        it = ds[iid]
        rp = _checkout(it["repo"], it["base_commit"])
        nodes = select_for_repo(rp)
        spec["repos"].append({"repo": it["repo"], "base_commit": it["base_commit"],
                              "pin_instance": iid, "nodes": nodes})
        print(f"{it['repo']}: {len(nodes)} nodes -> {[n['name'] for n in nodes]}")
    OUT.write_text(json.dumps(spec, indent=2))
    print(f"Froze {sum(len(r['nodes']) for r in spec['repos'])} nodes -> {OUT}")


if __name__ == "__main__":
    main()
