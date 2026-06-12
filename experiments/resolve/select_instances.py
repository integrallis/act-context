"""Deterministic, pre-registered instance selection for exp-014b.

PROTOCOL v2 (amended 2026-06-09 BEFORE any agent call or eval — see scaled_run.py docstring and
src-ng/paper_audit/SOTA_FAILURE_AUDIT.md E1): the sample is ALL multi-file SWE-bench Verified
instances (gold patch touches >= 2 distinct files). v1 had selected a 30-of-71 round-robin subset;
the audit found that underpowered for the registered McNemar test while saving almost nothing, so
the pool is taken whole. No randomness, no API; re-running yields byte-identical output. Ordering:
repos sorted, instance_ids sorted within repo (ordering is cosmetic — every instance runs).

Run:  uv run python experiments/014b-scaled-resolve/select_instances.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "scaled_instances.json"

_DIFF_FILE = re.compile(r"^diff --git a/(.+?) b/", re.M)


def gold_files(patch: str) -> set[str]:
    return set(_DIFF_FILE.findall(patch))


def select() -> dict:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    by_repo: dict[str, list[str]] = defaultdict(list)
    for x in ds:
        if len(gold_files(x["patch"])) >= 2:
            by_repo[x["repo"]].append(x["instance_id"])
    for r in by_repo:
        by_repo[r].sort()

    selected = [iid for r in sorted(by_repo) for iid in by_repo[r]]
    dist = {r: len(by_repo[r]) for r in sorted(by_repo)}

    return {
        "experiment": "exp-014b",
        "protocol_version": 2,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "criterion": ">=2 gold files (multi-file)",
        "selection": "ALL multi-file instances (entire pool; v2 amendment pre-data, was 30-of-71)",
        "N": len(selected),
        "repo_distribution": dist,
        "multi_file_pool_size": len(selected),
        "instances": selected,
    }


if __name__ == "__main__":
    spec = select()
    OUT.write_text(json.dumps(spec, indent=2))
    print(f"Froze {spec['N']} instances -> {OUT}")
    print("repo distribution:", spec["repo_distribution"])
