"""Pre-data expressibility audit for exp-014b v2 (free: no API, no Docker).

For every frozen instance, build the v2 arms at base_commit and verify that each gold hunk is
expressible (every edit-run requirement present verbatim) in each arm. Run BEFORE any spend; the
result (results/coverage_sweep.json) is part of the pre-registration record.

Run:  uv run python experiments/014b-scaled-resolve/coverage_sweep.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import arms_v2 as av  # noqa: E402
from dhcm_ng.context_arms import gold_files_from_patch  # noqa: E402
from dhcm_ng.extract import extract_tree  # noqa: E402
from scaled_run import ARMS, _checkout, load_instances  # noqa: E402


def main():
    from datasets import load_dataset
    ds = {x["instance_id"]: x for x in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    instances = load_instances()
    out, fully = {}, {a: 0 for a in ARMS}
    for n, iid in enumerate(instances, 1):
        it = ds[iid]
        try:
            rp = _checkout(it["repo"], it["base_commit"])
            tree = extract_tree(rp)
            arms = av.build_oracle_arms_v2(tree, rp, gold_files_from_patch(it["patch"]),
                                           av.gold_edited_lines(it["patch"]))
            pre = av.hunk_preimages(it["patch"])
            row = {}
            for a in ARMS:
                r = av.coverage_report(arms[a]["text"], pre)
                r["tokens"] = arms[a]["tokens"]
                row[a] = r
                fully[a] += r["full_coverage"]
            out[iid] = row
            print(f"[{n}/{len(instances)}] {iid}: " +
                  " | ".join(f"{a.split('_',1)[1]} {row[a]['covered']}/{row[a]['hunks']}"
                             f"{'✓' if row[a]['full_coverage'] else '✗'} tok={row[a]['tokens']}"
                             for a in ARMS), flush=True)
        except Exception as e:
            out[iid] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{n}/{len(instances)}] {iid}: ERROR {e}", flush=True)
            traceback.print_exc()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "coverage_sweep.json").write_text(json.dumps(out, indent=1))
    print("\nfully-expressible instances per arm:",
          {a: f"{fully[a]}/{len(instances)}" for a in ARMS})


if __name__ == "__main__":
    main()
