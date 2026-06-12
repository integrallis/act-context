"""exp-014c: sequence-slice arm vs keep/drop — staged runner (pre-registered, frozen pre-data).

Registered hypothesis H3: `oracle_seq` (keep/drop content + Mermaid interaction slice) resolves
more instances than `oracle_keep_drop`, paired exact McNemar over the same 70 gold-valid multi-file
Verified instances as exp-014b. A tie is reported as the result.

Both arms are built and run FRESH in this experiment (same day, same resolved model), so the pair
is clean; the re-run keep/drop additionally gives a free stability check against exp-014b's
keep/drop (any drift is reported, not hidden). Protocol identical to exp-014b v2: T=0,
max_tokens=8000, <=2 retry rounds, no truncation, coverage-fair arms, gold-validated instances.

Stages:  plan (free) -> build (API: 2 arms x 70) -> eval (Docker: 140) -> analyze (free).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP14B = HERE.parent / "resolve"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP14B))
sys.path.insert(0, str(HERE.parents[1] / "src"))

RESULTS = HERE / "results"
RUN_PREFIX = "exp014c"
ARMS = ["oracle_seq", "oracle_keep_drop"]

import scaled_run as sr  # noqa: E402  (reuse protocol pieces: prompt, retries, checkout)


def _instances() -> list[str]:
    spec = json.loads((EXP14B / "scaled_instances.json").read_text())
    gold = json.loads((EXP14B / "results" / f"gold.{sr.RUN_PREFIX}_gold.json").read_text())
    excluded = set(gold.get("submitted_ids", [])) - set(gold.get("resolved_ids", []))
    return [i for i in spec["instances"] if i not in excluded]


def stage_build(temperature: float):
    from datasets import load_dataset

    import arms_v2 as av
    import seq_arm
    from dhcm_ng.context_arms import gold_files_from_patch
    from dhcm_ng.extract import extract_tree
    from dhcm_ng.llm import frontier_client

    RESULTS.mkdir(parents=True, exist_ok=True)
    run = _instances()
    ds = {x["instance_id"]: x for x in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    fr = frontier_client()
    preds = {a: [] for a in ARMS}
    metrics = {a: {} for a in ARMS}
    slice_meta = {}

    for n, iid in enumerate(run, 1):
        it = ds[iid]
        rp = sr._checkout(it["repo"], it["base_commit"])
        tree = extract_tree(rp)
        built = seq_arm.build_seq_arm(tree, rp, gold_files_from_patch(it["patch"]),
                                      av.gold_edited_lines(it["patch"]))
        arm_text = {"oracle_seq": built["text"], "oracle_keep_drop": built["keep_drop_text"]}
        slice_meta[iid] = {"has_slice": built["has_slice"], "slice_tokens": built["slice_tokens"],
                           "seq_tokens": built["tokens"], "kd_tokens": built["keep_drop_tokens"]}
        for a in ARMS:
            patch, m = sr._gen_with_retries(fr, rp, it["repo"], it["problem_statement"],
                                            arm_text[a], temperature)
            preds[a].append({"instance_id": iid, "model_name_or_path": a, "model_patch": patch})
            metrics[a][iid] = m
        print(f"[{n}/{len(run)}] {iid}: slice={built['slice_tokens']}t "
              f"seq ok={metrics['oracle_seq'][iid]['edits_applied']} | "
              f"kd ok={metrics['oracle_keep_drop'][iid]['edits_applied']}", flush=True)
        for a in ARMS:
            (RESULTS / f"preds_{a}.json").write_text(json.dumps(preds[a], indent=2))
        (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (RESULTS / "slice_meta.json").write_text(json.dumps(slice_meta, indent=2))

    (RESULTS / "run_meta.json").write_text(json.dumps({
        "resolved_model_id": fr.last_model_id, "temperature": temperature,
        "max_tokens": sr.MAX_TOKENS, "n_instances": len(run),
        "built_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"Built {len(run)} x 2 -> {RESULTS}")


def stage_eval():
    run = _instances()
    for a in ARMS:
        cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
               "-d", "princeton-nlp/SWE-bench_Verified", "-p", str(RESULTS / f"preds_{a}.json"),
               "-id", f"{RUN_PREFIX}_{a}", "-n", "swebench", "--max_workers", "4",
               "--cache_level", "env", "-i", *run]
        print("RUN:", " ".join(cmd[:8]), f"... x {len(run)}", flush=True)
        subprocess.run(cmd, check=False, timeout=21600, cwd=RESULTS)


def _status(arm, instances):
    out, empty = {}, set()
    pf = RESULTS / f"preds_{arm}.json"
    if pf.exists():
        empty = {p["instance_id"] for p in json.loads(pf.read_text())
                 if not (p.get("model_patch") or "").strip()}
    base = RESULTS / "logs" / "run_evaluation" / f"{RUN_PREFIX}_{arm}" / arm
    for iid in instances:
        rep = base / iid / "report.json"
        if not rep.exists():
            out[iid] = "unresolved" if iid in empty else "missing"
            continue
        try:
            out[iid] = "resolved" if json.loads(rep.read_text())[iid]["resolved"] else "unresolved"
        except Exception:
            out[iid] = "missing"
    return out


def stage_analyze():
    from stats import paired_mcnemar, wilson
    run = _instances()
    status = {a: _status(a, run) for a in ARMS}
    incomplete = {i for i in run if any(status[a][i] == "missing" for a in ARMS)}
    uni = [i for i in run if i not in incomplete]
    resolved = {a: {i for i in uni if status[a][i] == "resolved"} for a in ARMS}
    sm = json.loads((RESULTS / "slice_meta.json").read_text())

    lines = [f"# exp-014c sequence-slice vs keep/drop (N={len(uni)}; {len(incomplete)} eval-incomplete excluded)\n"]
    if incomplete:
        lines.append(f"⚠ incomplete: {sorted(incomplete)} — re-run eval for these.\n")
    lines.append("| arm | resolved | rate | Wilson 95% CI | mean ctx tok |")
    lines.append("|---|---|---|---|---|")
    for a in ARMS:
        k, n = len(resolved[a]), len(uni)
        lo, hi = wilson(k, n)
        key = "seq_tokens" if a == "oracle_seq" else "kd_tokens"
        mt = round(sum(sm[i][key] for i in uni) / max(1, len(uni)), 1)
        lines.append(f"| {a} | {k}/{n} | {k/n:.3f} | [{lo:.2f}, {hi:.2f}] | {mt} |")
    r = paired_mcnemar(resolved["oracle_seq"], resolved["oracle_keep_drop"], set(uni))
    lines.append(f"\n**H3 (seq > keep/drop), exact McNemar:** seq-only {r['b_only_A']}, "
                 f"kd-only {r['c_only_B']}, p = {r['p_value']:.4f}")
    with_slice = [i for i in uni if sm[i]["has_slice"]]
    lines.append(f"\nInstances with a non-empty slice: {len(with_slice)}/{len(uni)}; mean slice size "
                 f"{round(sum(sm[i]['slice_tokens'] for i in with_slice)/max(1,len(with_slice)),1)} tok.")
    # stability check vs exp-014b's keep_drop (same protocol, earlier day)
    b14 = EXP14B / "results" / "analysis.json"
    if b14.exists():
        old = set(json.loads(b14.read_text())["resolved_ids"]["oracle_keep_drop"]) & set(uni)
        new = resolved["oracle_keep_drop"]
        lines.append(f"\nStability: keep/drop re-run resolves {len(new)} vs exp-014b's "
                     f"{len(old)} on the same instances; symmetric difference "
                     f"{len(old ^ new)} (model/temporal drift indicator).")
    out = {"universe": uni, "incomplete": sorted(incomplete),
           "resolved_ids": {a: sorted(resolved[a]) for a in ARMS}, "mcnemar_h3": r}
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def stage_plan():
    run = _instances()
    print(f"exp-014c PLAN: {len(run)} instances x 2 arms (oracle_seq, oracle_keep_drop re-run)")
    print(f"  build -> {len(run)*2} agent calls (API $); eval -> {len(run)*2} Docker evals")
    print("  H3: seq > keep/drop, paired exact McNemar; tie reported as the result")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["plan", "build", "eval", "analyze"], default="plan")
    p.add_argument("--temperature", type=float, default=0.0)
    a = p.parse_args()
    {"plan": stage_plan, "build": lambda: stage_build(a.temperature),
     "eval": stage_eval, "analyze": stage_analyze}[a.stage]()
