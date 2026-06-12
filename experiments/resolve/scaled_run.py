"""exp-014b v2: pre-registered resolve@cost run (oracle localization, all 71 multi-file, T=0).

PROTOCOL v2 — amended BEFORE any data collection, per the failure audit
(src-ng/paper_audit/SOTA_FAILURE_AUDIT.md §P0). Changes vs v1 (all instrument fixes, frozen now):
  1. NO context truncation (was: silent context[:50000] that blinded only oracle_full) and no
     issue truncation; real sizes logged per arm x instance.
  2. Coverage-fair arms (arms_v2): module preamble + class-attribute units + non-.py gold files
     rendered in the compressed arms; new-file edits expressible in every arm.
  3. Per arm x instance COVERAGE instrumentation (gold-hunk pre-image present in context?);
     analysis reported overall AND on the all-arms-covered subset.
  4. Edit retry loop: up to 2 feedback rounds on no_match/ambiguous/no_file edits
     (edit_patch_v2; unmatched edits are no longer silently dropped without recourse).
  5. Patch builder fixes: anchored a/ b/ strip, uniqueness required, new-file creation.
  6. Reasoning permitted before the edit blocks (parser is prose-tolerant); max_tokens 8000.
  7. Costs: cl100k on context + Anthropic count_tokens on the exact INITIAL prompt per episode
     (retry prompts are logged in metrics but not token-counted).
  8. Gold-patch validation for ALL instances (stage validate_gold) BEFORE build; instances whose
     gold patch fails the harness are excluded (recorded; exclusion is environment failure, not
     model failure).
  9. Sample = ALL 71 multi-file Verified instances (was 30-of-71).
Registered hypothesis H2 unchanged: oracle_tier resolves more than oracle_keep_drop; a tie is
reported as the result. Arms share gold-file localization; only representation varies.

STAGES (nothing heavy runs by accident):
  uv run python scaled_run.py --stage plan            # free dry-run (default)
  uv run python scaled_run.py --stage validate_gold   # Docker: gold patches through the harness
  uv run python scaled_run.py --stage build           # API spend: agent calls (3 arms x N, +retries)
  uv run python scaled_run.py --stage eval            # Docker: 3 x N evals
  uv run python scaled_run.py --stage analyze         # free -> results/ANALYSIS.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

RESULTS = HERE / "results"
WORK = Path("/tmp/loc_pareto_work")
RUN_PREFIX = "exp014b"
MAX_TOKENS = 8000
MAX_RETRY_ROUNDS = 2
ARMS = ["oracle_full", "oracle_tier", "oracle_keep_drop"]

_EDIT_PROMPT = """You are fixing a GitHub issue in the `{repo}` repository.

ISSUE:
{issue}

RELEVANT CODE (already localized for you):
{context}

Think briefly about the fix first if that helps. Then output ONE OR MORE edit blocks in EXACTLY
this format (the blocks are parsed mechanically; everything outside them is ignored):
*** FILE: <path relative to repo root>
*** SEARCH
<exact existing lines to replace — copy them VERBATIM from the code above, incl. indentation>
*** REPLACE
<the new lines>
*** END

Rules:
- each SEARCH block must be copied verbatim from the provided code and match exactly ONE location;
  keep it small and uniquely matching.
- to CREATE a new file, emit a block whose SEARCH section is empty.
- edit only what is necessary to fix the issue."""

_RETRY_SUFFIX = """

YOUR PREVIOUS EDIT BLOCKS (for reference):
{prev}

These edits FAILED to apply:
{feedback}

The other edits were applied successfully. Re-emit corrected edit blocks ONLY for the failed edits,
in the same *** FILE / SEARCH / REPLACE / END format."""


def load_instances() -> list[str]:
    spec = json.loads((HERE / "scaled_instances.json").read_text())
    return spec["instances"]


def _checkout(repo: str, commit: str) -> Path:
    rp = WORK / repo.replace("/", "_")
    if not rp.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(rp)],
                       check=True, capture_output=True, timeout=900)
    subprocess.run(["git", "fetch", "origin", commit], cwd=rp, capture_output=True, timeout=300)
    subprocess.run(["git", "checkout", "-f", commit], cwd=rp, check=True, capture_output=True, timeout=120)
    return rp


def _gen_with_retries(fr, repo_path, repo, issue, context, temperature):
    """One agent attempt + up to MAX_RETRY_ROUNDS feedback rounds on failed edits.
    Returns (patch, metrics)."""
    from edit_patch_v2 import apply_edits_detailed, failure_feedback, make_diff, parse_edits

    prompt = _EDIT_PROMPT.format(repo=repo, issue=issue, context=context)
    state: dict[str, str] = {}
    all_results: list[dict] = []
    stop_reasons: list[str] = []
    emitted = 0
    raw = fr.generate(prompt, max_tokens=MAX_TOKENS, temperature=temperature)
    stop_reasons.append(fr.last_stop_reason)
    edits = parse_edits(raw or "")
    emitted += len(edits)
    state, results = apply_edits_detailed(repo_path, edits, state)
    all_results = results
    rounds = 0
    while rounds < MAX_RETRY_ROUNDS:
        failed = [r for r in all_results if r["status"] not in ("matched", "created")]
        if not failed or not edits:
            break
        rounds += 1
        retry_prompt = prompt + _RETRY_SUFFIX.format(prev=(raw or "")[-6000:],
                                                     feedback=failure_feedback(all_results))
        raw = fr.generate(retry_prompt, max_tokens=MAX_TOKENS, temperature=temperature)
        stop_reasons.append(fr.last_stop_reason)
        edits = parse_edits(raw or "")
        emitted += len(edits)
        state, results = apply_edits_detailed(repo_path, edits, state)
        ok_now = {(r["file"], r["search"]) for r in results if r["status"] in ("matched", "created")}
        all_results = [r for r in all_results
                       if r["status"] in ("matched", "created") or (r["file"], r["search"]) not in ok_now]
        all_results += [r for r in results if r["status"] in ("matched", "created")]

    patch = make_diff(repo_path, state)
    ok = [r for r in all_results if r["status"] in ("matched", "created")]
    metrics = {
        "edits_emitted": emitted,
        "edits_applied": len(ok),
        "edits_failed_final": len(all_results) - len(ok),
        "failure_reasons": sorted({r["status"] for r in all_results} - {"matched", "created"}),
        "retry_rounds": rounds,
        "stop_reasons": stop_reasons,
        "truncated_any": "max_tokens" in stop_reasons,
        "diff_nonempty": bool(patch.strip()),
        "patch_len": len(patch),
    }
    return patch, metrics


def _excluded_by_gold() -> set[str]:
    rep = RESULTS / f"gold.{RUN_PREFIX}_gold.json"
    if not rep.exists():
        return set()
    d = json.loads(rep.read_text())
    resolved = set(d.get("resolved_ids", []))
    submitted = set(d.get("submitted_ids", d.get("completed_ids", [])))
    return submitted - resolved


def stage_validate_gold(instances):
    """Run every instance's GOLD patch through the harness. Failures = broken environments ->
    pre-registered exclusion (recorded), not model failures."""
    from datasets import load_dataset
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = {x["instance_id"]: x for x in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    preds = [{"instance_id": i, "model_name_or_path": "gold", "model_patch": ds[i]["patch"]}
             for i in instances]
    (RESULTS / "preds_gold.json").write_text(json.dumps(preds, indent=2))
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "-d", "princeton-nlp/SWE-bench_Verified", "-p", str(RESULTS / "preds_gold.json"),
           "-id", f"{RUN_PREFIX}_gold", "-n", "swebench", "--max_workers", "4",
           "--cache_level", "env", "-i", *instances]
    print("RUN:", " ".join(cmd[:10]), f"... (-i x {len(instances)})")
    subprocess.run(cmd, check=False, timeout=21600, cwd=RESULTS)
    excl = _excluded_by_gold()
    print(f"gold validation: {len(instances) - len(excl)}/{len(instances)} environments OK; "
          f"excluded: {sorted(excl)}")


def stage_build(instances, temperature):
    from datasets import load_dataset

    import arms_v2 as av
    from dhcm_ng.context_arms import gold_files_from_patch
    from dhcm_ng.extract import extract_tree
    from dhcm_ng.llm import frontier_client

    gold_rep = RESULTS / f"gold.{RUN_PREFIX}_gold.json"
    if not gold_rep.exists():
        sys.exit("REFUSING to build: run --stage validate_gold first (pre-registered order).")
    excluded = _excluded_by_gold()
    run = [i for i in instances if i not in excluded]
    print(f"building {len(run)} instances ({len(excluded)} excluded by gold validation: {sorted(excluded)})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = {x["instance_id"]: x for x in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    fr = frontier_client()
    preds = {a: [] for a in ARMS}
    metrics = {a: {} for a in ARMS}
    costs = {a: {} for a in ARMS}
    coverage = {a: {} for a in ARMS}

    for n, iid in enumerate(run, 1):
        it = ds[iid]
        rp = _checkout(it["repo"], it["base_commit"])
        tree = extract_tree(rp)
        gold_files = gold_files_from_patch(it["patch"])
        oracle = av.build_oracle_arms_v2(tree, rp, gold_files, av.gold_edited_lines(it["patch"]))
        pre = av.hunk_preimages(it["patch"])
        issue = it["problem_statement"]
        for a in ARMS:
            ctx = oracle[a]["text"]
            coverage[a][iid] = av.coverage_report(ctx, pre)
            patch, m = _gen_with_retries(fr, rp, it["repo"], issue, ctx, temperature)
            m["context_chars"] = len(ctx)
            m["issue_chars"] = len(issue)
            preds[a].append({"instance_id": iid, "model_name_or_path": a, "model_patch": patch})
            metrics[a][iid] = m
            full_prompt = _EDIT_PROMPT.format(repo=it["repo"], issue=issue, context=ctx)
            costs[a][iid] = {"cl100k_context": oracle[a]["tokens"],
                             "anthropic_prompt": fr.count_tokens(full_prompt)}
        print(f"[{n}/{len(run)}] {iid}: " +
              " | ".join(f"{a.split('_', 1)[1]} ctx={oracle[a]['tokens']}t "
                         f"cov={coverage[a][iid]['covered']}/{coverage[a][iid]['hunks']} "
                         f"ok={metrics[a][iid]['edits_applied']} r{metrics[a][iid]['retry_rounds']}"
                         f"{' TRUNC' if metrics[a][iid]['truncated_any'] else ''}" for a in ARMS),
              flush=True)
        # incremental persistence (multi-hour run; survive interruption)
        for a in ARMS:
            (RESULTS / f"preds_{a}.json").write_text(json.dumps(preds[a], indent=2))
        (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (RESULTS / "costs.json").write_text(json.dumps(costs, indent=2))
        (RESULTS / "coverage.json").write_text(json.dumps(coverage, indent=2))

    (RESULTS / "run_meta.json").write_text(json.dumps({
        "protocol_version": 2,
        "amendments": "see module docstring; frozen before any data collection",
        "resolved_model_id": fr.last_model_id, "requested_model": fr.model,
        "temperature": temperature, "max_tokens": MAX_TOKENS, "max_retry_rounds": MAX_RETRY_ROUNDS,
        "n_instances_run": len(run), "excluded_by_gold": sorted(excluded),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"\nBuilt {len(run)} x {len(ARMS)} predictions -> {RESULTS}")
    print(f"Resolved model snapshot: {fr.last_model_id}")


def _arm_status(arm, instances) -> dict[str, str]:
    """Per-instance eval status from the per-instance report.json files (survive partial
    re-runs, unlike the top-level report): resolved | unresolved | missing (errored/never ran,
    e.g. Docker pull rate limit — an ENVIRONMENT failure, never a model failure).
    An instance with an EMPTY submitted patch gets no report.json but is a MODEL failure:
    classified unresolved, never retried."""
    pred_file = RESULTS / f"preds_{arm}.json"
    empty = set()
    if pred_file.exists():
        empty = {p["instance_id"] for p in json.loads(pred_file.read_text())
                 if not (p.get("model_patch") or "").strip()}
    out = {}
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


def stage_eval(instances, retry_errors: bool = False):
    excluded = _excluded_by_gold()
    run = [i for i in instances if i not in excluded]
    for a in ARMS:
        ids = run
        if retry_errors:
            status = _arm_status(a, run)
            ids = [i for i in run if status[i] == "missing"]
            if not ids:
                print(f"{a}: no missing evals")
                continue
            print(f"{a}: retrying {len(ids)} missing evals")
        pred = str(RESULTS / f"preds_{a}.json")
        cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
               "-d", "princeton-nlp/SWE-bench_Verified", "-p", pred,
               "-id", f"{RUN_PREFIX}_{a}", "-n", "swebench", "--max_workers", "4",
               "--cache_level", "env", "-i", *ids]
        print("RUN:", " ".join(cmd[:8]), "... (-i x", len(ids), "instances)", flush=True)
        subprocess.run(cmd, check=False, timeout=21600, cwd=RESULTS)


def stage_analyze(instances):
    from stats import paired_mcnemar, wilson
    excluded = _excluded_by_gold()
    pre_universe = set(instances) - excluded
    status = {a: _arm_status(a, sorted(pre_universe)) for a in ARMS}
    incomplete = {i for i in pre_universe if any(status[a][i] == "missing" for a in ARMS)}
    universe = pre_universe - incomplete
    resolved = {a: {i for i in universe if status[a][i] == "resolved"} for a in ARMS}
    costs = json.loads((RESULTS / "costs.json").read_text()) if (RESULTS / "costs.json").exists() else {}
    cov = json.loads((RESULTS / "coverage.json").read_text()) if (RESULTS / "coverage.json").exists() else {}

    covered_universe = {i for i in universe
                        if all(cov.get(a, {}).get(i, {}).get("full_coverage") for a in ARMS)}

    lines = [f"# exp-014b v2 analysis (N={len(universe)}; excluded: {len(excluded)} gold-validation, "
             f"{len(incomplete)} eval-incomplete)\n"]
    if incomplete:
        lines.append(f"**⚠ {len(incomplete)} instances lack a completed eval in ≥1 arm (harness/"
                     f"environment errors, e.g. Docker pull rate limit) and are EXCLUDED from all "
                     f"comparisons** — re-run `--stage eval --retry-errors` to recover them:")
        lines.append(", ".join(f"`{i}`" for i in sorted(incomplete)) + "\n")

    def block(title, uni):
        lines.append(f"## {title} (n={len(uni)})\n")
        lines.append("| arm | resolved | rate | Wilson 95% CI | mean cl100k ctx tok | mean coverage |")
        lines.append("|---|---|---|---|---|---|")
        rows = {}
        for a in ARMS:
            k = len(resolved[a] & uni); n = len(uni)
            lo, hi = wilson(k, n)
            ct = {i: c for i, c in costs.get(a, {}).items() if i in uni}
            mt = round(sum(v["cl100k_context"] for v in ct.values()) / len(ct), 1) if ct else None
            cv = [cov.get(a, {}).get(i) for i in uni if cov.get(a, {}).get(i)]
            mc = (round(sum(c["covered"] for c in cv) / max(1, sum(c["hunks"] for c in cv)), 3)
                  if cv else None)
            rows[a] = {"resolved": k, "n": n, "rate": round(k / n, 4) if n else 0,
                       "ci": [round(lo, 3), round(hi, 3)], "mean_cl100k": mt, "hunk_coverage": mc}
            lines.append(f"| {a} | {k}/{n} | {k/n:.3f} | [{lo:.2f}, {hi:.2f}] | {mt} | {mc} |"
                         if n else f"| {a} | 0/0 | - | - | {mt} | {mc} |")
        lines.append("\n### McNemar (exact, two-sided)\n")
        lines.append("| A vs B | A-only | B-only | discordant | p |")
        lines.append("|---|---|---|---|---|")
        mc = {}
        for x, y in [("oracle_tier", "oracle_keep_drop"), ("oracle_tier", "oracle_full"),
                     ("oracle_keep_drop", "oracle_full")]:
            r = paired_mcnemar(resolved[x], resolved[y], uni)
            mc[f"{x}_vs_{y}"] = r
            lines.append(f"| {x} vs {y} | {r['b_only_A']} | {r['c_only_B']} | {r['discordant']} | "
                         f"{r['p_value']:.3f} |")
        lines.append("")
        return rows, mc

    rows_all, mc_all = block("All evaluated instances", universe)
    rows_cov, mc_cov = block("Expressible subset (every arm covers every gold hunk pre-image)",
                             covered_universe)

    out = {"universe": sorted(universe), "excluded_by_gold": sorted(excluded),
           "eval_incomplete": sorted(incomplete),
           "covered_universe": sorted(covered_universe),
           "all": {"arms": rows_all, "mcnemar": mc_all},
           "covered": {"arms": rows_cov, "mcnemar": mc_cov},
           "resolved_ids": {a: sorted(resolved[a] & universe) for a in ARMS}}
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS/'analysis.json'} and ANALYSIS.md")


def stage_plan(instances):
    print("exp-014b v2 PLAN (nothing runs):")
    print(f"  instances: {len(instances)} (frozen in scaled_instances.json; all multi-file Verified)")
    print(f"  arms: {ARMS} (coverage-fair arms_v2; NO truncation)")
    print(f"  agent: claude-sonnet-4-6 @ T=0, max_tokens={MAX_TOKENS}, "
          f"deterministic patches + {MAX_RETRY_ROUNDS} retry rounds")
    print(f"  --stage validate_gold -> {len(instances)} gold Docker evals (env check, no API)")
    print(f"  --stage build         -> ~{len(instances)*len(ARMS)} agent calls + retries (API $)")
    print(f"  --stage eval          -> {len(instances)*len(ARMS)} Docker evals")
    print(f"  --stage analyze       -> Wilson + McNemar, overall AND expressible-subset (free)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["plan", "validate_gold", "build", "eval", "analyze"],
                   default="plan")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None, help="use only the first K instances (smoke test)")
    p.add_argument("--retry-errors", action="store_true",
                   help="eval stage: re-run only instances whose eval errored/never completed")
    a = p.parse_args()
    inst = load_instances()
    if a.limit:
        inst = inst[:a.limit]
    {"plan": lambda: stage_plan(inst),
     "validate_gold": lambda: stage_validate_gold(inst),
     "build": lambda: stage_build(inst, a.temperature),
     "eval": lambda: stage_eval(inst, retry_errors=a.retry_errors),
     "analyze": lambda: stage_analyze(inst)}[a.stage]()
