# exp-015 — de-circularized probe re-grade (find/act boundary)

Re-measures the paper's headline probe claim — *"summaries cannot substitute for source when
acting"* (PAPER_WORKSHOP.md Tables 1–2 / Fig. 1) — with the failure-audit defects removed
(`src-ng/paper_audit/SOTA_FAILURE_AUDIT.md` D1/D2; REVISION_PLAN §C):

| original defect | fix here |
|---|---|
| probes minted from the scored source with "answerable ONLY from this source" → negatives partly entailed | probes minted from each class's **TEST FILE** (independent behavioral ground truth); answerer never sees tests; UNKNOWN allowed |
| same model generates, answers, and grades (self-judged) | answers graded by an **independent-family judge** (OpenAI `gpt-4o-mini`) + Claude as second judge → **Cohen's κ** reported |
| authors' own repos (`dhcm_v2`/`dhcm_ng`), in-distribution | **held-out**: seaborn / pylint / pytest at pinned SWE-bench base commits |
| one 3B summarizer at capped budget → "summaries can't act" confounded with summarizer quality | **H5 control**: identical summary prompt + 180-token budget from BOTH `qwen2.5-coder:3b` (original) and Claude Sonnet 4.6 (frontier upper bound) |
| T=1.0 single draws | T=0 everywhere |

**Pre-registered readout.** Per-arm accuracy + Wilson CIs (per judge); exact McNemar of each arm vs
`source`; κ between judges. Interpretation rule, stated before running: if `summary_frontier` ≈
`summary_qwen3b` ≪ `source`, the find/act boundary hardens (not a summarizer artifact); if
`summary_frontier` closes most of the gap, the paper's claim must be weakened to "cheap-model
summaries cannot act" and the H5 economic thesis is refuted; if `source` itself scores low under the
independent judge, the original 15/15 was judge leniency and the whole probe instrument is demoted.

## Run (staged; mint/represent/answer/grade spend API — ~$2–5 total, no Docker)
```bash
uv run python experiments/probe_regrade/select_nodes.py     # freeze 15 held-out classes (done)
uv run pytest experiments/probe_regrade/test_regrade.py -q  # 5 tests, free
uv run python experiments/probe_regrade/regrade_run.py --stage plan
uv run python experiments/probe_regrade/regrade_run.py --stage mint       # 15 Claude calls
uv run python experiments/probe_regrade/regrade_run.py --stage represent  # local qwen + 15 Claude
uv run python experiments/probe_regrade/regrade_run.py --stage answer     # 180 Claude calls, T=0
uv run python experiments/probe_regrade/regrade_run.py --stage grade      # 360 judge calls
uv run python experiments/probe_regrade/regrade_run.py --stage analyze    # free -> results/ANALYSIS.md
```

## Canonical results: strict-T=0 rerun (2026-06-12)

An external review caught that the originally shipped run generated the 3B summaries at the Ollama
client's default T=0.2 (all other stages T=0). `results/` now holds the corrected strict-T=0 rerun
on the SAME frozen probes; the deviating original ships in `results_shipped_t02/` for the delta:
source 30→27, summaries 3→4 each, signature 6→6 — ≤3 cells per arm, identical conclusions (and a
probe-level replication of the run-to-run noise floor measured in the resolve experiments).
