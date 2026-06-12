# exp-014b — pre-registered resolve@cost, protocol v2 (all 71 multi-file, coverage-fair)

Tests whether **how the non-locus context is rendered** (structured tiers vs binary keep/drop)
changes **resolution**, at fixed (oracle) localization, on the real SWE-bench Verified Docker
harness. Protocol v2 replaces v1 after the failure audit
(`src-ng/paper_audit/SOTA_FAILURE_AUDIT.md`) found the v1 runner pre-tainted; **all amendments are
instrument fixes frozen and committed BEFORE any data collection** (no agent call, no eval has run).

## v2 amendments (vs v1) — audit findings they fix
| # | Amendment | Audit finding |
|---|---|---|
| 1 | **No truncation** (was `context[:50000]`, which blinded ONLY `oracle_full` — 23/30 of the v1 sample exceeded it; the pilot's "compressed ≥ full" was this artifact) + no issue truncation; real sizes logged | B1 |
| 2 | **Coverage-fair arms** (`arms_v2.py`): module-level code, class-attribute units, non-`.py` gold files, and decorators rendered in the compressed arms; new-file edits expressible everywhere | B2 |
| 3 | **Expressibility instrumentation**: per arm×instance, every gold edit-run's pre-image verified present in the context (`coverage_sweep.py` pre-audit + per-run logging); analysis conditioned on it | B2/B3 |
| 4 | **Edit retry loop** (≤2 feedback rounds on no-match/ambiguous/no-file) via `edit_patch_v2.py`; failures no longer silently dropped | B5/E4 |
| 5 | **Patch builder fixes**: anchored `a/ b/` strip (v1's `lstrip("ab/")` mangled `a/astropy/...`), uniqueness required, new-file creation | E4 |
| 6 | Reasoning permitted before edit blocks (prose-tolerant parser); `max_tokens=8000` | B5 |
| 7 | Costs counted on the **exact** text sent (cl100k context + Anthropic `count_tokens` full prompt) | B4 |
| 8 | **Gold validation for ALL instances** before build; gold-failing environments excluded (recorded) | E4 |
| 9 | **Sample = all 71** multi-file instances (v1's 30-of-71 was underpowered for ~no saving) | E1 |
| 10 | Pre-registration **committed to git** before any run | E2 |

**Registered hypothesis H2 (unchanged):** `oracle_tier` resolves more than `oracle_keep_drop`.
A tie at N=71 is reported as the result. Disclosed limitation (unchanged, by design): compressed
arms encode gold edit *locations* (oracle ceiling, not a deployable policy); `oracle_full` does not.

## Run order (staged; each stage refuses to outrun its prerequisite)
```bash
uv run python experiments/resolve/select_instances.py     # freeze sample (done, committed)
uv run pytest experiments/resolve/test_scaled.py -q        # 12 tests, no API/Docker
uv run python experiments/resolve/coverage_sweep.py        # free pre-audit: expressibility per arm
uv run python experiments/resolve/scaled_run.py --stage plan           # dry run
uv run python experiments/resolve/scaled_run.py --stage validate_gold  # ⚠ Docker: 71 gold evals
uv run python experiments/resolve/scaled_run.py --stage build          # ⚠ API: ~213 calls + retries
uv run python experiments/resolve/scaled_run.py --stage eval           # ⚠ Docker: 213 evals
uv run python experiments/resolve/scaled_run.py --stage analyze        # free -> results/ANALYSIS.md
```
`--limit K` restricts to the first K instances (smoke test). `build` refuses to run before
`validate_gold`'s report exists.

## Decision rule
Wilson CIs per arm + exact McNemar (tier vs keep/drop; each vs full), reported BOTH overall and on
the all-arms-expressible subset. If McNemar shows no tier>keep/drop difference at N=71, the powered
conclusion is the null ("structured tiers do not beat keep/drop at oracle localization, single-shot").
If too few instances are solvable single-shot to discriminate, the pre-registered escalation is a
2-step agent loop on the solvable subset — as a NEW experiment, not a reinterpretation of this one.
