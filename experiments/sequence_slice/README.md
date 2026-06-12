# exp-014c — sequence-slice arm vs keep/drop (pre-registered)

Tests the part of the original DHCM design that exp-014b's H2 null did **not** cover: whether an
**interaction-level artifact** — a UML-style sequence slice of who-calls-whom around the edit site —
carries resolve signal that flat text and placeholders do not. (014b tested only the thin end of the
structured-artifact family: class skeletons and signatures.)

## Registered design (frozen and committed before any run)
- **Arms (2):** `oracle_seq` = `oracle_keep_drop`'s **byte-identical content** + an appended Mermaid
  `sequenceDiagram` derived deterministically (pure AST, no LLM) from the gold functions' calls:
  outgoing calls resolved against entities defined in the gold files, plus incoming calls to gold
  functions, in source order, capped at 40 edges with the overflow stated (`seq_arm.py`). The pair
  isolates exactly one variable: the interaction artifact. Measured slice cost: ~240–650 tokens.
- **Sample:** the same 70 gold-valid multi-file Verified instances as exp-014b (frozen there).
- **Both arms run fresh** (same day, same resolved model): the keep/drop re-run keeps the pair clean
  AND yields a free protocol-stability check against exp-014b's keep/drop (drift reported, not hidden).
- **Protocol:** identical to exp-014b v2 — Claude Sonnet 4.6, T=0, max_tokens 8000, ≤2 retry rounds,
  no truncation, coverage-fair base content, real Docker resolve, error≠unresolved discipline.
- **Registered hypothesis H3:** `oracle_seq` resolves more than `oracle_keep_drop` (exact two-sided
  McNemar). A tie is reported as the result. Power note: at ~36% base resolve and N=70, only large
  effects reach significance; the direction and discordant counts are reported regardless.

## Run
```bash
uv run pytest experiments/sequence_slice/test_014c.py -q   # 3 tests, free
uv run python experiments/sequence_slice/run_014c.py --stage plan
uv run python experiments/sequence_slice/run_014c.py --stage build   # ⚠ API: 140 calls (~$10)
uv run python experiments/sequence_slice/run_014c.py --stage eval    # ⚠ Docker: 140 evals
uv run python experiments/sequence_slice/run_014c.py --stage analyze # free -> results/ANALYSIS.md
```

## Interpretation rule (pre-stated)
- seq > keep/drop (significant): first positive evidence for interaction-level structured context;
  motivates carrying sequence artifacts into the agent-integration experiment (option B).
- tie/null: the structured-artifact family is now null at BOTH the static-digest end (014b) and the
  interaction end (this) in the single-shot oracle regime; option B proceeds with dynamics and
  budget allocation as the remaining untested mechanisms.
- seq < keep/drop: structured additions actively distract at this scale; reported as such.
