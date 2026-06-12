# What Context Does a Coding Agent Actually Need to *Act*?

This repository accompanies the paper of the same name ([`paper/main.pdf`](paper/main.pdf)). The
paper asks what a coding agent minimally needs to see when it must *edit* code, holding
localization fixed so that only the representation of the context varies, and scoring against real
issue resolution on SWE-bench Verified. Its central results: natural-language summaries carry
almost none of the signal that source code does (3/45 vs. 30/45 behavioral probes — for frontier
and 3B summarizers alike); rendering a file's remainder as UML skeletons and signatures resolves
no more issues than deleting it (N=70, pre-registered, the null published as the result); and
compressed context matches whole files at a third of the tokens — a resolved issue for 19K context
tokens instead of 94K. Along the way it measures a noise floor the field should know about:
temperature-0 API inference flips ~9% of per-instance outcomes between byte-identical runs.

Every number in the paper is a deterministic artifact of the code and cached raw results in this
repository; protocols and instance samples were frozen and committed before data collection.

## Layout

```
paper/                       NeurIPS-2026-format paper (main.tex, refs.bib, checklist, figures, PDF)
src/dhcm_ng/                 the instrument's library: AST extraction, representation rungs,
                             arm construction, detail(v) allocator, model clients
experiments/
  resolve/                   pre-registered resolve@cost study (N=70, 3 arms) + per-instance
                             harness reports, expressibility sweep, frozen sample
  sequence_slice/            pre-registered interaction-artifact follow-up + T=0 noise measurement
  probe_regrade/             the find/act boundary (held-out repos, test-minted probes,
                             independent judge, summarizer control)
results/legacy_probes/       superseded own-code probe results (paper App. D, no claims)
scripts/make_figures.py      regenerates every paper figure from the raw result files
reproduce.sh                 one command: tests + re-aggregated analyses + figures, no API needed
```

## Reproduce (no API keys, no Docker)

```bash
./reproduce.sh
```

This re-runs the unit tests, re-derives every ANALYSIS table from the cached per-instance harness
reports (resolve, sequence-slice, probe re-grade, map pilot, embeddings), and regenerates all paper
figures from the raw result JSONs.

## Re-run from scratch (API + Docker)

Each experiment directory's README documents its staged runner (`plan / validate_gold / build /
eval / analyze`). Full reruns need `ANTHROPIC_API_KEY` (agent) and `OPENAI_API_KEY` (independent
judge) in the environment, Docker with ~50 GB free for SWE-bench images, and roughly $25 of API
inference for the registered resolve run (compute details: paper App. B). Note the measured
run-to-run noise at temperature 0 (~9% of per-instance outcomes flip between identical runs):
exact per-instance reproduction is not expected; aggregate results are.

## Paper

```bash
cd paper && latexmk -pdf main.tex
```
