#!/usr/bin/env bash
# Reproduce every paper number and figure from cached raw results. No API keys, no Docker.
set -e
cd "$(dirname "$0")"
echo "== environment =="
uv sync --all-extras 2>&1 | tail -1
echo "== unit tests =="
uv run pytest experiments -q
echo "== resolve@cost (N=70): re-derive analysis from per-instance harness reports =="
uv run python experiments/resolve/scaled_run.py --stage analyze | tail -30
echo "== sequence slice + T=0 noise: re-derive =="
uv run python experiments/sequence_slice/run_014c.py --stage analyze | tail -15
echo "== find/act boundary: re-derive =="
uv run python experiments/probe_regrade/regrade_run.py --stage analyze | tail -25
echo "== figures =="
uv run python scripts/make_figures.py
echo "== done: analyses match the committed ANALYSIS.md files; figures in paper/figures/ =="
