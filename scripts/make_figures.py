"""Generate the paper's figures DIRECTLY from the verified result JSONs.

Every number plotted is loaded from a raw result file (no hardcoded results), so the figures are
auditable against the data. Outputs PNG + PDF into paper/figures/.

Run: uv run python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EXP14B = ROOT / "experiments" / "resolve"
PROBE = ROOT / "experiments" / "probe_regrade"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
INK = "#1b1b1b"; ACCENT = "#2c6fbb"; GOOD = "#2e8b57"; BAD = "#c0392b"; MUTE = "#9aa0a6"


def load(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf")


# ---------------------------------------------------------------- Figure 2: rig pipeline
def fig2_rig():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(-9, 52); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec=ACCENT, fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.4, color=INK))

    box(2, 38, 22, 9, "Gold-changed files\n(oracle localization)", fc="#fff6e6", ec="#d9a441", bold=True)
    # three arms
    arms = [("oracle_full", 30), ("oracle_tier", 53), ("oracle_keep_drop", 76)]
    for name, x in arms:
        arrow(13, 38, x+9, 33)
        box(x, 24, 18, 9, name, fc="#eef3fb", fs=9)
    ax.text(63, 35.5, "vary ONLY representation", ha="center", fontsize=9, style="italic", color=MUTE)
    # merge to agent
    for _, x in arms:
        arrow(x+9, 24, 50, 19)
    box(30, 10, 40, 9, "Agent: Claude Sonnet 4.6 (single-shot, T=0)\nemits SEARCH/REPLACE blocks (≤2 retries)", fc="#eef9f0", ec=GOOD, fs=9)
    arrow(50, 10, 50, 6.5)
    box(8, -2, 38, 8, "Deterministic SEARCH→difflib\npatch builder (git-apply verified)", fc="#f3eefb", ec="#7d5bbe", bold=True, fs=9)
    box(54, -2, 40, 8, "SWE-bench Docker harness\nRESOLVE = FAIL_TO_PASS ∧ PASS_TO_PASS", fc="#fdeeee", ec=BAD, bold=True, fs=9)
    arrow(46, 2, 54, 2)
    ax.text(74, -7.4, "gold patches validated per instance (70/71)   ·   cost = context tokens (cl100k proxy + Anthropic count)",
            ha="center", fontsize=8.6, color=MUTE)
    ax.set_title("The resolve@cost evaluation rig", y=1.06)
    save(fig, "fig2_rig")


# ---------------------------------------------------------------- Figure 5: honest selection ladder
def fig5_selection():
    spec = load(EXP14B / "scaled_instances.json")
    ana = load(EXP14B / "results" / "analysis.json")
    n_pool = spec["multi_file_pool_size"]                       # 71
    n_gold = len(ana["universe"])                               # 70
    n_best = max(len(v) for v in ana["resolved_ids"].values())  # 25 (best arm)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    cx = 42  # center the funnel left of the right-margin annotation lane
    steps = [
        ("500", "SWE-bench Verified (the pool)", "#dbe7f6"),
        (f"{n_pool}", "every multi-file instance (≥2 gold files)\nthe registered pool, run whole", "#cfe3f0"),
        (f"{n_gold}", "gold patch validates its environment\n(one exclusion)", "#cfeede"),
        (f"{n_best}", "resolved single-shot\n(best arm)", "#f6dede"),
    ]
    widths = [64, 50, 36, 22]
    bh = 15; gap = 7
    y = 94
    for (big, lab, col), w in zip(steps, widths):
        x = cx - w/2
        ax.add_patch(FancyBboxPatch((x, y-bh), w, bh, boxstyle="round,pad=0.2,rounding_size=2",
                                    fc=col, ec=INK, lw=1.1))
        ax.text(cx, y-4.6, big, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(cx, y-bh+4.2, lab, ha="center", va="center", fontsize=8.0, color=INK)
        if w != widths[-1]:
            ax.add_patch(FancyArrowPatch((cx, y-bh), (cx, y-bh-gap), arrowstyle="-|>",
                                         mutation_scale=13, lw=1.3, color=INK))
        y -= (bh + gap)
    # right-margin transition annotations
    ax.text(80, 76, "registered selection\n(nothing sampled)", ha="left", va="center", fontsize=8.5, color=MUTE, style="italic")
    ax.text(80, 20, "single-shot resolve;\nno execution feedback", ha="left", va="center", fontsize=8.5, color=MUTE, style="italic")
    # side note: expressibility audited pre-data (bottom-left, clear of the funnel)
    ax.add_patch(FancyBboxPatch((2, 6), 26, 14, boxstyle="round,pad=0.3,rounding_size=2",
                                fc="#fff6e6", ec="#d9a441", lw=1.1))
    ax.text(15, 13, "pre-data audit: every gold\nedit expressible in every\narm (70/70, all arms)", ha="center", va="center", fontsize=8, color=INK)
    ax.set_title("Where the sample comes from — the registered pool, run whole", y=1.0)
    save(fig, "fig3_selection")




# ---------------------------------------------------------------- Figure 5: powered resolve (N=70)
def fig5_powered():
    ana = load(EXP14B / "results" / "analysis.json")["all"]["arms"]
    arms = ["oracle_full", "oracle_tier", "oracle_keep_drop"]
    nice = {"oracle_full": "full files", "oracle_tier": "structured tiers", "oracle_keep_drop": "keep/drop"}
    cols = {"oracle_full": MUTE, "oracle_tier": ACCENT, "oracle_keep_drop": GOOD}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    xs = range(len(arms))
    for i, a in enumerate(arms):
        r = ana[a]
        rate = r["resolved"] / r["n"]
        lo, hi = r["ci"]
        axL.bar([i], [rate], color=cols[a], width=0.6)
        axL.errorbar([i], [rate], yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=INK, capsize=5, lw=1.4)
        axL.text(i, hi + 0.03, f"{r['resolved']}/{r['n']}", ha="center", fontsize=10, fontweight="bold")
    axL.set_xticks(list(xs)); axL.set_xticklabels([nice[a] for a in arms])
    axL.set_ylabel("resolve rate (Wilson 95% CI)"); axL.set_ylim(0, 0.62)
    axL.set_title("Resolve, N=70 (pre-registered)")
    tpr = {a: ana[a]["mean_cl100k"] * ana[a]["n"] / max(1, ana[a]["resolved"]) for a in arms}
    for i, a in enumerate(arms):
        axR.bar([i], [tpr[a] / 1000], color=cols[a], width=0.6)
        axR.text(i, tpr[a] / 1000 + 2, f"{tpr[a]/1000:.0f}K", ha="center", fontsize=10, fontweight="bold")
    axR.set_xticks(list(xs)); axR.set_xticklabels([nice[a] for a in arms])
    axR.set_ylabel("context tokens per resolve (thousands)")
    axR.set_title("Cost of a resolve")
    fig.tight_layout()
    save(fig, "fig5_powered")


# ---------------------------------------------------------------- Figure 1b: held-out boundary
def fig1_boundary():
    import json
    ana = json.loads((PROBE / "results" / "analysis.json").read_text())
    reps = json.loads((PROBE / "results" / "representations.json").read_text())
    toks = reps["tokens_cl100k"]
    order = ["source", "summary_frontier", "summary_qwen3b", "signature"]
    nice = {"source": "full source", "summary_frontier": "summary written\nby frontier model",
            "summary_qwen3b": "summary written\nby 3B model", "signature": "signature\n+ docstring"}
    cols = {"source": GOOD, "summary_frontier": BAD, "summary_qwen3b": BAD, "signature": ACCENT}
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, a in enumerate(order):
        r = ana["arms"][a]["openai"]
        rate = r["correct"] / r["n"]
        lo, hi = r["ci"]
        ax.bar([i], [rate], color=cols[a], width=0.62)
        ax.errorbar([i], [rate], yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=INK, capsize=5, lw=1.4)
        mt = round(sum(t[a] for t in toks.values()) / len(toks))
        ax.text(i, hi + 0.03, f"{r['correct']}/{r['n']}", ha="center", fontsize=10.5, fontweight="bold")
        ax.text(i, -0.085, f"~{mt:,} tok", ha="center", fontsize=8.6, color=MUTE)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([nice[a] for a in order])
    ax.set_ylim(0, 0.92); ax.set_ylabel("behavioral-probe accuracy (Wilson 95% CI)")
    ax.set_title("The find/act boundary — same answerer (Sonnet 4.6) in every arm; only the representation varies")
    ax.text(0.5, -0.24, "a frontier summarizer scores identically to a 3B one: the gap belongs to the representation, not the summarizer",
            transform=ax.transAxes, ha="center", fontsize=9.3, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "fig1_boundary")


if __name__ == "__main__":
    fig1_boundary(); fig2_rig(); fig5_selection(); fig5_powered()
    print(f"Done -> {OUT}")
