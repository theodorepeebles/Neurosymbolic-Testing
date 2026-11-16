"""
Graph for the RAG pilot: base qwen3:0.6b extraction with and without retrieved examples.

Kept separate from plot_results.py (the main pipeline-vs-baselines benchmark) while RAG is
an experiment. Reuses its style, helpers and statistics so the graphs match.

Usage (from logic/pipeline/):
    python plot_rag_results.py             # writes data/plots/6_rag_vs_no_rag.png

Reads:
    data/demo_runs/ns__qwen3-0.6b__cls-gt.jsonl          no RAG (static prompt examples)
    data/demo_runs_rag/ns__qwen3-0.6b-RAG__cls-gt.jsonl  RAG (3 retrieved training puzzles)

Error bars are 95% intervals: Wilson for accuracy, bootstrap over problems for the averages.
The p-value is an exact McNemar test on the shared problems.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, PercentFormatter

from demo_logger import DEMO_DIR, DEMO_RAG_DIR, load_demo_file
from plot_results import (PLOTS_DIR, INK, INK2, BASE_BAR, PIPE, wilson, mcnemar_exact,
                          fmt_p, style, footnote, save)

NO_RAG_PATH = os.path.join(DEMO_DIR, "ns__qwen3-0.6b__cls-gt.jsonl")
RAG_PATH = os.path.join(DEMO_RAG_DIR, "ns__qwen3-0.6b-RAG__cls-gt.jsonl")


def outcome(row: dict) -> str:
    if row["correct"]:
        return "correct"
    if row["z3_status"] is None:
        return "bad JSON / crash"
    if row["z3_status"] == "unsat":
        return "contradiction"
    return "solvable, wrong or ambiguous"


def summarize(rows: list[dict], rng: np.random.Generator, boot: int = 5000) -> dict:
    n = len(rows)
    correct = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    k = int(correct.sum())
    idx = rng.integers(0, n, size=(boot, n))

    def mean_ci(x):
        return float(x.mean()), *np.percentile(x[idx].mean(axis=1), [2.5, 97.5])

    acc_lo, acc_hi = wilson(k, n)
    return {
        "n": n, "k": k, "acc": (k / n, acc_lo, acc_hi),
        "in_tok": mean_ci(np.array([r["in_tok"] for r in rows], dtype=float)),
        "time_s": mean_ci(np.array([r["time_ms"] / 1000 for r in rows], dtype=float)),
        "outcomes": {o: sum(outcome(r) == o for r in rows) for o in
                     ["correct", "contradiction", "bad JSON / crash", "solvable, wrong or ambiguous"]},
    }


def graph_rag_vs_no_rag(runs: list[tuple[str, dict, bool]], p_note: str, out_dir: str) -> str:
    """Three bar panels side by side (accuracy, input tokens, time), one row per setup."""
    ys = np.arange(len(runs))
    panels = [
        ("Accuracy", "acc", lambda v: f"{v:.1%}", True),
        ("Input tokens per puzzle", "in_tok", lambda v: f"{v:,.0f}", False),
        ("Time per puzzle", "time_s", lambda v: f"{v:.1f} s", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, (title, key, fmt, is_pct) in zip(axes, panels):
        top = max(s[key][2] for _, s, _ in runs)
        for y, (label, s, is_rag) in zip(ys, runs):
            v, lo, hi = s[key]
            ax.barh(y, v, height=0.6, color=PIPE if is_rag else BASE_BAR, zorder=2)
            ax.plot([lo, hi], [y, y], color=INK2, lw=1.2, zorder=3)
            ax.annotate(fmt(v), (hi, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK if is_rag else INK2,
                        fontweight="semibold" if is_rag else "normal")
        if is_pct:
            ax.set_xlim(0, 0.5)
            ax.xaxis.set_major_locator(FixedLocator([0, .1, .2, .3, .4, .5]))
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        else:
            ax.set_xlim(0, top * 1.35)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [label for label, _, _ in runs])
    for tick, (_, _, is_rag) in zip(axes[0].get_yticklabels(), runs):
        if is_rag:
            tick.set_color(INK)
            tick.set_fontweight("semibold")
    axes[0].set_ylim(-0.6, len(runs) - 0.4)
    fig.suptitle("RAG vs. no RAG: base qwen3:0.6b extractor", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    footnote(fig, ["Base (not fine-tuned) qwen3:0.6b on the same 101 test puzzles, correct puzzle types "
                   "given (classifier skipped). Error bars: 95% intervals.",
                   p_note])
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    return save(fig, out_dir, "6_rag_vs_no_rag.png")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for p in (NO_RAG_PATH, RAG_PATH):
        if not os.path.exists(p):
            sys.exit(f"Missing results file: {os.path.normpath(p)}")

    no_rag = {r["problem_id"]: r for r in load_demo_file(NO_RAG_PATH)}
    rag = {r["problem_id"]: r for r in load_demo_file(RAG_PATH)}
    shared = sorted(set(no_rag) & set(rag))
    rng = np.random.default_rng(0)
    s_no = summarize([no_rag[i] for i in shared], rng)
    s_rag = summarize([rag[i] for i in shared], rng)

    p, only_no, only_rag = mcnemar_exact(np.array([no_rag[i]["correct"] for i in shared], dtype=float),
                                         np.array([rag[i]["correct"] for i in shared], dtype=float))
    p_note = (f"Accuracy gap: exact McNemar p = {fmt_p(p)} (only RAG right on {only_rag} puzzles, "
              f"only no-RAG right on {only_no}).")

    style()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = graph_rag_vs_no_rag([("RAG (3 retrieved examples)", s_rag, True),
                                ("No RAG (static examples)", s_no, False)], p_note, PLOTS_DIR)

    print(f"RAG pilot, base qwen3:0.6b, {len(shared)} shared puzzles")
    for label, s in (("No RAG", s_no), ("RAG", s_rag)):
        print(f"  {label:7} accuracy {s['k']}/{s['n']} ({s['acc'][0]:.1%})  "
              f"input tokens {s['in_tok'][0]:,.0f}  time {s['time_s'][0]:.1f} s  {s['outcomes']}")
    print(f"  {p_note}")
    print(f"Wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
