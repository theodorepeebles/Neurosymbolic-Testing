"""
Graphs and tables comparing the NS pipeline against direct-LLM baselines, built from the
demo JSONL files run.py writes (one file per setup, one line per problem).

Usage (from logic/pipeline/):
    python plot_results.py                     # every file in data/demo_runs/ except EXCLUDE
    python plot_results.py FILE.jsonl ...      # only these (bare names are looked up in data/demo_runs/)
    python plot_results.py --out SOME_DIR      # write somewhere other than data/plots/

Writes:
    1_accuracy_vs_cost.png     accuracy vs. average cost per problem
    2_accuracy_vs_params.png   accuracy vs. total parameter count
    3_cost_per_correct.png     cost per correct answer, each labeled "$x · N× pipeline"
    4_output_tokens.png        average output tokens per problem
    5_accuracy_and_cost.png    accuracy and cost per problem, side-by-side bar panels
    results.csv                one row per run, raw numbers (for analysis)
    results.txt                the same, as an aligned readable table (also printed)
    model_specs.csv / .txt     params + prices from pricing.py for every model used

csv_to_text() turns any CSV into an aligned text table; use it on other CSVs too.

Error bars are 95% intervals: Wilson for accuracy; bootstrap (resampling problems) for the
cost and token averages and for cost per correct. "p vs pipeline" is an exact McNemar test
on the problems both runs share: a small p (< 0.05) means the accuracy gap is unlikely to be chance.
Runs whose models have no price in pricing.py are left off the cost graphs, with a note.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # write files only, no window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, PercentFormatter

from demo_logger import DEMO_DIR, short_name, load_demo_file
from demo_summary import resolve, row_cost
from pricing import PRICING

PLOTS_DIR = os.path.normpath(os.path.join(DEMO_DIR, "..", "plots"))

# Result files skipped by default (passing a file by name still includes it).
EXCLUDE = {
    "baseline__gpt-oss-20b__think-high.jsonl",   # stopped at 48/101: too slow (4 timeouts)
}

# Colors (reference data-viz palette). The pipeline is the one highlighted series; every
# baseline is neutral ink, told apart by its direct label, not by color.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PIPE = "#2a78d6"
BASE = INK2
BASE_BAR = MUTED

# Thinking levels that count as "on" (filled marker); everything else is "off" (hollow).
THINK_ON = {"on", "high", "medium", "max"}
THINK_ORDER = {"off": 0, "low": 1, "medium": 2, "on": 3, "high": 3, "max": 4}


# ── formatting ────────────────────────────────────────────────────────────────

def money(x) -> str:
    """$ with enough decimals to show two significant figures: $0.012, $0.000034."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == 0:
        return "$0"
    if x >= 1:
        return f"${x:,.2f}"
    decimals = max(2, -int(math.floor(math.log10(x))) + 1)
    return f"${x:.{decimals}f}"


def times(r: float) -> str:
    if r >= 100:
        return f"{r:,.0f}×"
    if r >= 10:
        return f"{r:.0f}×"
    return f"{r:.1f}×"


def fmt_params(b) -> str:
    return "?" if b is None else f"{b:g}B"


def fmt_p(p) -> str:
    if p is None:
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── statistics ────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; well-behaved near 0% and 100%."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar test on paired right/wrong arrays.
    Returns (p, only_a_right, only_b_right)."""
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    n = only_a + only_b
    if n == 0:
        return 1.0, only_a, only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail), only_a, only_b


# ── loading ───────────────────────────────────────────────────────────────────

class Run:
    """One setup's results file plus everything the graphs and tables need from it."""

    def __init__(self, path: str):
        self.path = path
        self.file = os.path.basename(path)
        self.rows = load_demo_file(path)
        self.n = len(self.rows)
        self.is_pipeline = "cls_in_tok" in self.rows[0]

        if self.is_pipeline:
            r0 = self.rows[0]
            self.models = [r0["extraction_model"]] + ([r0["classifier_model"]] if r0["classifier_model"] else [])
            sizes = [PRICING.get(m, {}).get("total_params_B") for m in self.models]
            self.total_params = None if None in sizes else sum(sizes)
            self.active_params = None
            self.label = "NS pipeline (" + " + ".join(fmt_params(s) for s in sizes) + ")"
            self.family = None
            self.think = None
        else:
            model = self.rows[0]["model"]
            self.models = [model]
            spec = PRICING.get(model, {})
            self.total_params = spec.get("total_params_B")
            self.active_params = spec.get("active_params_B")
            m = re.search(r"__think-([^.]+)\.jsonl$", self.file)
            self.think = m.group(1) if m else "?"
            name = short_name(model)
            # Size only when the name doesn't already say it (e.g. nemotron-3-super -> 120B).
            size = "" if re.search(r"\d+(\.\d+)?b\b", name.lower()) or self.total_params is None \
                else f" ({fmt_params(self.total_params)})"
            self.label = f"{name}{size} · think {self.think}"
            self.family = model

        self.ids = [r["problem_id"] for r in self.rows]
        self.correct = np.array([1.0 if r["correct"] else 0.0 for r in self.rows])
        self.in_tok = np.array([r["in_tok"] for r in self.rows], dtype=float)
        self.out_tok = np.array([r["out_tok"] for r in self.rows], dtype=float)
        try:
            self.cost = np.array([row_cost(r) for r in self.rows], dtype=float)
            self.price_note = None
        except (KeyError, ValueError) as e:
            self.cost = None
            self.price_note = str(e).strip("'\"")

    @property
    def think_on(self) -> bool:
        return self.is_pipeline or self.think in THINK_ON

    def compute(self, boot: int, rng: np.random.Generator) -> None:
        n, k = self.n, int(self.correct.sum())
        self.k = k
        self.acc = k / n
        self.acc_lo, self.acc_hi = wilson(k, n)

        idx = rng.integers(0, n, size=(boot, n))

        def mean_ci(x):
            means = x[idx].mean(axis=1)
            return float(x.mean()), *np.percentile(means, [2.5, 97.5])

        self.out_mean, self.out_lo, self.out_hi = mean_ci(self.out_tok)
        self.in_mean = float(self.in_tok.mean())

        if self.cost is None:
            self.cpp = self.cpp_lo = self.cpp_hi = None
            self.cpc = self.cpc_lo = self.cpc_hi = None
            return
        self.cpp, self.cpp_lo, self.cpp_hi = mean_ci(self.cost)
        totals = self.cost[idx].sum(axis=1)
        rights = self.correct[idx].sum(axis=1)
        ratios = np.where(rights > 0, totals / np.maximum(rights, 1), np.nan)
        self.cpc = float(self.cost.sum() / k) if k else None
        if self.cpc is None or np.all(np.isnan(ratios)):
            self.cpc_lo = self.cpc_hi = None
        else:
            self.cpc_lo, self.cpc_hi = np.nanpercentile(ratios, [2.5, 97.5])


def load_runs(paths: list[str]) -> list[Run]:
    runs = []
    for p in paths:
        full = resolve(p)
        if not load_demo_file(full):
            print(f"  skipping {os.path.basename(full)}: empty")
            continue
        runs.append(Run(full))
    return runs


# ── chart helpers ─────────────────────────────────────────────────────────────

def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
    })


def footnote(fig, lines: list[str]) -> None:
    text = "\n".join(l for l in lines if l)
    if text:
        fig.text(0.01, 0.01, text, fontsize=8, color=MUTED, ha="left", va="bottom")


def log_limits(lo_vals, hi_vals, right_room: float) -> tuple[float, float]:
    lo, hi = min(lo_vals), max(hi_vals)
    span = max(math.log10(hi / lo), 0.5)
    return lo / 10 ** (span * 0.08 + 0.1), hi * 10 ** (span * right_room + 0.2)


def place_labels(ax, pts, texts, bold) -> None:
    """Label each point to its right, nudging labels apart vertically so they don't collide."""
    fig = ax.figure
    fig.canvas.draw()
    disp = ax.transData.transform(pts)
    ys = disp[:, 1].copy()
    px = fig.dpi / 72
    gap, reach = 13 * px, 230 * px     # min vertical gap; labels closer than `reach` sideways interact
    for _ in range(200):
        moved = False
        order = np.argsort(ys)
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i, j = order[a], order[b]
                if abs(disp[i, 0] - disp[j, 0]) < reach and ys[j] - ys[i] < gap:
                    push = (gap - (ys[j] - ys[i])) / 2 + 0.2
                    ys[i] -= push
                    ys[j] += push
                    moved = True
        if not moved:
            break
    for k, (x, y) in enumerate(pts):
        dy = (ys[k] - disp[k, 1]) / px
        ax.annotate(texts[k], (x, y), xytext=(9, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5,
                    color=INK if bold[k] else INK2, fontweight="semibold" if bold[k] else "normal",
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.9),
                    zorder=4,
                    arrowprops=None if abs(dy) < 4 else
                    dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=0, shrinkB=5))


def dodge(xs, step: float = 0.04) -> list[float]:
    """Spread out points whose x values (on a log axis) are nearly the same, e.g. 117B and 120.6B,
    so their markers and error bars don't sit on top of each other."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out, group = list(xs), [order[0]]

    def spread(g):
        for j, i in enumerate(g):
            out[i] = xs[i] * 10 ** ((j - (len(g) - 1) / 2) * step)

    for i in order[1:]:
        if xs[i] / xs[group[-1]] < 1.12:
            group.append(i)
        else:
            spread(group)
            group = [i]
    spread(group)
    return out


def draw_points(ax, runs, xs) -> None:
    """Accuracy points with light 95% error bars."""
    for r, x in zip(runs, xs):
        color = PIPE if r.is_pipeline else BASE
        ax.errorbar(x, r.acc, yerr=[[r.acc - r.acc_lo], [r.acc_hi - r.acc]],
                    fmt="none", ecolor=color, alpha=0.3, elinewidth=1.0, capsize=0, zorder=2)
        ax.plot(x, r.acc, marker="D" if r.is_pipeline else "o", ms=9 if r.is_pipeline else 7,
                mfc=color if r.think_on else SURFACE, mec=SURFACE if r.think_on else color,
                mew=1.6 if not r.think_on else 1.0, linestyle="none", zorder=3)


def pipeline_guides(ax, ref, x=None) -> list:
    """Dashed line at the pipeline's accuracy (and cost, if x is given) with short labels on the
    right edge saying which side is more / less accurate. Returns legend handles."""
    if ref is None:
        return []
    kw = dict(color=PIPE, lw=1, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.axhline(ref.acc, **kw)
    note = dict(transform=ax.get_yaxis_transform(), ha="right", fontsize=8.5, color=PIPE)
    ax.text(0.995, ref.acc + 0.006, "more accurate than pipeline ▲", va="bottom", **note)
    ax.text(0.995, ref.acc - 0.006, "less accurate than pipeline ▼", va="top", **note)
    if x is not None:
        ax.axvline(x, **kw)
        ax.text(x, 0.995, "  costs more than pipeline →", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8.5, color=PIPE)
    return [Line2D([], [], **{k: v for k, v in kw.items() if k != "zorder"},
                   label="NS pipeline's accuracy" + (" and cost" if x is not None else ""))]


def point_legend(ax, extra: list | None = None) -> None:
    handles = [
        Line2D([], [], marker="D", ms=8, mfc=PIPE, mec=SURFACE, ls="none", label="NS pipeline"),
        Line2D([], [], marker="o", ms=7, mfc=SURFACE, mec=BASE, mew=1.6, ls="none",
               label="Baseline, thinking off / low"),
        Line2D([], [], marker="o", ms=7, mfc=BASE, mec=SURFACE, ls="none",
               label="Baseline, thinking on / high"),
    ] + (extra or [])
    ax.legend(handles=handles, loc="lower right")


def accuracy_axis(ax, runs) -> None:
    lo = min(r.acc_lo for r in runs)
    ax.set_ylim(max(0.0, lo - 0.05), 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Accuracy (95% interval)")


# ── graphs ────────────────────────────────────────────────────────────────────

def graph_accuracy_vs_cost(runs, ref, unpriced, out_dir) -> str | None:
    priced = [r for r in runs if r.cpp]
    if not priced:
        print("  graph 1 skipped: no run has prices in pricing.py yet")
        return None
    fig, ax = plt.subplots(figsize=(10, 6.2))
    xs = [r.cpp for r in priced]
    draw_points(ax, priced, xs)

    ax.set_xscale("log")
    ax.set_xlim(*log_limits(xs, xs, 0.45))
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: money(v)))
    ax.set_xlabel("Average cost per problem (log scale)")
    accuracy_axis(ax, priced)
    ax.set_title("Accuracy vs. cost per problem")
    point_legend(ax, pipeline_guides(ax, ref if ref in priced else None, ref.cpp if ref in priced else None))
    place_labels(ax, [(r.cpp, r.acc) for r in priced], [r.label for r in priced],
                 [r.is_pipeline for r in priced])
    footnote(fig, ["Vertical bars: 95% intervals for accuracy. Cost ranges are in graph 5 and results.txt.",
                   f"Not shown (no price in pricing.py): {', '.join(unpriced)}" if unpriced else ""])
    fig.tight_layout(rect=(0, 0.06 if unpriced else 0.04, 1, 1))
    return save(fig, out_dir, "1_accuracy_vs_cost.png")


def graph_accuracy_vs_params(runs, ref, out_dir) -> str | None:
    sized = [r for r in runs if r.total_params]
    missing = [r.label for r in runs if not r.total_params]
    if not sized:
        print("  graph 2 skipped: no run has a parameter count in pricing.py")
        return None
    fig, ax = plt.subplots(figsize=(10, 6.2))
    xs = dodge([r.total_params for r in sized])
    draw_points(ax, sized, xs)

    ax.set_xscale("log")
    ax.set_xlim(*log_limits(xs, xs, 0.45))
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_params(v)))
    ax.set_xlabel("Total parameters (log scale)")
    accuracy_axis(ax, sized)
    ax.set_title("Accuracy vs. model size")
    point_legend(ax, pipeline_guides(ax, ref if ref in sized else None))
    place_labels(ax, list(zip(xs, [r.acc for r in sized])), [r.label for r in sized],
                 [r.is_pipeline for r in sized])
    footnote(fig, ["Pipeline size = extraction model + classifier. Models of almost the same size are "
                   "nudged apart sideways so they don't overlap. Vertical bars: 95% intervals.",
                   f"Not shown (no size in pricing.py): {', '.join(missing)}" if missing else ""])
    fig.tight_layout(rect=(0, 0.06 if missing else 0.04, 1, 1))
    return save(fig, out_dir, "2_accuracy_vs_params.png")


def graph_cost_per_correct(runs, ref, unpriced, out_dir) -> str | None:
    priced = sorted([r for r in runs if r.cpc], key=lambda r: r.cpc, reverse=True)
    if not priced:
        print("  graph 3 skipped: no run has prices in pricing.py yet")
        return None
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.55 * len(priced)))
    ys = np.arange(len(priced))
    for y, r in zip(ys, priced):
        color = PIPE if r.is_pipeline else BASE
        if r.cpc_lo is not None:
            ax.plot([r.cpc_lo, r.cpc_hi], [y, y], color=color, alpha=0.45, lw=1.4, zorder=2)
        ax.plot(r.cpc, y, marker="D" if r.is_pipeline else "o", ms=8, mfc=color, mec=SURFACE,
                ls="none", zorder=3)
        if ref is not None and ref.cpc and not r is ref:
            note = f"{money(r.cpc)} · {times(r.cpc / ref.cpc)} pipeline"
        else:
            note = f"{money(r.cpc)}" + (" · pipeline" if r.is_pipeline else "")
        ax.annotate(note, ((r.cpc_hi or r.cpc), y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK if r.is_pipeline else INK2,
                    fontweight="semibold" if r.is_pipeline else "normal")

    ax.set_yticks(ys, [r.label for r in priced])
    for tick, r in zip(ax.get_yticklabels(), priced):
        if r.is_pipeline:
            tick.set_color(INK)
            tick.set_fontweight("semibold")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(priced) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(*log_limits([r.cpc_lo or r.cpc for r in priced], [r.cpc_hi or r.cpc for r in priced], 0.6))
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: money(v)))
    ax.set_xlabel("Cost per correct answer (log scale, 95% interval)")
    ax.set_title("Cost per correct answer")
    footnote(fig, [f"Not shown (no price in pricing.py): {', '.join(unpriced)}" if unpriced else ""])
    fig.tight_layout(rect=(0, 0.05 if unpriced else 0, 1, 1))
    return save(fig, out_dir, "3_cost_per_correct.png")


def graph_output_tokens(runs, out_dir) -> str:
    ordered = sorted(runs, key=lambda r: r.out_mean, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.55 * len(ordered)))
    ys = np.arange(len(ordered))
    for y, r in zip(ys, ordered):
        ax.barh(y, r.out_mean, height=0.6, color=PIPE if r.is_pipeline else BASE_BAR, zorder=2)
        ax.plot([r.out_lo, r.out_hi], [y, y], color=INK2, lw=1.2, zorder=3)
        ax.annotate(f"{r.out_mean:,.0f}", (r.out_hi, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK if r.is_pipeline else INK2,
                    fontweight="semibold" if r.is_pipeline else "normal")
    ax.set_yticks(ys, [r.label for r in ordered])
    for tick, r in zip(ax.get_yticklabels(), ordered):
        if r.is_pipeline:
            tick.set_color(INK)
            tick.set_fontweight("semibold")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    ax.set_xlim(0, max(r.out_hi for r in ordered) * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Average output tokens per problem (95% interval)")
    ax.set_title("Output tokens per problem")
    footnote(fig, ["Pipeline = classifier + extraction tokens. Thinking tokens are included. "
                   "Tokens aren't directly comparable across models (different tokenizers)."])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, out_dir, "4_output_tokens.png")


def graph_accuracy_and_cost(runs, out_dir) -> str:
    """Two bar panels side by side, one row per run: accuracy and cost per problem. Separate
    panels because they're in different units; rows line up so each run reads straight across."""
    ordered = sorted(runs, key=lambda r: r.acc)          # most accurate at the top
    ys = np.arange(len(ordered))
    panels = [
        ("Accuracy", lambda r: (r.acc, r.acc_lo, r.acc_hi), lambda v: f"{v:.1%}", True),
        ("Cost per problem", lambda r: (r.cpp, r.cpp_lo, r.cpp_hi), money, False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.6 + 0.55 * len(ordered)), sharey=True)
    for ax, (title, get, fmt, is_pct) in zip(axes, panels):
        vals = [get(r) for r in ordered]
        top = max([hi or v for v, lo, hi in vals if v is not None] or [1])
        for y, r, (v, lo, hi) in zip(ys, ordered, vals):
            emph = dict(color=INK, fontweight="semibold") if r.is_pipeline else dict(color=INK2)
            if v is None:
                ax.annotate("no price", (0, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=MUTED)
                continue
            ax.barh(y, v, height=0.6, color=PIPE if r.is_pipeline else BASE_BAR, zorder=2)
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=INK2, lw=1.2, zorder=3)
            ax.annotate(fmt(v), (max(v, hi or v), y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, **emph)
        ax.set_xlim(0, 1.18 if is_pct else top * 1.3)
        ax.xaxis.set_major_locator(FixedLocator([0, .25, .5, .75, 1]) if is_pct else MaxNLocator(nbins=4))
        if is_pct:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}" if v else "$0"))
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [r.label for r in ordered])
    for tick, r in zip(axes[0].get_yticklabels(), ordered):
        if r.is_pipeline:
            tick.set_color(INK)
            tick.set_fontweight("semibold")
    axes[0].set_ylim(-0.6, len(ordered) - 0.4)
    fig.suptitle("Accuracy and cost per problem", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    footnote(fig, ["Error bars: 95% intervals."])
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, out_dir, "5_accuracy_and_cost.png")


def save(fig, out_dir, name) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ── tables ────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list[dict]) -> None:
    """Raw CSV for analysis: one column per value, plain numbers, empty = no value.
    utf-8-sig so Excel shows the · in run labels correctly."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: "" if v is None else v for k, v in row.items()} for row in rows])


def _num(v: str):
    """CSV cell -> float, or None if empty / not a number."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pretty(v: str) -> str:
    """Default cell format: integers with commas, other numbers to 4 significant figures."""
    x = _num(v)
    if x is None:
        return v or ""
    return f"{x:,.0f}" if x == int(x) and abs(x) < 1e15 else f"{x:.4g}"


def csv_to_text(csv_path: str, txt_path: str | None = None, columns=None,
                title: str | None = None, notes: list[str] | None = None) -> str:
    """Turn a CSV into an aligned plain-text table and write it next to the CSV (same name, .txt).

    columns: optional list of (header, fn) where fn(row_dict) -> str builds each cell from the
             CSV's raw values, so several columns can be combined or formatted ("91.1% (84-95%)").
             Default: every CSV column as-is, numbers tidied.
    Returns the text, so it can be printed as well."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if columns is None:
        columns = [(h, lambda r, h=h: _pretty(r[h])) for h in (rows[0] if rows else [])]
    headers = [h for h, _ in columns]
    cells = [[fn(r) for _, fn in columns] for r in rows]

    widths = [max([len(h)] + [len(c[i]) for c in cells]) for i, h in enumerate(headers)]
    # Text columns (mostly cells starting with a letter) left-aligned; number columns right-aligned.
    def is_text(i):
        vals = [c[i] for c in cells if c[i]]
        return bool(vals) and sum(v[0].isalpha() for v in vals) > len(vals) / 2
    left = [is_text(i) for i in range(len(headers))]
    line = lambda vals: " | ".join(v.ljust(w) if left[i] else v.rjust(w)
                                   for i, (v, w) in enumerate(zip(vals, widths)))
    out = []
    if title:
        out += [title, "=" * len(title), ""]
    out += [line(headers), "-+-".join("-" * w for w in widths)]
    out += [line(c) for c in cells]
    if notes:
        out += [""] + notes
    text = "\n".join(out) + "\n"

    txt_path = txt_path or os.path.splitext(csv_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def vs_pipeline(r, ref):
    """(p, only pipeline right, only this run right) on the problems both share; None for the pipeline."""
    if ref is None or r is ref:
        return None, None, None
    shared = sorted(set(ref.ids) & set(r.ids))
    ref_c, run_c = dict(zip(ref.ids, ref.correct)), dict(zip(r.ids, r.correct))
    return mcnemar_exact(np.array([ref_c[i] for i in shared]), np.array([run_c[i] for i in shared]))


def _fmt(fn, *keys):
    """Text-table cell builder: read keys from the CSV row as numbers, format with fn; '' if missing."""
    def cell(row):
        vals = [_num(row[k]) for k in keys]
        return "" if any(v is None for v in vals) else fn(*vals)
    return cell


# How results.csv is shown in results.txt: ranges folded into their value's column.
RESULTS_TEXT_COLUMNS = [
    ("Run", lambda r: r["run"]),
    ("Correct", _fmt(lambda k, n: f"{k:.0f} / {n:.0f}", "correct", "n")),
    ("Accuracy (95% range)", _fmt(lambda a, lo, hi: f"{a:.1%} ({lo:.0%}-{hi:.0%})", "accuracy", "acc_lo", "acc_hi")),
    ("Cost / problem", _fmt(money, "cost_per_problem")),
    ("Cost / correct (95% range)", _fmt(lambda c, lo, hi: f"{money(c)} ({money(lo)}-{money(hi)})",
                                        "cost_per_correct", "cpc_lo", "cpc_hi")),
    ("vs. pipeline", lambda r: "pipeline" if r["is_pipeline"] == "True"
                     else _fmt(times, "x_pipeline_cost_per_correct")(r)),
    ("Avg in tok", _fmt(lambda x: f"{x:,.0f}", "avg_in_tok")),
    ("Avg out tok", _fmt(lambda x: f"{x:,.0f}", "avg_out_tok")),
    ("Params (active)", lambda r: _fmt(fmt_params, "total_params_B")(r)
                        + _fmt(lambda a: f" ({fmt_params(a)})", "active_params_B")(r)),
    ("p vs. pipeline", _fmt(fmt_p, "p_vs_pipeline")),
    ("Only pipeline right", lambda r: r["only_pipeline_right"]),
    ("Only this right", lambda r: r["only_this_right"]),
]

RESULTS_TEXT_NOTES = [
    "Ranges are 95% intervals (accuracy: Wilson; cost: bootstrap over problems).",
    "vs. pipeline = this run's cost per correct answer / the pipeline's.",
    "p vs. pipeline = exact McNemar test on shared problems; below 0.05 means the accuracy gap is",
    "  unlikely to be chance. 'Only ... right' counts the problems exactly one of the two got right.",
]


def price_per_m(x: float) -> str:
    """$ per 1M tokens: at least 2 decimals, more only when needed ($0.60, $0.015, $0.014)."""
    s = f"{x:,.4f}".rstrip("0")
    return "$" + (s + "0" * (2 - len(s.split(".")[1])) if "." in s else s + ".00")

SPECS_TEXT_COLUMNS = [
    ("Model", lambda r: r["model"]),
    ("Total params", lambda r: _fmt(fmt_params, "total_params_B")(r) or "unknown"),
    ("Active params", lambda r: _fmt(fmt_params, "active_params_B")(r) or "unknown"),
    ("$ / 1M input", lambda r: _fmt(price_per_m, "in_per_M")(r) or "not set"),
    ("$ / 1M output", lambda r: _fmt(price_per_m, "out_per_M")(r) or "not set"),
    ("Price source", lambda r: r["price_source"]),
]


def results_table(runs, ref, out_dir) -> None:
    rows = []
    for r in sorted(runs, key=lambda r: -r.acc):
        p, only_ref, only_this = vs_pipeline(r, ref)
        ratio = r.cpc / ref.cpc if ref is not None and r is not ref and r.cpc and ref.cpc else None
        rows.append({
            "run": r.label, "file": r.file, "is_pipeline": r.is_pipeline, "n": r.n, "correct": r.k,
            "accuracy": round(r.acc, 4), "acc_lo": round(r.acc_lo, 4), "acc_hi": round(r.acc_hi, 4),
            "cost_per_problem": r.cpp, "cpp_lo": r.cpp_lo, "cpp_hi": r.cpp_hi,
            "cost_per_correct": r.cpc, "cpc_lo": r.cpc_lo, "cpc_hi": r.cpc_hi,
            "x_pipeline_cost_per_correct": ratio,
            "avg_in_tok": round(r.in_mean, 1), "avg_out_tok": round(r.out_mean, 1),
            "total_params_B": r.total_params, "active_params_B": r.active_params,
            "p_vs_pipeline": p, "only_pipeline_right": only_ref, "only_this_right": only_this,
        })
    path = os.path.join(out_dir, "results.csv")
    write_csv(path, rows)
    notes = RESULTS_TEXT_NOTES + (["Cost is empty for runs whose models have no price in pricing.py."]
                                  if any(r.cost is None for r in runs) else [])
    print("\n" + csv_to_text(path, columns=RESULTS_TEXT_COLUMNS, title="Results", notes=notes))


def specs_table(runs, out_dir) -> None:
    models = []
    for r in runs:
        for m in r.models:
            if m not in models:
                models.append(m)
    rows = []
    for m in models:
        s = PRICING.get(m, {})
        rows.append({"model": m, "total_params_B": s.get("total_params_B"),
                     "active_params_B": s.get("active_params_B"), "in_per_M": s.get("in_per_M"),
                     "out_per_M": s.get("out_per_M"), "price_source": s.get("price_source", "")})
    path = os.path.join(out_dir, "model_specs.csv")
    write_csv(path, rows)
    print("\n" + csv_to_text(path, columns=SPECS_TEXT_COLUMNS, title="Model specs (from pricing.py)"))


def headline(runs, ref) -> None:
    if ref is None:
        return
    beaten = [r for r in runs if not r.is_pipeline and r.acc < ref.acc and r.total_params]
    if not beaten:
        print("\nThe pipeline doesn't beat any baseline with a known size yet.")
        return
    top = max(beaten, key=lambda r: (r.total_params, -r.acc))
    p, _, _ = vs_pipeline(top, ref)
    line = (f"\nLargest model the pipeline beats: {top.label} ({fmt_params(top.total_params)}) - "
            f"{ref.acc:.1%} vs {top.acc:.1%} (p = {fmt_p(p)})")
    if ref.cpc and top.cpc:
        line += f", at {ref.cpc / top.cpc:.1%} of its cost per correct answer"
    print(line)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Graphs + tables for demo benchmark runs.")
    ap.add_argument("files", nargs="*", help="result files (default: every .jsonl in data/demo_runs/)")
    ap.add_argument("--out", default=PLOTS_DIR, help=f"output folder (default: {PLOTS_DIR})")
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap resamples for error bars")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = args.files or sorted(p for p in glob.glob(os.path.join(DEMO_DIR, "*.jsonl"))
                                 if os.path.basename(p) not in EXCLUDE)
    runs = load_runs(paths)
    if not runs:
        sys.exit("No results to plot.")

    rng = np.random.default_rng(args.seed)
    for r in runs:
        r.compute(args.boot, rng)

    # A run with fewer problems than the others is flagged in its label.
    full = max(r.n for r in runs)
    for r in runs:
        if r.n < full:
            r.label += f" [{r.n}/{full}]"

    pipelines = [r for r in runs if r.is_pipeline]
    ref = pipelines[0] if pipelines else None
    if len(pipelines) > 1:
        print(f"  several pipeline files; comparing against {ref.file}")
    if ref is None:
        print("  no pipeline file given: no 'x pipeline' or p-values")

    unpriced = [r.label for r in runs if r.cost is None]
    for r in runs:
        if r.price_note:
            print(f"  no cost for {r.label}: {r.price_note}")

    os.makedirs(args.out, exist_ok=True)
    style()
    written = [graph_accuracy_vs_cost(runs, ref, unpriced, args.out),
               graph_accuracy_vs_params(runs, ref, args.out),
               graph_cost_per_correct(runs, ref, unpriced, args.out),
               graph_output_tokens(runs, args.out),
               graph_accuracy_and_cost(runs, args.out)]

    results_table(runs, ref, args.out)
    specs_table(runs, args.out)
    headline(runs, ref)

    print(f"\nWrote to {os.path.normpath(args.out)}:")
    for p in [w for w in written if w] + ["results.csv", "results.txt", "model_specs.csv", "model_specs.txt"]:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
