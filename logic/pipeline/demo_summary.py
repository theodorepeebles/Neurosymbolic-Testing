"""
Comparison table for demo benchmark runs (the JSONL files written by run.py with LOG_TO_DEMO).

Usage (from logic/pipeline/):
    python demo_summary.py ns__SFT-0.6b-v4__cls-qwen3-8b.jsonl baseline__gpt-oss-120b__think-on.jsonl

A bare filename is looked up in data/demo_runs/ if it isn't found as given.
Prints one row per file, using only the common fields. Cost columns show "n/a" when a
model in the file has no price filled in pricing.py.
"""

import json
import os
import sys

from demo_logger import DEMO_DIR
from pricing import cost


def resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    in_demo_dir = os.path.join(DEMO_DIR, path)
    if os.path.exists(in_demo_dir):
        return in_demo_dir
    raise FileNotFoundError(f"{path} (also looked in {os.path.normpath(DEMO_DIR)})")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_cost(r: dict) -> float:
    """One problem's cost. A pipeline line prices classifier and extraction tokens at each
    model's own rate; a baseline line prices in_tok/out_tok at its one model's rate."""
    if "cls_in_tok" not in r:
        return cost(r["model"], r["in_tok"], r["out_tok"])
    total = cost(r["extraction_model"], r["ext_in_tok"], r["ext_out_tok"])
    if r["classifier_model"]:   # None when ground-truth domains were used
        total += cost(r["classifier_model"], r["cls_in_tok"], r["cls_out_tok"])
    return total


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    total_in = sum(r["in_tok"] for r in rows)
    total_out = sum(r["out_tok"] for r in rows)

    try:
        total_cost = sum(row_cost(r) for r in rows)
    except (KeyError, ValueError):   # a model has no price in pricing.py
        total_cost = None

    return {
        "n": n,
        "correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "avg_in": total_in / n if n else 0.0,
        "avg_out": total_out / n if n else 0.0,
        "total_tok": total_in + total_out,
        "cost_per_problem": total_cost / n if total_cost is not None and n else None,
        "cost_per_correct": total_cost / n_correct if total_cost is not None and n_correct else None,
        "avg_time_s": sum(r["time_ms"] for r in rows) / n / 1000 if n else 0.0,
    }


def money(x) -> str:
    return "n/a" if x is None else f"${x:.6f}"


def print_table(named: list[tuple[str, dict]]) -> None:
    headers = ["Run", "Accuracy", "Avg in", "Avg out", "Total tok",
               "$/problem", "$/correct", "Avg time (s)"]
    table = [[name,
              f"{s['correct']}/{s['n']} ({s['accuracy']:.1%})",
              f"{s['avg_in']:.0f}",
              f"{s['avg_out']:.0f}",
              f"{s['total_tok']}",
              money(s["cost_per_problem"]),
              money(s["cost_per_correct"]),
              f"{s['avg_time_s']:.1f}"]
             for name, s in named]
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print(line(row))


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit("usage: python demo_summary.py FILE.jsonl [FILE.jsonl ...]")
    named = []
    for p in paths:
        full = resolve(p)
        name = os.path.basename(full).removesuffix(".jsonl")
        named.append((name, summarize(load(full))))
    print_table(named)


if __name__ == "__main__":
    main(sys.argv[1:])
