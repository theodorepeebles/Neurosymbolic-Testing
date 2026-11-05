"""
Test domain classifiers on their own (no extraction, no Z3).

Runs each model in CLASSIFIERS over sft_test.jsonl and compares its predicted
domains against the row's ground-truth active_domains.

  Exact        predicted set == gold set
  Covers gold  no gold domain missing (extras allowed) - a missing domain breaks
               extraction, so this matters most for full-pipeline accuracy
"""

import json
import os
import time

from pipeline import classify_domains, new_usage

CLASSIFIERS = [
    # "qwen3:8b",
     "qwen3:4b",
    # "qwen3:1.7b",
    # "qwen3:0.6b",
    # cloud fallbacks if the local models are all weak:
    # "gpt-oss:120b-cloud",
    # "gemma4:31b-cloud",
]

NUM_TEST_EXAMPLES = None  # None = all rows

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sft_test.jsonl")


def load_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("active_domains"), str):
                row["active_domains"] = json.loads(row["active_domains"])
            rows.append(row)
    return rows[:limit] if limit is not None else rows


rows = load_rows(TEST_SET_PATH, NUM_TEST_EXAMPLES)
n = len(rows)
summary = []

for model in CLASSIFIERS:
    print(f"\n=== {model} ===")
    exact = covers = errors = 0
    total_ms = 0
    usage = new_usage()

    for i, row in enumerate(rows, 1):
        gold = set(row["active_domains"])
        print(f"  [{i}/{n}] ...", end="\r", flush=True)  # progress; overwritten by the next line
        t0 = time.time()
        try:
            pred = classify_domains(row["problem_text"], model=model, usage=usage)
            err = None
        except Exception as e:
            pred, err = None, str(e).splitlines()[0]
        total_ms += int((time.time() - t0) * 1000)

        if pred is None:
            errors += 1
            print(f"  [{i}/{n}] ERROR  gold={sorted(gold)}  {err}")
            continue

        pred = set(pred)
        exact += pred == gold
        covers += gold <= pred
        if pred != gold:
            print(f"  [{i}/{n}] WRONG  gold={sorted(gold)}  pred={sorted(pred)}")

    summary.append({
        "model": model, "exact": exact, "covers": covers, "errors": errors,
        "avg_ms": total_ms / n if n else 0,
        "avg_out": usage["out_tok"] / n if n else 0,
    })
    print(f"  exact {exact}/{n}  covers {covers}/{n}  errors {errors}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print(f"CLASSIFIER SUMMARY  ({n} problems)")
print(f"{'=' * 72}")
w = max(len(s["model"]) for s in summary) + 2 if summary else 10
print(f"  {'Model':<{w}}  {'Exact':<14}  {'Covers gold':<14}  {'Errors':<7}  {'Avg ms':<8}  Avg out tok")
for s in summary:
    ex = f"{s['exact']}/{n} ({s['exact'] / n * 100:.0f}%)" if n else "-"
    cv = f"{s['covers']}/{n} ({s['covers'] / n * 100:.0f}%)" if n else "-"
    print(f"  {s['model']:<{w}}  {ex:<14}  {cv:<14}  {s['errors']:<7}  "
          f"{s['avg_ms']:<8.0f}  {s['avg_out']:.0f}")
