"""
Local testing harness for a trained extraction model.

Reads the held-out test set (sft_test.jsonl), runs each problem through the NS
pipeline with the trained model, compares the model's extraction against the gold
extraction stored in the row, and logs a full per-row comparison into sft_test.db.

When every row has been processed, sft_test.db is fully populated for analysis.

Set NUM_TEST_EXAMPLES below to cap how many rows are run (None = all).
"""

import time
import json
import os
import uuid
import traceback
from functools import partial

from pipeline import (
    run_ns_pipeline, baseline_llm_solve, constraint_type_counts,
    extract_logic_problem, z3_solve,
)
from validators import build_hybrid_schema
from logger import init_db, log_attempt
import eval_metrics
import attribution

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# How many test examples to run. None = all rows in sft_test.jsonl.
NUM_TEST_EXAMPLES = None

RUN_BASELINE = False

# Model used for the direct-LLM baseline (run once per problem, so it isn't tied to a
# single MODEL_SETS entry). Single source of truth for the baseline model.
BASELINE_MODEL = "qwen3:8b"

LOG_TO_DB = True

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sft_test.jsonl")

MODEL_SETS = [
    # {
    #     "name": "qwen3:0.6b",
    #     "classifier_llm": "qwen3:8b",
    #     "extraction_llm": "qwen3:0.6b",
    #     "formatting_llm": "qwen3:8b",
    # },
      {
        "name": "SFT_Extraction_Qwen3_0.6b-v4",
        "classifier_llm": "qwen3:8b",
        "extraction_llm": "SFT_Extraction_Qwen3_0.6b-v4",
        "formatting_llm": "qwen3:8b",
    },
]

# Models extracted with finetuned=True (minimal FT_EXTRACTION_SYSTEM prompt, raw JSON).
# All others use the default rich domain-specific prompt with examples.
FINETUNED_MODELS = {"SFT_Extraction_Qwen3_0.6b"}


def load_test_set(path: str, limit=None) -> list[dict]:
    """Load gold rows from sft_test.jsonl. Each row: problem_text, active_domains,
    extracted_json (gold)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # active_domains / extracted_json are stored as JSON strings
            if isinstance(row.get("active_domains"), str):
                row["active_domains"] = json.loads(row["active_domains"])
            if isinstance(row.get("extracted_json"), str):
                row["gold"] = json.loads(row["extracted_json"])
            else:
                row["gold"] = row.get("extracted_json")
            rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def derive_ground_truth(gold: dict, active_domains: list[str]):
    """Z3-solve the gold extraction to recover the correct answer label.

    Returns the single verified choice label, or None if the gold puzzle is UNSAT
    or doesn't resolve to exactly one verified label. Source-agnostic: works for
    any dataset row regardless of which generator produced it.
    """
    try:
        gold_lp = build_hybrid_schema(active_domains)(**gold)
        res = z3_solve(gold_lp)
        if res.get("status") != "sat" or not res.get("question_results"):
            return None
        verified = [l for l, v in res["question_results"][0].items() if v]
        return verified[0] if len(verified) == 1 else None
    except Exception:
        return None


if LOG_TO_DB:
    init_db()
test_rows = load_test_set(TEST_SET_PATH, NUM_TEST_EXAMPLES)
total_problems = len(test_rows)
results = []

for i, row in enumerate(test_rows, 1):
    problem        = row["problem_text"]
    active_domains = row["active_domains"]
    gold           = row["gold"]
    gt             = derive_ground_truth(gold, active_domains)  # Z3 on the gold extraction
    run_id         = row.get("run_id") or uuid.uuid4().hex

    print(f"\n[{i}/{total_problems}]")
    print(f"  Problem: {problem}")

    record = {
        "problem":        problem,
        "ground_truth":   gt,
        "baseline_correct": None,
        "ns_results":     {},
    }

    # Baseline — run once per problem, not once per model set
    if RUN_BASELINE:
        t0 = time.time()
        baseline_answer, _baseline_parse_error, _baseline_raw = baseline_llm_solve(problem, model=BASELINE_MODEL)
        record["baseline_correct"] = (
            baseline_answer is not None and gt is not None
            and baseline_answer.upper() == gt.upper()
        )

    # NS pipeline — once per model set
    for ms in MODEL_SETS:
        print(f"  --- NS extract [{ms['name']}] ---")

        extract_fn = partial(
            extract_logic_problem,
            model=ms["extraction_llm"],
            finetuned=ms["extraction_llm"] in FINETUNED_MODELS,
        )

        error_tb = None
        t0 = time.time()
        try:
            ns = run_ns_pipeline(
                problem,
                extract_fn,
                active_domains=active_domains,
                classifier_model=ms["classifier_llm"],
                formatter_model=ms["formatting_llm"],
                verbalize=True,
            )
        except Exception:
            error_tb = traceback.format_exc()
            ns = {"extracted": None, "z3_status": None, "question_results": None,
                  "llm_calls": 0, "unmatched_errors": [], "formatted_output": None,
                  "extraction_raw": None}
        gen_ms = int((time.time() - t0) * 1000)

        # Did the model land on exactly the right answer?
        ns_correct = False
        if ns.get("question_results"):
            first_q = ns["question_results"][0]
            correct_labels = [label for label, verified in first_q.items() if verified]
            if len(correct_labels) == 1 and gt is not None:
                ns_correct = correct_labels[0].upper() == gt.upper()

        extracted_obj  = ns.get("extracted")
        extracted_dict = extracted_obj.model_dump() if extracted_obj else None
        # Store with nested evidence_text stripped (kept only on top-level constraints), matching
        # how gold is serialized — model_dump_json would otherwise emit "evidence_text": null on
        # every nested wrapper child. extracted_dict (above) stays unstripped for metrics/attribution.
        extracted_json = (json.dumps(eval_metrics.strip_problem_nested_evidence(extracted_obj.model_dump()))
                          if extracted_obj else None)
        counts         = constraint_type_counts(extracted_obj) if extracted_obj else None
        z3_status      = ns.get("z3_status")

        metrics = eval_metrics.build_comparison_metrics(
            gold, extracted_dict, active_domains, problem
        )

        # Attribute each extracted constraint to a span of the problem text.
        # Option A (LLM evidence_text) when present, else entity-filtered BM25 fallback.
        attribution_methods = attribution_spans = None
        if extracted_dict is not None:
            attribution_methods, attribution_spans = attribution.build_attribution(
                extracted_dict, problem
            )

        db_row = {
            "run_id":                run_id,
            "attempt_number":        1,
            "extraction_model_name": ms["name"],
            "environment":           "controlled",
            "problem_text":          problem,
            "active_domains":        active_domains,
            "prompt_token_count":    None,
            **metrics,
            # Gold extraction, stored alongside the model's output so the DB is self-contained
            # (decoupled from sft_test.jsonl). Keep the dataset's raw string verbatim for diff
            # fidelity; fall back to serialising the parsed gold dict.
            "expected_json":         (row["extracted_json"]
                                      if isinstance(row.get("extracted_json"), str)
                                      else (json.dumps(gold) if gold is not None else None)),
            "extracted_json":        extracted_json,
            "schema_valid":          extracted_obj is not None,
            "constraint_type_counts": counts,
            "z3_result":             z3_status.upper() if z3_status else ("ERROR" if error_tb else None),
            "answer_correct":        ns_correct,
            "ground_truth_answer":   gt,
            "error_traceback":       error_tb,
            "completion_token_count": None,
            "generation_time_ms":    gen_ms,
            "attribution_methods":   attribution_methods,
            "attribution_spans":     attribution_spans,
        }
        if LOG_TO_DB:
            log_attempt(db_row)

        record["ns_results"][ms["name"]] = {
            "ns_correct": ns_correct,
            "z3_status":  z3_status,
            "exact_global": metrics["exact_global_constraint_match"],
            "gen_ms":     gen_ms,
        }
        print(f"  Z3: {z3_status} \n correct: {ns_correct}  "
              f"exact_global_match: {metrics['exact_global_constraint_match']} \n  ({gen_ms} ms)")
        if ns.get("formatted_output"):
            print("  --- Explanation ---")
            print(ns["formatted_output"])

    results.append(record)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = len(results)
print(f"\n{'='*60}")
print(f"TEST SUMMARY  ({'logged to sft_test.db' if LOG_TO_DB else 'DB logging disabled'})")
print(f"{'='*60}")
print(f"  Total examples : {total}")
print()

col_w = max((len(ms["name"]) for ms in MODEL_SETS), default=10) + 12
header = f"  {'Model':<{col_w}}  {'Correct':<10}  Accuracy"
print(header)
print(f"  {'-'*len(header.strip())}")

for ms in MODEL_SETS:
    count = sum(1 for r in results if r["ns_results"].get(ms["name"], {}).get("ns_correct") is True)
    pct = (count / total * 100) if total else 0
    print(f"  {ms['name']:<{col_w}}  {f'{count}/{total}':<10}  {pct:.1f}%")

if RUN_BASELINE:
    correct_baseline = sum(1 for r in results if r["baseline_correct"] is True)
    pct = (correct_baseline / total * 100) if total else 0
    print(f"  {'Baseline LLM':<{col_w}}  {f'{correct_baseline}/{total}':<10}  {pct:.1f}%")
