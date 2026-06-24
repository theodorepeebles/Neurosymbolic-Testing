import time
import uuid
import json
import os
from functools import partial
from pipeline import run_ns_pipeline, baseline_llm_solve, MAX_ATTEMPTS, constraint_type_counts, extract_logic_problem, extract_finetuned
from test_suite import test_suite
from logger import init_db, log_attempt

results = []
total_problems = len(test_suite)
all_unmatched = []

# ── TEMPORARY TEST: domain lookup from test_suite_data.jsonl ─────────────────
# Feeds ground-truth active_domains into run_ns_pipeline so the classifier is
# bypassed. Lets us test extraction + Z3 in isolation. Remove this block and
# the active_domains= kwarg in the run_ns_pipeline call below when done.
_ts_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_suite_data.jsonl")
_ts_domain_lookup = {}
with open(_ts_data_path, encoding="utf-8") as _f:
    for _line in _f:
        _entry = json.loads(_line)
        _ts_domain_lookup[_entry["problem_text"]] = json.loads(_entry["active_domains"])
# ── END TEMPORARY TEST ────────────────────────────────────────────────────────

RUN_BASELINE = False

MODEL_SETS = [
    {
        "name": "qwen3:8b",
        "classifier_llm": "qwen3:8b",
        "extraction_llm": "qwen3:8b",
        "formatting_llm": "qwen3:8b",
    },
    {
        "name": "SFT_Extraction_Qwen3_0.6b",
        "classifier_llm": "qwen3:8b",
        "extraction_llm": "SFT_Extraction_Qwen3_0.6b",
        "formatting_llm": "qwen3:8b",
    },
]

# Models that use extract_finetuned (minimal FT_EXTRACTION_SYSTEM prompt).
# All others use extract_logic_problem (rich domain-specific prompt with examples).
FINETUNED_MODELS = {"SFT_Extraction_Qwen3_0.6b"}

init_db()

for i, entry in enumerate(test_suite, 1):
    problem = entry["problem"]
    gt = entry["answer"]

    print(f"\n[{i}/{total_problems}]")
    print(f"  Problem: {problem}")

    record = {
        "problem":              problem,
        "ground_truth":         gt,
        "baseline_answer":      None,
        "baseline_answer_raw":  None,
        "baseline_parse_error": None,
        "baseline_correct":     None,
        "baseline_time":        None,
        "ns_results":           {},
    }

    # Baseline — run once per problem, not once per model set
    if RUN_BASELINE:
        print(f"  Waiting for Baseline LLM solving...")
        t0 = time.time()
        baseline_answer, baseline_parse_error, baseline_raw = baseline_llm_solve(problem)
        record["baseline_time"]        = round(time.time() - t0, 3)
        record["baseline_answer"]      = baseline_answer
        record["baseline_answer_raw"]  = baseline_raw
        record["baseline_parse_error"] = baseline_parse_error
        record["baseline_correct"]     = baseline_answer is not None and baseline_answer.upper() == gt.upper()
        print(f"  Got baseline LLM response ({record['baseline_time']:.2f}s)")
    else:
        print(f"  Baseline LLM solving skipped (RUN_BASELINE=False)")

    # NS pipeline — once per model set
    for ms in MODEL_SETS:
        print(f"\n  --- NS extract [{ms['name']}] ---")

        if ms["extraction_llm"] in FINETUNED_MODELS:
            extract_fn = partial(extract_finetuned, model=ms["extraction_llm"])
        else:
            extract_fn = partial(extract_logic_problem, model=ms["extraction_llm"])

        t0 = time.time()
        ns = run_ns_pipeline(
            problem,
            extract_fn,
            active_domains=_ts_domain_lookup.get(problem),  # TEMPORARY TEST — remove with lookup block above
            classifier_model=ms["classifier_llm"],
            formatter_model=ms["formatting_llm"],
        )
        ns_time = round(time.time() - t0, 3)

        print(f"  Extracted   : {ns['extracted']}")
        print(f"  Z3 status   : {ns['z3_status']}")
        print(f"  Questions   : {ns['question_results']}")

        ns_correct = False
        if ns["question_results"]:
            first_q = ns["question_results"][0]
            correct_labels = [label for label, verified in first_q.items() if verified]
            if len(correct_labels) == 1:
                ns_correct = correct_labels[0].upper() == gt.upper()

        # SQLite logging
        extracted_json = ns["extracted"].model_dump_json() if ns["extracted"] else None
        counts = constraint_type_counts(ns["extracted"]) if ns["extracted"] else None
        z3_status = ns.get("z3_status")
        log_attempt(
            run_id=uuid.uuid4().hex,
            attempt_number=1,
            problem_text=problem,
            active_domains=ns.get("active_domains") or [],
            extracted_json=extracted_json,
            schema_valid=ns["extracted"] is not None,
            z3_result=z3_status.upper() if z3_status else "ERROR",
            answer_correct=ns_correct,
            ground_truth_answer=gt,
            model_name=ms["name"],
            constraint_type_counts=counts,
        )

        record["ns_results"][ms["name"]] = {
            "extracted":        ns["extracted"],
            "extraction_raw":   ns.get("extraction_raw"),
            "active_domains":   ns.get("active_domains"),
            "question_results": ns["question_results"],
            "ns_correct":       ns_correct,
            "ns_llm_calls":     ns["llm_calls"],
            "ns_time":          ns_time,
            "formatted_output": ns["formatted_output"],
            "unmatched_errors": ns["unmatched_errors"],
        }
        all_unmatched.extend(ns["unmatched_errors"])

    results.append(record)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
for i, r in enumerate(results, 1):
    print(f"{'='*60}")
    print(f"RESULT {i}")
    print(f"{'='*60}")
    print(f"  Problem        : {r['problem']}")
    print(f"  Ground truth   : {r['ground_truth']}")
    if RUN_BASELINE:
        print(f"  Baseline LLM raw        : {r['baseline_answer_raw']}")
        print(f"  Baseline LLM stripped   : {r['baseline_answer']}")
        print(f"  Was Baseline LLM correct? : {r['baseline_correct']}")

    for ms_name, ns in r["ns_results"].items():
        print(f"\n  [NS — {ms_name}]")
        print(f"    Active domains : {ns['active_domains']}")
        if ns["extracted"]:
            dump = ns["extracted"].model_dump()
            print(f"    Extracted      :")
            print(f"      entities     : {dump['entities']}")
            print(f"      constraints  : {dump['constraints']}")
            print(f"      questions    : {dump['questions']}")
        else:
            print(f"    Extracted      : FAILED")
        print(f"    Question results  : {ns['question_results']}")
        print(f"    Formatted         : {ns['formatted_output']}")
        print(f"    Correct?          : {ns['ns_correct']}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total            = len(results)
correct_baseline = sum(1 for r in results if r["baseline_correct"] is True)

print(f"\n{'='*60}")
print(f"COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"  Total problems : {total}")
print()

col_w = max(len(ms["name"]) for ms in MODEL_SETS) + 12
header = f"  {'Method':<{col_w}}  {'Correct':<10}  Accuracy"
print(header)
print(f"  {'-'*len(header.strip())}")

if RUN_BASELINE:
    baseline_acc = f"{correct_baseline}/{total}"
    pct = (correct_baseline / total * 100) if total else 0
    print(f"  {'Baseline LLM (qwen3:8b)':<{col_w}}  {baseline_acc:<10}  {pct:.1f}%")

ns_correct_counts = {}
for ms in MODEL_SETS:
    count = sum(1 for r in results if r["ns_results"].get(ms["name"], {}).get("ns_correct") is True)
    ns_correct_counts[ms["name"]] = count
    label = f"NS — {ms['name']}"
    acc_str = f"{count}/{total}"
    pct = (count / total * 100) if total else 0
    print(f"  {label:<{col_w}}  {acc_str:<10}  {pct:.1f}%")

if RUN_BASELINE:
    print()
    print(f"  NS vs Baseline delta:")
    for ms in MODEL_SETS:
        delta = ns_correct_counts[ms["name"]] - correct_baseline
        sign = "+" if delta >= 0 else ""
        print(f"    {ms['name']:<20}: {sign}{delta} problems")

    baseline_failures = [r for r in results if r["baseline_parse_error"] is not None]
    if baseline_failures:
        print(f"\n  Baseline parse failures:")
        for r in baseline_failures:
            print(f"    [{r['problem'][:40]}...] {r['baseline_parse_error']}")

if all_unmatched:
    print(f"\n{'='*60}")
    print(f"UNMATCHED ERRORS — write hints for these")
    print(f"{'='*60}")
    for e in all_unmatched:
        print(f"  [{e['error_type']}] {e['error_msg']}")

# ── PERFORMANCE COMPARISON FILE ───────────────────────────────────────────────
# Build a lookup from problem text → expected answer using the test suite
ts_lookup = {entry["problem"]: entry["answer"] for entry in test_suite}

lines = []
lines.append("EXTRACTION PERFORMANCE COMPARISON")
lines.append("=" * 80)
lines.append(f"Run date : {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"Models   : {', '.join(ms['name'] for ms in MODEL_SETS)}")
lines.append("")

for i, r in enumerate(results, 1):
    problem = r["problem"]
    expected_answer = ts_lookup.get(problem, "N/A")

    lines.append("=" * 80)
    lines.append(f"PROBLEM {i}")
    lines.append("=" * 80)
    lines.append(f"Problem text:")
    lines.append(f"  {problem}")
    lines.append(f"Expected answer: {expected_answer}")
    lines.append("")

    for ms_name, ns in r["ns_results"].items():
        lines.append(f"  [ Model: {ms_name} ]")
        lines.append(f"  Active domains: {ns.get('active_domains')}")
        lines.append("")

        lines.append("  RAW EXTRACTION RESPONSE:")
        lines.append("  " + "-" * 40)
        raw = ns.get("extraction_raw") or "(none)"
        for line in raw.splitlines():
            lines.append(f"  {line}")
        lines.append("")

        lines.append("  EXTRACTED JSON (parsed):")
        lines.append("  " + "-" * 40)
        if ns["extracted"]:
            extracted_json = json.loads(ns["extracted"].model_dump_json())
            for line in json.dumps(extracted_json, indent=2).splitlines():
                lines.append(f"  {line}")
        else:
            lines.append("  (extraction failed)")
        lines.append("")

        lines.append(f"  NS answer correct? {ns['ns_correct']}  |  question results: {ns['question_results']}")
        lines.append("")

output_path = os.path.join(os.path.dirname(__file__), "performance_comparison.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nPerformance comparison written to: {output_path}")
