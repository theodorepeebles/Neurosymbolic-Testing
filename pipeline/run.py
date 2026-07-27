import time
from pipeline import run_ns_pipeline, baseline_llm_solve, MAX_ATTEMPTS
from test_suite import test_suite

# ── TEST LOOP ─────────────────────────────────────────────────────────────────

results = []

total_problems = len(test_suite)

all_unmatched = []

for i, entry in enumerate(test_suite, 1):
    problem = entry["problem"]
    gt = entry["answer"]

    print(f"\n[{i}/{total_problems}]")
    print(f"  Problem: {problem}")

    record = {
        "problem": problem,
        "ground_truth": gt,
        "extracted": None,
        "z3_answer": None,
        "ns_correct": None,
        "ns_llm_calls": 0,
        "ns_time": None,
        "baseline_answer": None,
        "baseline_answer_raw": None,
        "baseline_parse_error": None,
        "baseline_correct": None,
        "baseline_time": None,
        "baseline_llm_calls": 1,
        "baseline_error_raw": None,
        "baseline_error_pct": None,
        "formatted_output": None,
        "unmatched_errors": []
    }

    # baseline
    print(f"  Waiting for Baseline LLM solving...")
    t0 = time.time()
    baseline_answer, baseline_parse_error, baseline_raw = baseline_llm_solve(problem)
    t_baseline_end = time.time()
    print(f"  Got baseline LLM response ({t_baseline_end - t0:.2f}s)")

    record["baseline_time"] = round(t_baseline_end - t0, 3)
    record["baseline_answer"] = baseline_answer
    record["baseline_answer_raw"] = baseline_raw
    record["baseline_parse_error"] = baseline_parse_error
    record["baseline_correct"] = baseline_answer is not None and abs(baseline_answer - gt) < 0.01

    # Baseline error stats
    if baseline_answer is not None and gt != 0:
        record["baseline_error_raw"] = baseline_answer - gt
        record["baseline_error_pct"] = round((baseline_answer - gt) / gt * 100, 2)

    # NS pipeline
    t0 = time.time()
    ns = run_ns_pipeline(problem)
    
    record["ns_time"] = round(time.time() - t0, 3)
    record["extracted"] = ns["extracted"]
    record["z3_answer"] = ns["z3_answer"]
    record["formatted_output"] = ns["formatted_output"]
    record["ns_llm_calls"] = ns["llm_calls"]
    record["unmatched_errors"] = ns["unmatched_errors"]
    record["ns_correct"] = ns["z3_answer"] is not None and abs(ns["z3_answer"] - gt) < 0.01
    
    all_unmatched.extend(ns["unmatched_errors"])
    results.append(record)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
for i, r in enumerate(results, 1):
    print(f"{'='*60}")
    print(f"RESULT {i}")
    print(f"{'='*60}")
    print(f"  Problem        : {r['problem']}")
    print(f"  Ground truth   : {r['ground_truth']}")

    # If extraction was successful
    if r["extracted"]:
        # Convert the Pydantic object back to a standard dictionary for easy printing
        dump = r["extracted"].model_dump()

        print(f"  Extracted      :")
        print(f"    variables    : {dump['variables']}")
        print(f"    constraints  : {dump['constraints']}")
        print(f"    output_var   : {dump['output_variable']}")
    else:
        print(f"  Extracted      : FAILED")

    print(f"  Z3 answer      : {r['z3_answer']}")
    print(f"  Formatted      : {r['formatted_output']}")
    print(f"  Baseline LLM raw        : {r['baseline_answer_raw']}")
    print(f"  Baseline LLM stripped   : {r['baseline_answer']}")
    # printing none below means there was no valid answer to check
    print(f"  Was NeuroSymbolic correct?     : {r['ns_correct']}")
    print(f"  Was Baseline LLM correct?      : {r['baseline_correct']}")
    print()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total  = len(results)
solved = sum(1 for r in results if r["z3_answer"] is not None)
failed = sum(1 for r in results if r["extracted"] is None)


correct_baseline = sum(1 for r in results if r["baseline_correct"] is True)
correct_ns = sum(1 for r in results if r["ns_correct"] is True)

print(f"{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Total problems     : {total}")
print(f"  Z3 solved          : {solved}")
print(f"  Extract failed     : {failed}")
print(f"  Baseline accuracy  : {correct_baseline}/{total}")
print(f"  Neurosymbolic acc  : {correct_ns}/{total}")
print(f"  Delta              : +{correct_ns - correct_baseline} problems")

# Calculate and print Average Baseline Errors (excluding parse failures and correct answers)
baseline_errors = [r for r in results if r.get("baseline_error_raw") is not None and not r["baseline_correct"]]
if baseline_errors:
    avg_pct_err = sum(abs(r["baseline_error_pct"]) for r in baseline_errors) / len(baseline_errors)
    print(f"  Avg Baseline Error : {avg_pct_err:.2f}%")


# baseline parse failures listed so you can see what went wrong
baseline_failures = [r for r in results if r["baseline_parse_error"] is not None]
if baseline_failures:
    print(f"\n  Baseline parse failures:")
    for r in baseline_failures:
        print(f"    [{r['problem'][:40]}...] {r['baseline_parse_error']}")

if all_unmatched:
    print(f"{'='*60}")
    print(f"UNMATCHED ERRORS — write hints for these")
    print(f"{'='*60}")
    for e in all_unmatched:
        print(f"  [{e['error_type']}] {e['error_msg']}")