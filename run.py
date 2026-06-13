import time
import uuid
from pipeline import run_ns_pipeline, baseline_llm_solve, MAX_ATTEMPTS, constraint_type_counts, extract_logic_problem, extract_finetuned
from test_suite import test_suite
from logger import init_db, log_attempt

results = []
total_problems = len(test_suite)
all_unmatched = []

extract_fn = extract_finetuned
MODEL_NAME = "qwen3-ns" #fine tuned model

init_db()

# Run the logic_test_suite problems through the pipeline
for i, entry in enumerate(test_suite, 1):
    problem = entry["problem"]
    gt = entry["answer"] 

    print(f"\n[{i}/{total_problems}]")
    print(f"  Problem: {problem}")

    record = {
        "problem":                problem,
        "ground_truth":           gt,
        "extracted":              None,
        "active_domains":         None,         
        "question_results":       None,          
        "ns_correct":             None,
        "ns_llm_calls":           0,
        "ns_time":                None,
        "baseline_answer":        None,
        "baseline_answer_raw":    None,
        "baseline_parse_error":   None,
        "baseline_correct":       None,
        "baseline_time":          None,
        "baseline_llm_calls":     1,
        "formatted_output":       None,
        "unmatched_errors":       []
    }

    # baseline
    print(f"  Waiting for Baseline LLM solving...")
    t0 = time.time()
    baseline_answer, baseline_parse_error, baseline_raw = baseline_llm_solve(problem)
    t_baseline_end = time.time()
    print(f"  Got baseline LLM response ({t_baseline_end - t0:.2f}s)")

    record["baseline_time"]         = round(t_baseline_end - t0, 3)
    record["baseline_answer"]       = baseline_answer
    record["baseline_answer_raw"]   = baseline_raw
    record["baseline_parse_error"]  = baseline_parse_error
    record["baseline_correct"]      = baseline_answer is not None and baseline_answer.upper() == gt.upper()

    # NS pipeline
    t0 = time.time()

    print(f"  --- NS extract ---")
    ns = run_ns_pipeline(problem, extract_fn)
    print(f"  Extracted   : {ns['extracted']}")
    print(f"  Z3 status   : {ns['z3_status']}")
    print(f"  Questions   : {ns['question_results']}")
    

    record["ns_time"]           = round(time.time() - t0, 3)
    record["extracted"]         = ns["extracted"]
    record["active_domains"]    = ns.get("active_domains")  
    record["question_results"]  = ns["question_results"]      
    record["formatted_output"]  = ns["formatted_output"]
    record["ns_llm_calls"]      = ns["llm_calls"]
    record["unmatched_errors"]  = ns["unmatched_errors"]

    # correctness check — find the verified label from question_results
    # question_results is a list of dicts, one per question
    # each dict maps label -> bool; correct label is the one that is True
    ns_correct = False

    #NEED TO ADD FUNCTIONALITY FOR VERIFYING MULTIPLE QUESTIONS ON THE SAME PROBLEM
    if ns["question_results"]:
        # for single-question problems, check question 0
        first_q = ns["question_results"][0]
        correct_labels = [label for label, verified in first_q.items() if verified]
        if len(correct_labels) == 1: #  only mark correct if exactly one label verified as true; 0 or 2+ means extraction was broken
            ns_correct = correct_labels[0].upper() == gt.upper()
    record["ns_correct"] = ns_correct

    # SQLite Logging
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
        model_name=MODEL_NAME,
        constraint_type_counts=counts,
    )

    # add unmatched errros
    all_unmatched.extend(ns["unmatched_errors"])
    results.append(record)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
for i, r in enumerate(results, 1):
    print(f"{'='*60}")
    print(f"RESULT {i}")
    print(f"{'='*60}")
    print(f"  Problem        : {r['problem']}")
    print(f"  Ground truth   : {r['ground_truth']}")
    print(f"  Active domains : {r['active_domains']}")   

    if r["extracted"]:
        dump = r["extracted"].model_dump()
        print(f"  Extracted      :")
        print(f"    entities     : {dump['entities']}")    
        print(f"    constraints  : {dump['constraints']}")
        print(f"    questions    : {dump['questions']}")       
    else:
        print(f"  Extracted      : FAILED")

    print(f"  Question results  : {r['question_results']}")  
    print(f"  Formatted         : {r['formatted_output']}")
    print(f"  Baseline LLM raw        : {r['baseline_answer_raw']}")
    print(f"  Baseline LLM stripped   : {r['baseline_answer']}")
    print(f"  Was NeuroSymbolic correct?     : {r['ns_correct']}")
    print(f"  Was Baseline LLM correct?      : {r['baseline_correct']}")
    print()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total           = len(results)
solved          = sum(1 for r in results if r["question_results"] is not None) 
failed          = sum(1 for r in results if r["extracted"] is None)
correct_baseline = sum(1 for r in results if r["baseline_correct"] is True)
correct_ns      = sum(1 for r in results if r["ns_correct"] is True)

print(f"{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Total problems     : {total}")
print(f"  Z3 solved          : {solved}")
print(f"  Extract failed     : {failed}")
print(f"  Baseline accuracy  : {correct_baseline}/{total}")
print(f"  Neurosymbolic acc  : {correct_ns}/{total}")
print(f"  Delta              : +{correct_ns - correct_baseline} problems")

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