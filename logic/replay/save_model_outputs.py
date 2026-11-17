"""
One-off script: save the real model outputs for the 10 replay-demo puzzles.

The source eval run (data/demo_runs/ns__SFT-0.6b-v4__cls-qwen3-4b-q4_K_M.jsonl) kept each
puzzle's answer and token counts but not what the models wrote. This script reruns the two
model calls for the 10 demo puzzles, with the same models and settings as that run, and saves
only what the models produced (predicted domains, raw extraction JSON, token counts, timing)
to data/demo_model_outputs.json. The replay server builds everything else from that file.

Needs Ollama running with qwen3:4b-q4_K_M and SFT_Extraction_Qwen3_0.6b-v4. Run once:

    python logic/replay/save_model_outputs.py          # runs only puzzles not in the file yet
    python logic/replay/save_model_outputs.py --all    # reruns all 10

Each puzzle is checked against the source run: the predicted domains must be the same list
(order included, since it feeds the extraction prompt and schema), and the Z3 status and final
answer must be the same. The file is only written if all 10 match.

The classifier is not run at temperature 0 (pipeline.ask_llm only pins extraction calls), so
its domains can differ between runs. It is retried up to MAX_CLASSIFIER_ATTEMPTS times until
it gives the domains recorded in the source run; the attempt count is saved in the file.
A recorded wrong prediction can be a rare one: for puzzle 3470c07b... the classifier needed
1, then 6 tries, and once did not repeat it in 10. If that happens, run the script again;
puzzles already in the file are kept, so only the missing ones are retried.

Saved timings are for this machine, with both models already loaded (see warm_up).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(HERE, "..", "pipeline")
DATA_DIR = os.path.join(HERE, "..", "data")
sys.path.insert(0, PIPELINE_DIR)

try:  # same optional local .env as chatbox.py; must load before pipeline reads OLLAMA_URL
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PIPELINE_DIR, ".env"))
except ImportError:
    pass

from pipeline import ask_llm, classify_domains, extract_logic_problem, z3_solve, new_usage
from validators import build_hybrid_schema

from demo_examples import DEMO_PROBLEM_IDS

# Same models and settings as the source run (run.py MODEL_SETS / FINETUNED_MODELS).
SOURCE_RUN = "ns__SFT-0.6b-v4__cls-qwen3-4b-q4_K_M.jsonl"
CLASSIFIER_MODEL = "qwen3:4b-q4_K_M"
EXTRACTION_MODEL = "SFT_Extraction_Qwen3_0.6b-v4"
EXTRACTION_FINETUNED = True

MAX_CLASSIFIER_ATTEMPTS = 10

RUN_PATH = os.path.join(DATA_DIR, "demo_runs", SOURCE_RUN)
TEST_SET_PATH = os.path.join(DATA_DIR, "sft_test.jsonl")
OUT_PATH = os.path.join(DATA_DIR, "demo_model_outputs.json")


def load_jsonl(path: str, key: str) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row[key]] = row
    return rows


def classify_until_match(problem_text: str, recorded: list[str]) -> dict:
    """Call the classifier until it returns the domains recorded in the source run.
    Returns the matching call's output, or raises after MAX_CLASSIFIER_ATTEMPTS."""
    seen = []
    for attempt in range(1, MAX_CLASSIFIER_ATTEMPTS + 1):
        usage = new_usage()
        t0 = time.time()
        try:
            domains = classify_domains(problem_text, model=CLASSIFIER_MODEL, usage=usage)
        except ValueError as e:   # unknown domain name; a connection error (RuntimeError) still stops the script
            seen.append(f"error: {e}")
            continue
        time_ms = int((time.time() - t0) * 1000)
        if domains == recorded:
            return {
                "predicted_domains": domains,
                "classifier": {
                    "model": CLASSIFIER_MODEL,
                    "in_tok": usage["in_tok"],
                    "out_tok": usage["out_tok"],
                    "time_ms": time_ms,
                    "attempts": attempt,
                },
            }
        seen.append(domains)
        print(f"    attempt {attempt}: got {domains}, source run has {recorded}; retrying")
    raise RuntimeError(
        f"classifier never returned {recorded} in {MAX_CLASSIFIER_ATTEMPTS} attempts (got {seen})"
    )


def run_one(problem_id: str, problem_text: str, run_row: dict) -> dict:
    out = {"problem_id": problem_id}
    out.update(classify_until_match(problem_text, run_row["predicted_domains"]))
    domains = out["predicted_domains"]

    LogicProblem = build_hybrid_schema(domains)
    usage = new_usage()
    t0 = time.time()
    extracted, _unmatched, llm_calls, raw = extract_logic_problem(
        problem_text, domains, LogicProblem,
        model=EXTRACTION_MODEL, finetuned=EXTRACTION_FINETUNED, usage=usage,
    )
    time_ms = int((time.time() - t0) * 1000)
    if extracted is None:
        raise RuntimeError("extraction did not produce valid JSON for the schema")

    # Same answer rule as run.py: every verified label of the first question, joined by commas.
    z3_result = z3_solve(extracted)
    verified = ([label for label, ok in z3_result["question_results"][0].items() if ok]
                if z3_result.get("question_results") else [])
    answer = ",".join(verified) or None

    if z3_result["status"] != run_row["z3_status"]:
        raise RuntimeError(f"z3 status {z3_result['status']!r}, source run has {run_row['z3_status']!r}")
    if answer != run_row["answer"]:
        raise RuntimeError(f"answer {answer!r}, source run has {run_row['answer']!r}")

    out["extraction_raw"] = raw
    out["extraction"] = {
        "model": EXTRACTION_MODEL,
        "in_tok": usage["in_tok"],
        "out_tok": usage["out_tok"],
        "time_ms": time_ms,
        "llm_calls": llm_calls,
    }
    # Extraction runs at temperature 0, so the same token counts as the source run mean the
    # same prompt and (almost certainly) the same output. Reported, not required.
    same_tokens = (usage["in_tok"] == run_row["ext_in_tok"]
                   and usage["out_tok"] == run_row["ext_out_tok"])
    print(f"    domains {domains} (attempt {out['classifier']['attempts']})  answer {answer}  "
          f"extraction tokens {usage['in_tok']}/{usage['out_tok']} "
          f"({'same as' if same_tokens else 'DIFFERENT from'} source run "
          f"{run_row['ext_in_tok']}/{run_row['ext_out_tok']})")
    return out


def warm_up() -> None:
    """Load both models into memory first, so the first puzzle's saved timing is the model
    call itself and not ~20 s of reading the model from disk. The replies are discarded."""
    print("Loading models...")
    for model in (CLASSIFIER_MODEL, EXTRACTION_MODEL):
        ask_llm(prompt="Reply with OK.", system="", model=model)


def load_saved() -> dict:
    """Entries already in the output file, by problem id. Each one matched the source run
    when it was saved."""
    if not os.path.isfile(OUT_PATH):
        return {}
    with open(OUT_PATH, encoding="utf-8") as f:
        return {ex["problem_id"]: ex for ex in json.load(f)["examples"]}


def main():
    run_rows = load_jsonl(RUN_PATH, "problem_id")
    test_rows = load_jsonl(TEST_SET_PATH, "run_id")
    saved = {} if "--all" in sys.argv else load_saved()
    if any(pid not in saved for pid in DEMO_PROBLEM_IDS):
        warm_up()

    examples, failures = [], []
    for n, problem_id in enumerate(DEMO_PROBLEM_IDS, 1):
        print(f"[{n}/{len(DEMO_PROBLEM_IDS)}] {problem_id}", flush=True)
        if problem_id in saved:
            examples.append(saved[problem_id])
            print("    already saved, kept (--all reruns it)")
            continue
        try:
            examples.append(run_one(problem_id, test_rows[problem_id]["problem_text"],
                                    run_rows[problem_id]))
        except Exception as e:  # noqa: BLE001 - report every failing id, then exit non-zero
            failures.append((problem_id, f"{type(e).__name__}: {e}"))
            print(f"    FAILED: {type(e).__name__}: {e}")

    if failures:
        print(f"\n{len(failures)} of {len(DEMO_PROBLEM_IDS)} did not match the source run. "
              f"Nothing was written.")
        for problem_id, err in failures:
            print(f"  {problem_id}: {err}")
        sys.exit(1)

    payload = {
        "source_run": SOURCE_RUN,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("Real model outputs for the replay demo, written by "
                 "logic/replay/save_model_outputs.py. Each entry matched the source run's "
                 "predicted domains, Z3 status and answer."),
        "examples": examples,
    }
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, OUT_PATH)
    print(f"\nAll {len(examples)} match the source run. Wrote {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
