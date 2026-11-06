"""
JSONL logger for the demo benchmark: one file per run setup, one line per problem.

This is the file-based counterpart to logger.py (which writes to sft_test.db). run.py
writes here when LOG_TO_DEMO is on; demo_summary.py turns the files into a comparison table.

Every line has the common fields, so pipeline and baseline files compare the same way:
    problem_id, answer, correct_answer, correct, in_tok, out_tok, time_ms, error
Baseline lines add:
    model
Pipeline lines add:
    predicted_domains, correct_domains, classifier_correct,
    classifier_model, cls_in_tok, cls_out_tok, extraction_model, ext_in_tok, ext_out_tok,
    z3_status
For pipeline lines, in_tok/out_tok are classifier + extraction added together.
The model names are there so demo_summary.py can price tokens with cost().
"""

import json
import os

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demo_runs")

COMMON_FIELDS = ["problem_id", "answer", "correct_answer", "correct",
                 "in_tok", "out_tok", "time_ms", "error"]
BASELINE_FIELDS = ["model"]
PIPELINE_FIELDS = ["predicted_domains", "correct_domains", "classifier_correct",
                   "classifier_model", "cls_in_tok", "cls_out_tok",
                   "extraction_model", "ext_in_tok", "ext_out_tok", "z3_status"]


def short_name(model: str) -> str:
    """Filename-safe model name: 'gpt-oss:120b-cloud' -> 'gpt-oss-120b',
    'nemotron-3-super:cloud' -> 'nemotron-3-super'."""
    name = model.removesuffix(":latest").removesuffix(":cloud").removesuffix("-cloud")
    return name.replace(":", "-").replace("/", "-")


def think_tag(think) -> str:
    """True -> 'think-on', False -> 'think-off', 'high' -> 'think-high'."""
    if think is True:
        return "think-on"
    if think is False or think is None:
        return "think-off"
    return f"think-{think}"


def pipeline_filename(extraction_name: str, classifier_model: str | None) -> str:
    """e.g. ns__SFT-0.6b-v4__cls-qwen3-8b.jsonl. classifier_model=None means ground-truth domains."""
    cls = short_name(classifier_model) if classifier_model else "gt"
    return f"ns__{short_name(extraction_name)}__cls-{cls}.jsonl"


def baseline_filename(model: str, think) -> str:
    """e.g. baseline__gpt-oss-120b__think-on.jsonl"""
    return f"baseline__{short_name(model)}__{think_tag(think)}.jsonl"


def start_demo_file(filename: str) -> str:
    """Create (or empty) a run's file so a rerun doesn't mix with old results. Returns its path."""
    os.makedirs(DEMO_DIR, exist_ok=True)
    path = os.path.join(DEMO_DIR, filename)
    open(path, "w", encoding="utf-8").close()
    return path


def resume_demo_file(filename: str) -> tuple[str, set]:
    """Open a run's file for resuming: keep its results and return (path, problem_ids already done).

    A run stopped with Ctrl+C can leave a half-written last line; any line that isn't valid
    JSON is dropped (and the file rewritten without it) so that problem is simply run again.
    """
    os.makedirs(DEMO_DIR, exist_ok=True)
    path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").close()
        return path, set()

    good, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        with open(path, "w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r) + "\n")
        print(f"  dropped {dropped} unreadable line(s) from {filename}")
    return path, {r["problem_id"] for r in good}


def load_demo_file(path: str) -> list[dict]:
    """All results in a run's file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log_demo(path: str, record: dict) -> None:
    """Append one problem's result to a run's JSONL file.

    Fields are written in a fixed order (common, then baseline- or pipeline-only).
    Raises if a field is missing, so a bad record fails loudly.
    """
    extra = PIPELINE_FIELDS if "cls_in_tok" in record else BASELINE_FIELDS
    missing = [f for f in COMMON_FIELDS + extra if f not in record]
    if missing:
        raise KeyError(f"log_demo record is missing fields: {missing}")
    ordered = {f: record[f] for f in COMMON_FIELDS + extra}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ordered) + "\n")
