"""
Run the symbolic explanation engine on a single stored run.

Fetches one row from sft_test.db by run_id, rebuilds the LogicProblem (gold extraction by
default, the model's extraction with --source extracted), runs explanation.explain_problem,
prints a human-readable breakdown, and (unless --no-json) upserts the serialized
ExplanationStruct(s) into data/explanations.jsonl (one row per run_id).

Usage:
    python logic/explanation_debug.py <run_id>
    python logic/explanation_debug.py                       # omit run_id -> pick a random row
    python logic/explanation_debug.py <run_id> --source extracted
    python logic/explanation_debug.py <run_id> --no-json
    python logic/explanation_debug.py <run_id> --db ../data/sft_test.db

The DB defaults to ../data/sft_test.db (the file logger.py writes). Find a run_id to try
(sqlite3 / any DB browser):
    SELECT run_id FROM extraction_attempts LIMIT 5;

Paths default relative to this file (logic/), so the command works from any working directory.
The JSON output is written to ../data/explanations.jsonl as JSON-lines: one object per run_id,
shaped {run_id, source, model, generated_at, ground_truth_answer, explanations:[<struct>, ...]}.
The console view is for humans; the JSONL is the machine handoff to the later LLM stage.
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

from validators import build_hybrid_schema
from explanation import explain_problem, explanation_to_dict, build_context

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "..", "data", "sft_test.db")
JSONL_PATH = os.path.join(HERE, "..", "data", "explanations.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# DB access
# ─────────────────────────────────────────────────────────────────────────────

def _require_db(db_path: str) -> None:
    if not os.path.isfile(db_path):
        sys.exit(f"Database not found: {db_path}\n"
                 f"Run run.py first to populate sft_test.db, or pass --db.")


def pick_random_run_id(db_path: str) -> str:
    """Pick a random run_id from the DB (used when no run_id is given on the CLI)."""
    _require_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT run_id FROM extraction_attempts ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        sys.exit(f"No rows in {db_path}")
    return row[0]


def fetch_run(db_path: str, run_id: str) -> sqlite3.Row:
    _require_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT run_id, extraction_model_name, problem_text, active_domains, "
            "expected_json, extracted_json, ground_truth_answer "
            "FROM extraction_attempts WHERE run_id = ? LIMIT 1",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        sys.exit(f"No row with run_id={run_id!r} in {db_path}")
    return row


def _loads(maybe_json):
    """active_domains / *_json are stored as JSON text; tolerate already-parsed values."""
    if maybe_json is None:
        return None
    if isinstance(maybe_json, str):
        return json.loads(maybe_json)
    return maybe_json


# ─────────────────────────────────────────────────────────────────────────────
# JSONL upsert (one row per run_id)
# ─────────────────────────────────────────────────────────────────────────────

def upsert_jsonl(path: str, run_id: str, record: dict) -> None:
    rows: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                rid = obj.get("run_id")
                if rid:
                    rows[rid] = obj
    rows[run_id] = record
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Console rendering
# ─────────────────────────────────────────────────────────────────────────────

QTYPE_FRAME = {
    ("could_be_true",  "MUS_REFUTATION"): "is IMPOSSIBLE",
    ("could_be_false", "MUS_REFUTATION"): "must ALWAYS be true (cannot be false)",
    ("must_be_true",   "COUNTEREXAMPLE"): "is NOT necessarily true",
    ("must_be_false",  "COUNTEREXAMPLE"): "is NOT necessarily false (it can hold)",
}


def _clue(cid: str, labels: dict, span_index: dict) -> str:
    label = labels.get(cid, cid)
    sp = span_index.get(cid)
    where = f" [chars {sp[0]}-{sp[1]}]" if sp else (" [structural]" if cid.startswith("implicit.") else " [no span]")
    return f"{cid}: {label}{where}"


def render(structs, labels, source, model, ground_truth):
    out = []
    for st in structs:
        span_index = st.constraint_span_index
        out.append("=" * 78)
        out.append(f"QUESTION {st.question_index}   (source={source}, model={model})")
        out.append("-" * 78)
        out.append(f"Problem: {st.problem_text}")
        out.append("")

        verified = [c.answer for c in st.correct]
        gt_note = ""
        if ground_truth is not None:
            ok = (len(verified) == 1 and verified[0] == ground_truth)
            gt_note = f"   [ground_truth={ground_truth} -> {'MATCH' if ok else 'MISMATCH'}]"
        out.append(f"Verified (correct): {verified or '(none)'}{gt_note}")
        out.append("")

        for c in st.correct:
            out.append(f"CORRECT [{c.answer}]")
            if c.forced_bindings:
                out.append("  Forced bindings:")
                for var, val, cids in c.forced_bindings:
                    by = ", ".join(cids)
                    out.append(f"    {var} = {val}   (forced by {by})")
            if c.free_bindings:
                out.append("  Free (consistent but not uniquely determined):")
                for var, val in c.free_bindings:
                    out.append(f"    {var} = {val}   (one valid choice)")
            out.append("")

        for w in st.wrong:
            frame = QTYPE_FRAME.get((w.question_type, w.query_type), w.query_type)
            out.append(f"WRONG [{w.answer}]  ({w.question_type})  ->  \"{w.proposition_text}\" {frame}")
            out.append(f"  tier: {w.tier.value}")

            if w.query_type == "MUS_REFUTATION":
                if w.single_refutations:
                    out.append("  Direct single-clue violations:")
                    for s in w.single_refutations:
                        out.append(f"    - {_clue(s.constraint_ids[0], labels, span_index)}")
                if w.all_mus:
                    out.append(f"  Primary reason (minimal conflicting clue set):")
                    for cid in w.all_mus[0]:
                        out.append(f"    - {_clue(cid, labels, span_index)}")
                    if w.narrative_chain:
                        out.append("  Narrative chain:")
                        for i, s in enumerate(w.narrative_chain, 1):
                            mark = "  <-- CONTRADICTION" if s.is_contradiction else ""
                            cid = s.constraint_ids[0]
                            out.append(f"    {i}. (depth {s.depth}) {_clue(cid, labels, span_index)}{mark}")
                    if len(w.all_mus) > 1:
                        out.append(f"  Other independent reasons (MUSes): "
                                   + "; ".join("{" + ", ".join(m) + "}" for m in w.all_mus[1:]))
                if w.mcs:
                    fix = ", ".join(labels.get(c, c) for c in w.mcs)
                    out.append(f"  Would become valid if removed (MCS): {fix}")
            else:  # COUNTEREXAMPLE
                out.append("  Witness arrangement (a valid model where the claim does not hold as required):")
                for var, val in sorted((w.counterexample_model or {}).items()):
                    out.append(f"    {var} = {val}")
            out.append("")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run the symbolic explanation engine on a stored run.")
    ap.add_argument("run_id", nargs="?", default=None,
                    help="run_id from extraction_attempts (omit to pick a random row)")
    ap.add_argument("--source", choices=["gold", "extracted"], default="gold",
                    help="which extraction to explain (default: gold)")
    ap.add_argument("--db", default=DEFAULT_DB, help="path to sft_test.db")
    ap.add_argument("--no-json", action="store_true", help="skip writing data/explanations.jsonl")
    args = ap.parse_args()

    run_id = args.run_id
    if run_id is None:
        run_id = pick_random_run_id(args.db)
        print(f"No run_id given - picked a random one: {run_id}\n")

    row = fetch_run(args.db, run_id)
    problem_text = row["problem_text"]
    active_domains = _loads(row["active_domains"])
    model = row["extraction_model_name"]
    ground_truth = row["ground_truth_answer"]

    source_json = _loads(row["expected_json"] if args.source == "gold" else row["extracted_json"])
    if source_json is None:
        sys.exit(f"Row has no {args.source} extraction (column is NULL) for run_id={run_id}")

    lp = build_hybrid_schema(active_domains)(**source_json)
    structs = explain_problem(lp, problem_text)
    labels = build_context(lp).cid_label  # cid -> human label, for console only

    print(render(structs, labels, args.source, model, ground_truth))

    if not args.no_json:
        record = {
            "run_id": run_id,
            "source": args.source,
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ground_truth_answer": ground_truth,
            "explanations": [explanation_to_dict(st) for st in structs],
        }
        upsert_jsonl(JSONL_PATH, run_id, record)
        print("=" * 78)
        print(f"Wrote explanation for run_id={run_id} -> {os.path.normpath(JSONL_PATH)}")


if __name__ == "__main__":
    main()
