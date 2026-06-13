"""
SQLite logger for the neurosymbolic logic puzzle pipeline.

One row per pipeline run (the final accepted extraction + its correctness).
Schema is forward-compatible with per-attempt logging for DPO: run_id groups
attempts, attempt_number orders them. Standard library only.

Stored-type notes (SQLite has no native bool/uuid/datetime):
  - booleans  -> INTEGER 0/1 (NULL allowed for "unknown")
  - run_id    -> TEXT (uuid4 hex)
  - timestamp -> TEXT (ISO 8601, UTC)
  - JSON cols -> TEXT (json.dumps)
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "data/logic_runs.db"

DDL = """
CREATE TABLE IF NOT EXISTS extraction_attempts (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 TEXT    NOT NULL,   -- groups all attempts for one problem
    attempt_number         INTEGER NOT NULL,   -- 1-indexed; always 1 in one-row-per-run mode
    problem_text           TEXT    NOT NULL,
    active_domains         TEXT    NOT NULL,   -- JSON array, e.g. ["ordering","grouping"]
    extracted_json         TEXT,               -- canonical JSON of the LogicProblem (NULL if extraction failed)
    schema_valid           INTEGER NOT NULL,   -- 0/1: did a LogicProblem object get built
    z3_result              TEXT,               -- 'SAT' | 'UNSAT' | 'UNKNOWN' | 'ERROR' | NULL
    answer_correct         INTEGER,            -- 0/1/NULL: matched ground truth
    ground_truth_answer    TEXT,
    constraint_type_counts TEXT,               -- JSON object, nullable (derivable from extracted_json)
    model_name             TEXT    NOT NULL,
    timestamp              TEXT    NOT NULL    -- ISO 8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_run_id   ON extraction_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_correct  ON extraction_attempts(answer_correct);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create the table and indexes if they don't exist. Idempotent."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)
        conn.commit()
    finally:
        conn.close()


def new_run_id() -> str:
    """One id per problem; reuse it across attempts of the same problem."""
    return uuid.uuid4().hex


def log_attempt(
    run_id: str,
    attempt_number: int,
    problem_text: str,
    active_domains: list[str],
    extracted_json: Optional[str],
    schema_valid: bool,
    z3_result: Optional[str],
    answer_correct: Optional[bool],
    ground_truth_answer: Optional[str],
    model_name: str,
    constraint_type_counts: Optional[dict] = None,
    db_path: str = DB_PATH,
) -> None:
    """Insert one row. Open/commit/close per call — fine at this volume."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO extraction_attempts (
                run_id, attempt_number, problem_text, active_domains,
                extracted_json, schema_valid, z3_result, answer_correct,
                ground_truth_answer, constraint_type_counts, model_name, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                attempt_number,
                problem_text,
                json.dumps(active_domains),
                extracted_json,
                int(schema_valid),
                z3_result,
                None if answer_correct is None else int(answer_correct),
                ground_truth_answer,
                None if constraint_type_counts is None else json.dumps(constraint_type_counts),
                model_name,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _categorize() -> str:
    """SQL CASE block for labeling rows by training data type."""
    return """
        CASE
            WHEN answer_correct = 1 AND extracted_json IS NOT NULL THEN 'sft_positive'
            WHEN schema_valid = 1 AND z3_result = 'SAT'  AND answer_correct = 0 THEN 'dpo_logical_fail'
            WHEN schema_valid = 1 AND z3_result = 'UNSAT'                       THEN 'dpo_unsat_negative'
            ELSE 'unusable'
        END as category
    """


def export_sft_positives(output_path: str = "data/sft_positives.jsonl", db_path: str = DB_PATH) -> int:
    """Export only correct extractions. Gold SFT training examples.
        Returns the number of exported rows. """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT run_id, problem_text, active_domains,
                   extracted_json, model_name, timestamp
            FROM extraction_attempts
            WHERE answer_correct = 1
              AND extracted_json IS NOT NULL
            ORDER BY timestamp
            """
        ).fetchall()
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")
        return len(rows)
    finally:
        conn.close()


def export_all(output_path: str = "data/all_attempts.jsonl", db_path: str = DB_PATH) -> int:
    """Export all rows with a computed category label for downstream filtering.
        Returns the number of exported rows. """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *, {_categorize()}
            FROM extraction_attempts
            ORDER BY timestamp
            """
        ).fetchall()
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")
        return len(rows)
    finally:
        conn.close()