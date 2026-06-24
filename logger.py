"""
SQLite logger for the neurosymbolic logic puzzle pipeline.

This database is the **testing / validation log** (sft_test.db). It is populated by
run.py when a trained extraction model is evaluated over sft_test.jsonl: one row per
(test example x model), recording how the model's extraction compared against the gold
extraction stored in the dataset row. It is NOT written to during dataset generation —
the dataset (sft_dataset.jsonl) and this DB are deliberately decoupled.

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

# all callers of this module live in NS_Math/logic/, so the path is relative to that directory
DB_PATH = "../data/sft_test.db"

DDL = """
CREATE TABLE IF NOT EXISTS extraction_attempts (
    -- ==========================================
    -- [CORE IDENTIFIERS & METADATA]
    -- ==========================================
    id                                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                                TEXT    NOT NULL,
    attempt_number                        INTEGER NOT NULL,
    extraction_model_name                 TEXT    NOT NULL,
    timestamp                             TEXT    NOT NULL,
    environment                           TEXT    NOT NULL, -- 'controlled' (expected known) or 'real_world' (expected NULL)

    -- ==========================================
    -- [INPUT PRIMITIVES & TEXT METRICS]
    -- ==========================================
    problem_text                          TEXT    NOT NULL,
    active_domains                        TEXT    NOT NULL,
    prompt_token_count                    INTEGER,
    text_word_count                       INTEGER,          -- Raw word count of the problem text
    text_lexical_density                  REAL,             -- Measure of unique vocabulary to track prompt complexity

    -- ==========================================
    -- [ENTITY & DOMAIN BOUNDARY COMPARISONS]
    -- ==========================================
    expected_entity_count                 INTEGER,
    extracted_entity_count                INTEGER,
    exact_entity_match                    INTEGER,          -- 0/1: Set equality (Order-agnostic: ["Sam","Bob"] == ["Bob","Sam"])

    expected_slot_count                   INTEGER,          -- Populated if 'ordering' active
    extracted_slot_count                  INTEGER,

    expected_group_count                  INTEGER,          -- Populated if 'grouping' active
    extracted_group_count                 INTEGER,

    -- ==========================================
    -- [GLOBAL CONSTRAINT COMPARISONS]
    -- ==========================================
    expected_global_constraint_count      INTEGER,
    extracted_global_constraint_count     INTEGER,
    exact_global_constraint_match         INTEGER,          -- 0/1: Semantic set equality of the constraint objects, ignoring JSON array order

    -- ==========================================
    -- [QUESTION CONSTRAINT COMPARISONS]
    -- ==========================================
    expected_question_constraint_count    INTEGER,
    extracted_question_constraint_count   INTEGER,
    exact_question_constraint_match       INTEGER,          -- 0/1: Semantic set equality of the constraint objects, ignoring JSON array order

    -- ==========================================
    -- [ANSWER CHOICE COMPARISONS]
    -- ==========================================
    expected_choice_count                 INTEGER,
    extracted_choice_count                INTEGER,

    expected_choice_constraint_count      INTEGER,          -- Total constraints across all A/B/C/D blocks
    extracted_choice_constraint_count     INTEGER,
    exact_choice_constraint_match         INTEGER,          -- 0/1: Semantic set equality of the constraint objects, ignoring JSON array order

    -- ==========================================
    -- [LOGICAL COMPLEXITY TRACKING]
    -- ==========================================
    expected_logical_wrapper_count        INTEGER,          -- Counts of nested wrappers ('if_then', 'not', 'or', 'and')
    extracted_logical_wrapper_count       INTEGER,

    -- ==========================================
    -- [RAW OUTPUT & EVALUATION RESULTS]
    -- ==========================================
    extracted_json                        TEXT,
    schema_valid                          INTEGER NOT NULL,
    constraint_type_counts                TEXT,             -- JSON dict mapping rule primitives to counts

    z3_result                             TEXT,             -- 'SAT' | 'UNSAT' | 'UNKNOWN' | 'ERROR' | NULL
    answer_correct                        INTEGER,
    ground_truth_answer                   TEXT,
    error_traceback                       TEXT,

    -- ==========================================
    -- [PERFORMANCE TRACKING]
    -- ==========================================
    completion_token_count                INTEGER,
    generation_time_ms                    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_run_id   ON extraction_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_correct  ON extraction_attempts(answer_correct);
"""

# Every insertable column, in DDL order, excluding the autoincrement `id`.
# log_attempt builds its INSERT from this list, so adding a column means editing
# the DDL above and this list only.
COLUMNS = [
    "run_id", "attempt_number", "extraction_model_name", "timestamp", "environment",
    "problem_text", "active_domains", "prompt_token_count", "text_word_count",
    "text_lexical_density",
    "expected_entity_count", "extracted_entity_count", "exact_entity_match",
    "expected_slot_count", "extracted_slot_count",
    "expected_group_count", "extracted_group_count",
    "expected_global_constraint_count", "extracted_global_constraint_count",
    "exact_global_constraint_match",
    "expected_question_constraint_count", "extracted_question_constraint_count",
    "exact_question_constraint_match",
    "expected_choice_count", "extracted_choice_count",
    "expected_choice_constraint_count", "extracted_choice_constraint_count",
    "exact_choice_constraint_match",
    "expected_logical_wrapper_count", "extracted_logical_wrapper_count",
    "extracted_json", "schema_valid", "constraint_type_counts",
    "z3_result", "answer_correct", "ground_truth_answer", "error_traceback",
    "completion_token_count", "generation_time_ms",
]

# Columns declared NOT NULL in the DDL — must be present (non-None) in every row.
_REQUIRED = {
    "run_id", "attempt_number", "extraction_model_name", "timestamp", "environment",
    "problem_text", "active_domains", "schema_valid",
}

# Columns stored as JSON text; dicts/lists are json.dumps'd on the way in.
_JSON_COLUMNS = {"active_domains", "constraint_type_counts"}

# Columns that hold booleans in Python but INTEGER 0/1 in SQLite.
_BOOL_COLUMNS = {"exact_entity_match", "exact_global_constraint_match",
                 "exact_question_constraint_match", "exact_choice_constraint_match",
                 "schema_valid", "answer_correct"}


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


def _coerce(col: str, value):
    """Convert a Python value to its SQLite-storable form for column `col`."""
    if value is None:
        return None
    if col in _JSON_COLUMNS and not isinstance(value, str):
        return json.dumps(value)
    if col in _BOOL_COLUMNS and isinstance(value, bool):
        return int(value)
    return value


def log_attempt(row: dict, db_path: str = DB_PATH) -> None:
    """Insert one row from a dict keyed by column name.

    Callers (run.py) compute the full comparison row and hand it over here. Missing
    optional columns default to NULL; `timestamp` defaults to now (UTC). Open/commit/
    close per call — fine at this volume.
    """
    row = dict(row)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    missing = [c for c in _REQUIRED if row.get(c) is None]
    if missing:
        raise ValueError(f"log_attempt missing required column(s): {missing}")

    values = [_coerce(c, row.get(c)) for c in COLUMNS]
    placeholders = ", ".join("?" for _ in COLUMNS)
    sql = f"INSERT INTO extraction_attempts ({', '.join(COLUMNS)}) VALUES ({placeholders})"

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()
