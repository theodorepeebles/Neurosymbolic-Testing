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
    -- [CONSTRAINT TYPE BREAKDOWN]
    -- Per-type counts across all parts of the problem (global + question + choice),
    -- recursing into logical wrappers. One pair per type defined in validators.py.
    -- See eval_metrics.COUNTED_CONSTRAINT_TYPES.
    -- ==========================================
    expected_before_count                 INTEGER,
    extracted_before_count                INTEGER,
    expected_immediately_before_count     INTEGER,
    extracted_immediately_before_count    INTEGER,
    expected_adjacent_count               INTEGER,
    extracted_adjacent_count              INTEGER,
    expected_slot_fixed_count             INTEGER,
    extracted_slot_fixed_count            INTEGER,
    expected_is_truth_teller_count        INTEGER,
    extracted_is_truth_teller_count       INTEGER,
    expected_is_deceiver_count            INTEGER,
    extracted_is_deceiver_count           INTEGER,
    expected_same_group_count             INTEGER,
    extracted_same_group_count            INTEGER,
    expected_different_group_count        INTEGER,
    extracted_different_group_count       INTEGER,
    expected_exactly_n_count              INTEGER,
    extracted_exactly_n_count             INTEGER,
    expected_is_in_count                  INTEGER,
    extracted_is_in_count                 INTEGER,
    expected_if_then_count                INTEGER,
    extracted_if_then_count               INTEGER,
    expected_not_count                    INTEGER,
    extracted_not_count                   INTEGER,
    expected_and_count                    INTEGER,
    extracted_and_count                   INTEGER,
    expected_or_count                     INTEGER,
    extracted_or_count                    INTEGER,

    -- ==========================================
    -- [RAW OUTPUT & EVALUATION RESULTS]
    -- ==========================================
    expected_json                         TEXT,             -- Gold extraction (run_id-keyed); makes the DB self-contained / decoupled from sft_test.jsonl
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
    generation_time_ms                    INTEGER,

    -- ==========================================
    -- [CONSTRAINT -> SOURCE-TEXT ATTRIBUTION]
    -- Maps each extracted constraint to where in problem_text it came from.
    -- Keys are namespaced constraint ids (c_0 = track_c_0, q0.qc_0, q0.A_0; see
    -- attribution.py). methods: cid -> 'option_a'|'bm25_fallback'|'unattributed'.
    -- spans:   cid -> [char_start, char_end] into problem_text (only when located).
    -- ==========================================
    attribution_methods                   TEXT,             -- JSON dict cid -> attribution method
    attribution_spans                     TEXT              -- JSON dict cid -> [start, end]
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
    "expected_before_count", "extracted_before_count",
    "expected_immediately_before_count", "extracted_immediately_before_count",
    "expected_adjacent_count", "extracted_adjacent_count",
    "expected_slot_fixed_count", "extracted_slot_fixed_count",
    "expected_is_truth_teller_count", "extracted_is_truth_teller_count",
    "expected_is_deceiver_count", "extracted_is_deceiver_count",
    "expected_same_group_count", "extracted_same_group_count",
    "expected_different_group_count", "extracted_different_group_count",
    "expected_exactly_n_count", "extracted_exactly_n_count",
    "expected_is_in_count", "extracted_is_in_count",
    "expected_if_then_count", "extracted_if_then_count",
    "expected_not_count", "extracted_not_count",
    "expected_and_count", "extracted_and_count",
    "expected_or_count", "extracted_or_count",
    "expected_json", "extracted_json", "schema_valid", "constraint_type_counts",
    "z3_result", "answer_correct", "ground_truth_answer", "error_traceback",
    "completion_token_count", "generation_time_ms",
    "attribution_methods", "attribution_spans",
]

# Columns declared NOT NULL in the DDL — must be present (non-None) in every row.
_REQUIRED = {
    "run_id", "attempt_number", "extraction_model_name", "timestamp", "environment",
    "problem_text", "active_domains", "schema_valid",
}

# Columns stored as JSON text; dicts/lists are json.dumps'd on the way in.
_JSON_COLUMNS = {"active_domains", "constraint_type_counts", "expected_json",
                 "attribution_methods", "attribution_spans"}

# Columns that hold booleans in Python but INTEGER 0/1 in SQLite.
_BOOL_COLUMNS = {"exact_entity_match", "exact_global_constraint_match",
                 "exact_question_constraint_match", "exact_choice_constraint_match",
                 "schema_valid", "answer_correct"}

# SQLite affinity per column, used only by the auto-migration below. Anything not
# listed here is added as INTEGER (all the count columns).
_NON_INTEGER_TYPES = {
    "run_id": "TEXT", "extraction_model_name": "TEXT", "timestamp": "TEXT",
    "environment": "TEXT", "problem_text": "TEXT", "active_domains": "TEXT",
    "text_lexical_density": "REAL", "expected_json": "TEXT", "extracted_json": "TEXT",
    "constraint_type_counts": "TEXT", "z3_result": "TEXT",
    "ground_truth_answer": "TEXT", "error_traceback": "TEXT",
    "attribution_methods": "TEXT", "attribution_spans": "TEXT",
}


def _ensure_columns(conn) -> None:
    """Add any COLUMNS missing from an existing table (idempotent migration).

    `CREATE TABLE IF NOT EXISTS` leaves a pre-existing table untouched, so new
    columns added to the DDL/COLUMNS list won't appear on an old sft_test.db
    without this. ALTER TABLE ADD COLUMN appends nullable columns (existing rows
    get NULL), which is exactly the behaviour we want.
    """
    existing = {r[1] for r in conn.execute("PRAGMA table_info(extraction_attempts)")}
    for col in COLUMNS:
        if col not in existing:
            col_type = _NON_INTEGER_TYPES.get(col, "INTEGER")
            conn.execute(f"ALTER TABLE extraction_attempts ADD COLUMN {col} {col_type}")


def init_db(db_path: str = DB_PATH) -> None:
    """Create the table and indexes if they don't exist, then migrate in any new
    columns. Idempotent."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)
        _ensure_columns(conn)
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
