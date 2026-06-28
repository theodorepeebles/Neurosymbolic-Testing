import json
import os
import sqlite3
from datetime import datetime

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier, _tree

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Which model(s) to analyze, by extraction_model_name as logged in the DB.
#   None         -> all models (pooled together)
#   ["a"]        -> just model "a"
#   ["a", "b"]   -> models "a" and "b" pooled together
MODEL_NAMES = ["SFT_Extraction_Qwen3_0.6b-v4"]

# Numeric feature columns fed to the decision tree (domain dummies are added separately).
# This is the exact set the tree has always used; it is listed explicitly because the
# export query is now `SELECT *` (which also returns text columns that must not be features).
FEATURE_COLUMNS = [
    "text_word_count",
    "text_lexical_density",
    "expected_entity_count",
    "expected_slot_count",
    "expected_group_count",
    "expected_global_constraint_count",
    "expected_logical_wrapper_count",
    "expected_question_constraint_count",
    "expected_choice_count",
    "expected_choice_constraint_count",
    "expected_before_count",
    "expected_immediately_before_count",
    "expected_adjacent_count",
    "expected_slot_fixed_count",
    "expected_is_truth_teller_count",
    "expected_is_deceiver_count",
    "expected_same_group_count",
    "expected_different_group_count",
    "expected_exactly_n_count",
    "expected_is_in_count",
    "expected_if_then_count",
    "expected_not_count",
    "expected_and_count",
    "expected_or_count",
]

# Where structured failure reports (consumed by the HTML debug viewer) are written.
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "debug_reports")


def get_failure_rules(tree, feature_names, min_failure_rate=0.60, min_samples=5):
    """
    Walks the decision tree leaf nodes and collects paths that heavily lead to
    failures (Class 0), ignoring successful or low-sample branches.

    Each returned entry is (rule, failure_rate, total_samples, leaf_node_id). The leaf
    node id lets callers recover the exact rows in the pocket via tree.apply().
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node],  current_rule + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], current_rule + [f"{name} > {threshold:.2f}"])
        else:
            # sklearn >=1.4 stores tree_.value as normalized class proportions (sum to 1.0),
            # not raw counts. Use n_node_samples for the true count; normalize value for the
            # rate so this works whether value holds counts or proportions.
            node_value   = tree_.value[node][0]
            s            = node_value.sum()
            failure_rate = node_value[0] / s if s > 0 else 0.0  # class index 0 = answer_correct==0 (failure)
            total        = int(tree_.n_node_samples[node])
            if failure_rate >= min_failure_rate and total >= min_samples:
                rules.append((current_rule, failure_rate, total, node))

    recurse(0, [])
    return sorted(rules, key=lambda x: x[1], reverse=True)


def _parse_condition(rule):
    """Turn human rule clauses (["x <= 0.50", "y > 1.00"]) into structured form."""
    parsed = []
    for clause in rule:
        if " <= " in clause:
            feature, threshold = clause.split(" <= ", 1)
            op = "<="
        elif " > " in clause:
            feature, threshold = clause.split(" > ", 1)
            op = ">"
        else:
            continue
        try:
            threshold = float(threshold)
        except ValueError:
            pass
        parsed.append({"feature": feature.strip(), "op": op, "threshold": threshold})
    return parsed


def _records_from_df(db_df):
    """JSON-safe list of row dicts from a DataFrame (NaN->null, numpy scalars->py),
    with active_domains parsed from its stored JSON string into a list."""
    records = json.loads(db_df.to_json(orient="records"))
    for r in records:
        ad = r.get("active_domains")
        if isinstance(ad, str):
            try:
                r["active_domains"] = json.loads(ad)
            except (ValueError, TypeError):
                r["active_domains"] = []
        elif ad is None:
            r["active_domains"] = []
    return records


def _domains_union(records):
    """Sorted set of all single domains seen across rows (for the viewer's filter buttons)."""
    domains = set()
    for r in records:
        for d in r.get("active_domains") or []:
            domains.add(d)
    return sorted(domains)


def _write_report(db_path, generated_at, model_names, baseline, threshold,
                  domains, columns, records, profiles_out):
    """Write one self-contained profiles_<ts>.json report for the HTML debug viewer."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "ns_failure_report",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "model_names": list(model_names) if model_names else None,
        "db_path": os.path.abspath(db_path),
        "baseline_failure_rate": baseline,
        "profile_threshold": threshold,
        "domains": domains,
        "columns": columns,
        "rows": records,
        "profiles": profiles_out,
    }
    out_path = os.path.join(REPORTS_DIR, f"profiles_{generated_at.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, separators=(",", ":"))
    return out_path


def analyze_and_export_failure_targets(db_path: str, model_names=None):
    """
    Queries the SQLite data pipeline, builds the programmatic tree, isolates independent
    failure profiles, prints them, and writes a structured JSON report for the debug viewer.

    `model_names` filters by extraction_model_name: None = all models, or a list
    of names to pool together.
    """
    generated_at = datetime.now()
    label = "ALL MODELS" if not model_names else ", ".join(model_names)
    print(f"Querying DB: {db_path}")
    print(f"  Model(s)         : {label}")
    conn = sqlite3.connect(db_path)
    # TODO: re-add prompt_token_count once pipeline populates it (currently always NULL)
    where = "environment = 'controlled'"
    params = []
    if model_names:
        placeholders = ", ".join("?" for _ in model_names)
        where += f" AND extraction_model_name IN ({placeholders})"
        params = list(model_names)
    # Model-filtered rows drive the tree/profiles. SELECT * so we keep id/run_id/json/etc.
    df = pd.read_sql_query(f"SELECT * FROM extraction_attempts WHERE {where}", conn, params=params)
    # Whole-DB snapshot powers the viewer's "Whole DB" tab + row lookups. Scope it to the
    # same model(s) as the profiles (but without the environment restriction, so it captures
    # all of that model's rows — just without the profile analysis). None → every row, all models.
    if model_names:
        db_placeholders = ", ".join("?" for _ in model_names)
        db_df = pd.read_sql_query(
            f"SELECT * FROM extraction_attempts WHERE extraction_model_name IN ({db_placeholders})",
            conn, params=list(model_names))
    else:
        db_df = pd.read_sql_query("SELECT * FROM extraction_attempts", conn)
    conn.close()

    records = _records_from_df(db_df)
    columns = list(db_df.columns)
    domains = _domains_union(records)

    profiles_out = []
    baseline = None
    threshold = None

    if df.empty or df["answer_correct"].dropna().empty:
        print(f"Insufficient data to extract patterns for {label}.")
        out_path = _write_report(db_path, generated_at, model_names, baseline, threshold,
                                 domains, columns, records, profiles_out)
        print(f"\nWrote debug report: {out_path}")
        return

    df = df.dropna(subset=["answer_correct"]).reset_index(drop=True)

    total_rows  = len(df)
    n_failures  = int((df["answer_correct"] == 0).sum())
    n_successes = int((df["answer_correct"] == 1).sum())
    print(f"  Rows loaded      : {total_rows}")
    print(f"  Failures (0)     : {n_failures}  ({n_failures / total_rows * 100:.1f}%)")
    print(f"  Successes (1)    : {n_successes}  ({n_successes / total_rows * 100:.1f}%)")

    print("\nBuilding features...")
    # Parse active_domains JSON strings and multi-label binarize to one column per domain
    df["active_domains"] = df["active_domains"].apply(
        lambda v: json.loads(v) if isinstance(v, str) else (v or [])
    )
    mlb = MultiLabelBinarizer()
    domain_dummies = pd.DataFrame(
        mlb.fit_transform(df["active_domains"]),
        columns=[f"domain_{c}" for c in mlb.classes_],
        index=df.index,
    )

    # Tree features are the explicit numeric set + domain dummies (not the text columns).
    X = pd.concat([df[FEATURE_COLUMNS], domain_dummies], axis=1)
    y = df["answer_correct"]  # 0 = fail, 1 = pass

    # NULL in slot/group counts means the domain doesn't apply → fill with 0
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    feature_count = X_imputed.shape[1]
    print(f"  Feature columns  : {feature_count}  ({', '.join(f'domain_{c}' for c in mlb.classes_)} + numeric)")

    print("\nTraining decision tree (max_depth=4)...")
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    tree.fit(X_imputed, y)
    print(f"  Tree depth used  : {tree.get_depth()}  |  Leaves: {tree.get_n_leaves()}")

    print("\nExtracting failure profiles...")
    # min_failure_rate above baseline so we only flag pockets meaningfully worse than average
    # min_samples=3 to surface small-but-bad pockets that min_samples_leaf=5 may not split further
    baseline = n_failures / total_rows
    threshold = max(0.65, baseline + 0.10)
    failure_profiles = get_failure_rules(tree, X_imputed.columns, min_failure_rate=threshold, min_samples=3)
    print(f"  Baseline failure rate : {baseline * 100:.1f}%  |  Profile threshold : {threshold * 100:.1f}%")

    # Map every row to its leaf so we can recover the exact pocket behind each profile.
    leaf_of_row = tree.apply(X_imputed)

    print(f"\n[SURGICAL DATA FIX TARGETS FOR {label}]")
    print("=" * 80)
    if not failure_profiles:
        print("No significant failure pockets detected! Your model is scaling smoothly.")
    else:
        for idx, (rule, rate, samples, leaf_id) in enumerate(failure_profiles, 1):
            print(f"Profile #{idx}:")
            print(f"  Trap Metric : {rate * 100:.1f}% failure rate over {samples} validation runs")
            print(f"  Condition   : {' AND '.join(rule)}")
            print("-" * 80)

            row_ids = [int(i) for i in df.loc[leaf_of_row == leaf_id, "id"].tolist()]
            profiles_out.append({
                "index": idx,
                "failure_rate": rate,
                "sample_count": samples,
                "condition_text": " AND ".join(rule),
                "condition": _parse_condition(rule),
                "row_ids": row_ids,
            })

    out_path = _write_report(db_path, generated_at, model_names, baseline, threshold,
                             domains, columns, records, profiles_out)
    print(f"\nWrote debug report: {out_path}")


if __name__ == "__main__":
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sft_test.db")
    analyze_and_export_failure_targets(DB_PATH, MODEL_NAMES)
