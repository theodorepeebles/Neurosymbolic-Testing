import json
import os
import sqlite3

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier, _tree

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Which model(s) to analyze, by extraction_model_name as logged in the DB.
#   None         -> all models (pooled together)
#   ["a"]        -> just model "a"
#   ["a", "b"]   -> models "a" and "b" pooled together
MODEL_NAMES = ["SFT_Extraction_Qwen3_0.6b-v2"]


def get_failure_rules(tree, feature_names, min_failure_rate=0.60, min_samples=5):
    """
    Walks the decision tree leaf nodes and collects paths that heavily lead to
    failures (Class 0), ignoring successful or low-sample branches.
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
                rules.append((current_rule, failure_rate, total))

    recurse(0, [])
    return sorted(rules, key=lambda x: x[1], reverse=True)


def analyze_and_export_failure_targets(db_path: str, model_names=None):
    """
    Queries the SQLite data pipeline, builds the programmatic tree,
    and isolates independent failure profiles.

    `model_names` filters by extraction_model_name: None = all models, or a list
    of names to pool together.
    """
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
    query = f"""
        SELECT
            active_domains,
            text_word_count,
            text_lexical_density,
            expected_entity_count,
            expected_slot_count,
            expected_group_count,
            expected_global_constraint_count,
            expected_logical_wrapper_count,
            expected_question_constraint_count,
            expected_choice_count,
            expected_choice_constraint_count,
            expected_before_count,
            expected_immediately_before_count,
            expected_adjacent_count,
            expected_slot_fixed_count,
            expected_is_truth_teller_count,
            expected_is_deceiver_count,
            expected_same_group_count,
            expected_different_group_count,
            expected_exactly_n_count,
            expected_is_in_count,
            expected_if_then_count,
            expected_not_count,
            expected_and_count,
            expected_or_count,
            answer_correct
        FROM extraction_attempts
        WHERE {where}
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty or df["answer_correct"].dropna().empty:
        print(f"Insufficient data to extract patterns for {label}.")
        return

    df = df.dropna(subset=["answer_correct"])

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

    X_raw = df.drop(columns=["answer_correct", "active_domains"])
    X = pd.concat([X_raw, domain_dummies], axis=1)
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
    failure_profiles = get_failure_rules(tree, X_imputed.columns, min_failure_rate=max(0.65, baseline + 0.10), min_samples=3)
    print(f"  Baseline failure rate : {baseline * 100:.1f}%  |  Profile threshold : {max(0.65, baseline + 0.10) * 100:.1f}%")

    print(f"\n[SURGICAL DATA FIX TARGETS FOR {label}]")
    print("=" * 80)
    if not failure_profiles:
        print("No significant failure pockets detected! Your model is scaling smoothly.")
        return

    for idx, (rule, rate, samples) in enumerate(failure_profiles, 1):
        print(f"Profile #{idx}:")
        print(f"  Trap Metric : {rate * 100:.1f}% failure rate over {samples} validation runs")
        print(f"  Condition   : {' AND '.join(rule)}")
        print("-" * 80)


if __name__ == "__main__":
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sft_test.db")
    analyze_and_export_failure_targets(DB_PATH, MODEL_NAMES)
