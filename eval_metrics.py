"""
Comparison metrics for evaluating a model's extraction against the gold extraction.

Pure functions over plain dicts (LogicProblem.model_dump() output), so they don't
depend on the pydantic schema objects. Used by run.py to populate the expected_* /
extracted_* / exact_*_match columns of sft_test.db.

Exact matches are order-agnostic *and* key-order-agnostic: each constraint dict is
frozen with json.dumps(..., sort_keys=True) (sort_keys recurses into nested wrappers)
and compared as a set, so ["A","B"] == ["B","A"] and {"l":1,"r":2} == {"r":2,"l":1}.
"""

import json

WRAPPER_TYPES = {"if_then", "not", "and", "or"}

# Every constraint type defined in validators.py, each broken out into its own
# expected_/extracted_ count column. Keep in sync with validators.py
# (DOMAIN_CONSTRAINT_CLASSES leaf types + the wrappers built in build_hybrid_schema).
# Each contributes an `expected_<type>_count` and `extracted_<type>_count` metric.
COUNTED_CONSTRAINT_TYPES = (
    # ordering
    "before", "immediately_before", "adjacent", "slot_fixed",
    # knights_and_knaves
    "is_truth_teller", "is_deceiver",
    # grouping
    "same_group", "different_group", "exactly_n", "is_in",
    # logical wrappers
    "if_then", "not", "and", "or",
)


def _strip_evidence(constraint):
    """Return a deep copy of a constraint dict with every `evidence_text` removed,
    recursing into logical wrappers. Attribution adds evidence_text to extracted (and,
    eventually, gold) constraints; it's source-sentence provenance, not semantics, so it
    must NOT affect exact_*_match comparisons (two equal constraints can cite different
    sentences, and old gold rows carry none)."""
    if not isinstance(constraint, dict):
        return constraint
    out = {k: v for k, v in constraint.items() if k != "evidence_text"}
    for key in ("antecedent", "consequent", "claim"):
        if isinstance(out.get(key), dict):
            out[key] = _strip_evidence(out[key])
    if isinstance(out.get("claims"), list):
        out["claims"] = [_strip_evidence(c) for c in out["claims"]]
    return out


def get_canonical_set(constraint_list) -> set:
    """Freeze a list of constraint dicts into an order-/key-agnostic set of strings.
    evidence_text is stripped first so provenance never affects exact-match equality."""
    return set(json.dumps(_strip_evidence(c), sort_keys=True) for c in (constraint_list or []))


def _count_wrappers(constraint) -> int:
    """Recursively count logical-wrapper constraints within one constraint dict."""
    if not isinstance(constraint, dict):
        return 0
    n = 1 if constraint.get("type") in WRAPPER_TYPES else 0
    for key in ("antecedent", "consequent", "claim"):
        if isinstance(constraint.get(key), dict):
            n += _count_wrappers(constraint[key])
    for sub in constraint.get("claims", []) or []:
        n += _count_wrappers(sub)
    return n


def _count_of_type(constraint, target_type: str) -> int:
    """Recursively count occurrences of `target_type` within one constraint dict,
    descending into logical wrappers (if_then / not / and / or) the same way
    _count_wrappers does, so nested constraints are included."""
    if not isinstance(constraint, dict):
        return 0
    n = 1 if constraint.get("type") == target_type else 0
    for key in ("antecedent", "consequent", "claim"):
        if isinstance(constraint.get(key), dict):
            n += _count_of_type(constraint[key], target_type)
    for sub in constraint.get("claims", []) or []:
        n += _count_of_type(sub, target_type)
    return n


def _type_count(p: dict, target_type: str) -> int:
    """Count occurrences of `target_type` across every constraint in the problem
    (global + question + choice), recursing into logical wrappers."""
    return sum(
        _count_of_type(c, target_type)
        for c in _global_constraints(p) + _question_constraints(p) + _choice_constraints(p)
    )


def _global_constraints(p: dict) -> list:
    return p.get("constraints", []) or []


def _question_constraints(p: dict) -> list:
    out = []
    for q in p.get("questions", []) or []:
        out.extend(q.get("question_constraints", []) or [])
    return out


def _choice_constraints(p: dict) -> list:
    out = []
    for q in p.get("questions", []) or []:
        for ch in q.get("answer_choices", []) or []:
            out.extend(ch.get("constraints", []) or [])
    return out


def _choice_count(p: dict) -> int:
    return sum(len(q.get("answer_choices", []) or []) for q in p.get("questions", []) or [])


def _wrapper_count(p: dict) -> int:
    return sum(
        _count_wrappers(c)
        for c in _global_constraints(p) + _question_constraints(p) + _choice_constraints(p)
    )


def text_metrics(problem_text: str):
    """(word_count, lexical_density) — unique lowercased words / total words."""
    words = (problem_text or "").split()
    wc = len(words)
    lex = len({w.lower() for w in words}) / wc if wc else 0.0
    return wc, lex


def build_comparison_metrics(expected: dict, extracted, active_domains: list[str],
                             problem_text: str) -> dict:
    """Return a dict of every expected_*/extracted_*/exact_*_match + text metric column.

    `expected` is the gold LogicProblem dict (always present). `extracted` is the model's
    LogicProblem dict, or None if the model failed to produce a valid extraction (in which
    case extracted_* counts are None and every exact_*_match is 0).
    """
    wc, lex = text_metrics(problem_text)
    has_ext = extracted is not None
    ext = extracted or {}

    def cmp(exp_list, ext_list) -> int:
        return int(has_ext and get_canonical_set(exp_list) == get_canonical_set(ext_list))

    exp_global, ext_global = _global_constraints(expected), _global_constraints(ext)
    exp_qc, ext_qc = _question_constraints(expected), _question_constraints(ext)
    exp_cc, ext_cc = _choice_constraints(expected), _choice_constraints(ext)
    exp_entities = expected.get("entities", []) or []
    ext_entities = ext.get("entities", []) or []

    ordering = "ordering" in active_domains
    grouping = "grouping" in active_domains

    metrics = {
        "text_word_count": wc,
        "text_lexical_density": lex,

        "expected_entity_count": len(exp_entities),
        "extracted_entity_count": len(ext_entities) if has_ext else None,
        "exact_entity_match": int(has_ext and set(exp_entities) == set(ext_entities)),

        "expected_slot_count": expected.get("num_slots") if ordering else None,
        "extracted_slot_count": ext.get("num_slots") if (has_ext and ordering) else None,
        "expected_group_count": expected.get("num_groups") if grouping else None,
        "extracted_group_count": ext.get("num_groups") if (has_ext and grouping) else None,

        "expected_global_constraint_count": len(exp_global),
        "extracted_global_constraint_count": len(ext_global) if has_ext else None,
        "exact_global_constraint_match": cmp(exp_global, ext_global),

        "expected_question_constraint_count": len(exp_qc),
        "extracted_question_constraint_count": len(ext_qc) if has_ext else None,
        "exact_question_constraint_match": cmp(exp_qc, ext_qc),

        "expected_choice_count": _choice_count(expected),
        "extracted_choice_count": _choice_count(ext) if has_ext else None,
        "expected_choice_constraint_count": len(exp_cc),
        "extracted_choice_constraint_count": len(ext_cc) if has_ext else None,
        "exact_choice_constraint_match": cmp(exp_cc, ext_cc),

        "expected_logical_wrapper_count": _wrapper_count(expected),
        "extracted_logical_wrapper_count": _wrapper_count(ext) if has_ext else None,
    }

    # Per-type constraint counts (expected + extracted) across the whole problem.
    for t in COUNTED_CONSTRAINT_TYPES:
        metrics[f"expected_{t}_count"]  = _type_count(expected, t)
        metrics[f"extracted_{t}_count"] = _type_count(ext, t) if has_ext else None

    return metrics
