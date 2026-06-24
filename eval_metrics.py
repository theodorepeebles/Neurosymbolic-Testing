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


def get_canonical_set(constraint_list) -> set:
    """Freeze a list of constraint dicts into an order-/key-agnostic set of strings."""
    return set(json.dumps(c, sort_keys=True) for c in (constraint_list or []))


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

    return {
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
