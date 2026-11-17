"""
Build the full trace the replay page animates, for one puzzle.

build_trace(problem_text, domains, extraction) takes what the two models produced (the
classifier's domains and the extractor's JSON) and does everything after that with the
existing pipeline code, no model calls:

    extraction -> rules + source spans     attribution.build_attribution
    solving    -> one solution, each rule  explanation.build_context + Z3
                  checked against it
    choices    -> verdict per choice       pipeline.z3_solve
                  eliminated: MUS or       explanation.explain_question
                  counterexample
                  confirmed: MUS or        the same explanation.py functions
                  witness

It returns one JSON-safe dict. The page gets the whole dict up front and does all the pacing.
The same function serves live mode later: pass fresh model output and source="live".

Gold data is optional and used for three things only: the correct domains, the correct
answer, and (when the pipeline's answer is wrong) where the extraction differs from gold.
The rules, solution and verdicts always come from the model's extraction.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))

from z3 import Bool, Not, Solver, get_version_string, is_true, sat, unsat

import explanation
from attribution import build_attribution
from pipeline import format_constraint, z3_solve
from validators import DOMAIN_CONSTRAINT_CLASSES, build_hybrid_schema

TRACE_VERSION = 1
WRAPPER_TYPES = ["if_then", "not", "and", "or"]
ENGINE = f"Z3 {get_version_string()}"


# ── small helpers ────────────────────────────────────────────────────────────

def _leaf_types(domains: list[str]) -> list[str]:
    """Constraint types the extraction schema allows for these domains (validators.py)."""
    types = []
    for d in domains:
        for cls in DOMAIN_CONSTRAINT_CLASSES[d]:
            types.extend(cls.model_fields["type"].annotation.__args__)
    return types


TYPE_DOMAIN = {t: d for d in DOMAIN_CONSTRAINT_CLASSES for t in _leaf_types([d])}


def _utf16_span(text: str, span) -> list[int] | None:
    """attribution spans index Python characters; the browser indexes UTF-16 units."""
    if span is None:
        return None
    return [len(text[:i].encode("utf-16-le")) // 2 for i in span]


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _var_info(name: str) -> tuple[str, str]:
    kind, entity = name.split("_", 1)   # slot_/group_/kk_ prefixes from pipeline.encode
    return kind, entity


def _types_in(c: dict) -> set[str]:
    """Every constraint type used in one constraint dict, wrappers included."""
    out = {c.get("type")}
    for key in ("antecedent", "consequent", "claim"):
        if isinstance(c.get(key), dict):
            out |= _types_in(c[key])
    for child in c.get("claims") or []:
        out |= _types_in(child)
    return out


def _canonical(c: dict) -> str:
    """A constraint dict as a string that is equal for logically identical constraints:
    evidence_text dropped, and order ignored where it carries no meaning."""
    def norm(x):
        if not isinstance(x, dict):
            return x
        out = {k: norm(v) for k, v in x.items() if k != "evidence_text"}
        t = out.get("type")
        if t in ("same_group", "different_group", "exactly_n"):
            out["entities"] = sorted(out.get("entities") or [])
        elif t == "adjacent":
            out["left"], out["right"] = sorted([out["left"], out["right"]])
        elif t in ("and", "or"):
            out["claims"] = sorted((norm(cl) for cl in x.get("claims") or []),
                                   key=lambda d: json.dumps(d, sort_keys=True))
        return out
    return json.dumps(norm(c), sort_keys=True)


def _evidence_key(c: dict) -> str:
    return (c.get("evidence_text") or "").lower().strip(".,; ")


# ── stage builders ───────────────────────────────────────────────────────────

def _classification_stage(domains, gold_domains, meta) -> dict:
    stage = {
        "predicted_domains": list(domains),
        "all_domains": list(DOMAIN_CONSTRAINT_CLASSES),
        "gold_domains": None, "correct": None, "extra": [], "missing": [],
        "meta": meta,
    }
    if gold_domains is not None:
        stage["gold_domains"] = list(gold_domains)
        stage["extra"] = [d for d in domains if d not in gold_domains]
        stage["missing"] = [d for d in gold_domains if d not in domains]
        stage["correct"] = set(domains) == set(gold_domains)
    return stage


def _rule(cid: str, c, kind: str, methods: dict, spans: dict, text: str) -> dict:
    return {
        "id": cid,
        "kind": kind,                       # "clue" | "given" | "choice"
        "label": format_constraint(c),
        "span": _utf16_span(text, spans.get(cid)),
        "method": methods.get(cid, "unattributed"),   # option_a | bm25_fallback | unattributed
    }


def _extraction_stage(lp, q_index, domains, methods, spans, text, raw, meta) -> dict:
    q = lp.questions[q_index]
    rules = [_rule(f"c_{i}", c, "clue", methods, spans, text)
             for i, c in enumerate(lp.constraints)]
    rules += [_rule(f"q{q_index}.qc_{i}", c, "given", methods, spans, text)
              for i, c in enumerate(q.question_constraints)]
    choices = []
    for ch in q.answer_choices:
        choices.append({
            "label": ch.label,
            "type": ch.type,
            "text": explanation.proposition_text(ch),
            "rules": [_rule(f"q{q_index}.{ch.label}_{j}", c, "choice", methods, spans, text)
                      for j, c in enumerate(ch.constraints)],
        })
    return {
        "allowed_types": _leaf_types(domains),
        "wrapper_types": WRAPPER_TYPES,
        "entities": list(lp.entities),
        "num_slots": getattr(lp, "num_slots", None),
        "num_groups": getattr(lp, "num_groups", None),
        "rules": rules,
        "choices": choices,
        "raw": raw,
        "meta": meta,
    }


def _solving_stage(ctx, named, q_index, span_index) -> dict:
    """One solution of the rules (clues + the question's given + Z3's built-in rules), and each
    rule evaluated in it. If the rules contradict each other, the unsat core instead."""
    builtin = [{"id": cid, "label": ctx.cid_label[cid].replace(" [structural]", "")}
               for cid, _ in named if cid.startswith("implicit.")]
    t0 = time.perf_counter()
    solver = Solver()
    trackers = {}
    for cid, expr in named:
        tracker = Bool(f"replay_trk_{cid}")
        trackers[str(tracker)] = cid
        solver.assert_and_track(expr, tracker)
    status = solver.check()
    elapsed = time.perf_counter() - t0

    stage = {"status": str(status), "builtin_rules": builtin, "variables": [], "checks": [],
             "unsat_core": [], "meta": {"engine": ENGINE, "time_ms": _ms(elapsed)}}
    if status == unsat:
        stage["unsat_core"] = [trackers[str(b)] for b in solver.unsat_core()
                               if str(b) in trackers]
        return stage
    if status != sat:
        return stage

    model = solver.model()
    forced, _free, _order = explanation.annotate_puzzle_solution(ctx, q_index, span_index)
    forced_by = {var: cids for var, _val, cids in forced}
    for name, zv in ctx.zvars.items():
        kind, entity = _var_info(name)
        stage["variables"].append({
            "var": name, "kind": kind, "entity": entity,
            "value": explanation._py_val(model, zv),
            "forced": name in forced_by,          # same value in every solution
            "forced_by": forced_by.get(name, []),
        })
    stage["checks"] = [{"id": cid,
                        "satisfied": bool(is_true(model.eval(expr, model_completion=True)))}
                       for cid, expr in named]
    return stage


def _mus_proof(all_mus, chain) -> dict:
    return {
        "kind": "mus",
        "rules": [step.constraint_ids[0] for step in chain] or list(all_mus[0]),
        "alternatives": [list(m) for m in all_mus[1:]],
    }


def _witness_proof(model: dict) -> dict:
    return {"kind": "witness", "model": model}


def _prove_confirmed(choice, ctx, named, span_index) -> dict:
    """Why a verified choice holds, from the same explanation.py machinery that explains the
    eliminated ones. Verified on SAT (could_be_*): a witness solution. Verified on UNSAT
    (must_be_*): the minimal rule set that makes the asserted opposite impossible."""
    hyp = explanation.choice_hypothesis(choice, ctx.zvars)
    asserted = Not(hyp) if explanation.NEGATE[choice.type] else hyp
    if explanation.VERIFIED_ON_SAT[choice.type]:
        return _witness_proof(explanation.build_counterexample_model(asserted, named, ctx.zvars))
    text_positions = {cid: (span_index[cid][0] if cid in span_index else 9999)
                      for cid, _ in named}
    muses, _msses = explanation.marco_enumerate(named, asserted)
    all_mus = explanation.rank_muses(explanation._dedup(muses), span_index, text_positions)
    chain = explanation.build_narrative_chain(all_mus[0], explanation.choice_vars(choice),
                                              ctx.cid_vars, span_index, text_positions)
    return _mus_proof(all_mus, chain)


def _choices_stage(lp, q_index, ctx, named, span_index, text) -> dict:
    q = lp.questions[q_index]
    t0 = time.perf_counter()
    # The verdicts are the pipeline's own: the same z3_solve call the eval run scored.
    z3_result = z3_solve(lp)
    verdicts = z3_result["question_results"][q_index]
    struct = explanation.explain_question(lp, q_index, ctx, span_index, text)
    wrong = {w.answer: w for w in struct.wrong}
    if {label for label, ok in verdicts.items() if not ok} != set(wrong):
        raise RuntimeError("pipeline.z3_solve and explanation.py disagree on the verdicts")

    results = []
    for ch in q.answer_choices:
        negated = explanation.NEGATE[ch.type]
        confirmed = bool(verdicts[ch.label])
        if confirmed:
            proof = _prove_confirmed(ch, ctx, named, span_index)
        else:
            w = wrong[ch.label]
            if w.query_type == "MUS_REFUTATION":
                proof = _mus_proof(w.all_mus, w.narrative_chain)
            else:
                proof = _witness_proof(w.counterexample_model)
        results.append({
            "label": ch.label,
            "type": ch.type,
            "asserted": "opposite" if negated else "claim",   # what Z3 was asked to assume
            "z3": "unsat" if proof["kind"] == "mus" else "sat",
            "verdict": "confirmed" if confirmed else "eliminated",
            "focus_vars": sorted(explanation.choice_vars(ch)),
            "proof": proof,
        })
    elapsed = time.perf_counter() - t0

    verified = [label for label, ok in verdicts.items() if ok]
    return {
        "results": results,
        "verified": verified,
        "answer": ",".join(verified) or None,    # same rule as run.py
        "outcome": {0: "none", 1: "single"}.get(len(verified), "multiple"),
        "gold_answer": None, "correct": None,
        "meta": {"engine": ENGINE, "time_ms": _ms(elapsed)},
    }


# ── gold: correct answer, and the diff for wrong answers ─────────────────────

def _gold_answer(gold_lp, q_index) -> str | None:
    """The single choice Z3 verifies on the gold extraction (run.py's derive_ground_truth)."""
    res = z3_solve(gold_lp)
    if res.get("status") != "sat" or not res.get("question_results"):
        return None
    verified = [label for label, ok in res["question_results"][q_index].items() if ok]
    return verified[0] if len(verified) == 1 else None


def _diff_rules(where, model_pairs, gold_pairs, gold_domains) -> tuple[list, list]:
    """Compare two lists of (id, dict, label). Returns (changed, missing): model rules with no
    identical gold rule (paired with the gold rule quoting the same phrase, if any), and gold
    rules the model has no counterpart for."""
    gold_left = list(gold_pairs)
    unmatched = []
    for cid, c, label in model_pairs:
        hit = next((g for g in gold_left if _canonical(g[1]) == _canonical(c)), None)
        if hit:
            gold_left.remove(hit)
        else:
            unmatched.append((cid, c, label))
    changed = []
    for cid, c, label in unmatched:
        partner = next((g for g in gold_left
                        if _evidence_key(g[1]) and _evidence_key(g[1]) == _evidence_key(c)), None)
        if partner:
            gold_left.remove(partner)
        extra_domains = sorted({TYPE_DOMAIN[t] for t in _types_in(c)
                                if t in TYPE_DOMAIN and TYPE_DOMAIN[t] not in gold_domains})
        changed.append({
            "id": cid, "where": where, "model": label,
            "gold": partner[2] if partner else None,
            # the rule uses a type that only exists in a domain this puzzle doesn't have
            "types_from_extra_domains": extra_domains,
        })
    missing = [{"where": where, "gold": label} for _cid, _c, label in gold_left]
    return changed, missing


def _gold_diff(lp, gold_lp, q_index, gold_domains) -> dict:
    def pairs(prefix, objs):
        return [(f"{prefix}{i}", c.model_dump(), format_constraint(c)) for i, c in enumerate(objs)]

    mq, gq = lp.questions[q_index], gold_lp.questions[q_index]
    changed, missing = _diff_rules("clue", pairs("c_", lp.constraints),
                                   pairs("c_", gold_lp.constraints), gold_domains)
    c2, m2 = _diff_rules("given", pairs(f"q{q_index}.qc_", mq.question_constraints),
                         pairs(f"q{q_index}.qc_", gq.question_constraints), gold_domains)
    changed += c2
    missing += m2

    gold_choices = {ch.label: ch for ch in gq.answer_choices}
    choice_types = []
    for ch in mq.answer_choices:
        g = gold_choices.pop(ch.label, None)
        if g is None:
            missing_gold = []
        else:
            if g.type != ch.type:
                choice_types.append({"label": ch.label, "model": ch.type, "gold": g.type})
            missing_gold = g.constraints
        c3, m3 = _diff_rules(f"choice {ch.label}",
                             pairs(f"q{q_index}.{ch.label}_", ch.constraints),
                             pairs(f"q{q_index}.{ch.label}_", missing_gold), gold_domains)
        changed += c3
        missing += m3

    fields = []
    for name in ("num_slots", "num_groups"):
        mv, gv = getattr(lp, name, None), getattr(gold_lp, name, None)
        if mv != gv:
            fields.append({"name": name, "model": mv, "gold": gv})
    if set(lp.entities) != set(gold_lp.entities):
        fields.append({"name": "entities", "model": list(lp.entities),
                       "gold": list(gold_lp.entities)})
    return {"changed": changed, "missing": missing, "choice_types": choice_types,
            "fields": fields}


# ── the one entry point ──────────────────────────────────────────────────────

def build_trace(problem_text: str, domains: list[str], extraction, *, source: str = "replay",
                problem_id: str | None = None, model_meta: dict | None = None,
                gold_domains: list[str] | None = None, gold_extraction: dict | None = None,
                question_index: int = 0) -> dict:
    """Everything the page shows for one puzzle, from the models' output onward.

    problem_text    the puzzle as given to the models
    domains         the classifier's predicted domains, in the order it gave them
    extraction      the extractor's output: the raw JSON string, or the parsed dict
    source          "replay" (saved model output) or "live" (fresh model output)
    model_meta      optional real metadata: {"classifier": {...}, "extraction": {...}}, each
                    with model / in_tok / out_tok / time_ms where known
    gold_*          optional; only for the correct domains and answer, and the diff shown
                    when the pipeline's answer is wrong
    """
    model_meta = model_meta or {}
    if isinstance(extraction, str):
        raw, extracted_dict = extraction, json.loads(extraction)
    else:
        raw, extracted_dict = json.dumps(extraction), extraction

    lp = build_hybrid_schema(domains)(**extracted_dict)
    methods, spans = build_attribution(lp.model_dump(), problem_text)
    span_index = {cid: tuple(v) for cid, v in spans.items()}
    ctx = explanation.build_context(lp)
    named = ctx.base_named + ctx.question_named.get(question_index, [])

    solving = _solving_stage(ctx, named, question_index, span_index)
    if solving["status"] == "sat":
        choices = _choices_stage(lp, question_index, ctx, named, span_index, problem_text)
    else:   # contradictory rules: no choice can be tested
        choices = {"results": [], "verified": [], "answer": None, "outcome": "unsolved",
                   "gold_answer": None, "correct": None,
                   "meta": {"engine": ENGINE, "time_ms": 0.0}}

    gold_diff = None
    if gold_extraction is not None and gold_domains is not None:
        gold_lp = build_hybrid_schema(gold_domains)(**gold_extraction)
        gold = _gold_answer(gold_lp, question_index)
        choices["gold_answer"] = gold
        choices["correct"] = gold is not None and choices["verified"] == [gold]
        if not choices["correct"]:
            gold_diff = _gold_diff(lp, gold_lp, question_index, gold_domains)

    return {
        "version": TRACE_VERSION,
        "source": source,
        "problem_id": problem_id,
        "problem_text": problem_text,
        "question_index": question_index,
        "stages": {
            "classification": _classification_stage(domains, gold_domains,
                                                    model_meta.get("classifier")),
            "extraction": _extraction_stage(lp, question_index, domains, methods, spans,
                                            problem_text, raw, model_meta.get("extraction")),
            "solving": solving,
            "choices": choices,
        },
        "gold_diff": gold_diff,
    }
