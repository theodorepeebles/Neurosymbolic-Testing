"""
Symbolic explanation engine (Stage 1) for the neurosymbolic logic-puzzle pipeline.

Given an extracted LogicProblem (the same object pipeline.z3_solve consumes), this produces,
for every multiple-choice answer, formally-correct facts about *why* it is right or wrong:

  - wrong answers (MUS_REFUTATION): the minimal unsatisfiable clue sets (MUSes) the answer
    violates, a topologically-ordered narrative chain for the preferred MUS, the minimal
    correction subset (MCS) / largest satisfiable subset (MSS), and a confidence tier;
  - wrong answers (COUNTEREXAMPLE): a satisfying witness model showing the claim need not hold;
  - the correct answer: a full forced/free variable-binding annotation of the puzzle's solution.

This is the "hard boundary": everything here is Z3-derived and formally correct. A later LLM
stage will narrate around the ExplanationStruct without inventing reasoning.

Answer choices in this system are PARTIAL propositions (choice.constraints + a question type),
not full variable assignments. The proposition is asserted as the hard hypothesis (negated per
question type, see NEGATE/VERIFIED_ON_SAT) and MUSes are found over the base + question +
implicit (structural) constraints. The engine builds its own parallel, fully-named constraint
set so implicit rules (all_different / valid ranges) can appear in a MUS — pipeline.z3_solve
adds those *untracked*, so it can't.

Standalone: pipeline.py / run.py are untouched. To exercise the engine without a database:

    python explanation.py        # runs the built-in smoke test over prompts.EXAMPLE_JSONS

See explanation_debug.py to run it on a real run_id from sft_test.db.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Optional
from collections import defaultdict

from z3 import (
    Solver, sat, unsat, Bool, BoolVal, Int, IntVal,
    Implies, And, Or, Not, Distinct, is_true, is_false,
)

from pipeline import (
    encode, format_constraint, _used_types,
    ORDERING_TYPES, GROUPING_TYPES, KK_TYPES,
)
from attribution import build_attribution, constraint_to_query


# ─────────────────────────────────────────────────────────────────────────────
# Data structures (the ExplanationStruct family)
# ─────────────────────────────────────────────────────────────────────────────

class ExplanationTier(Enum):
    T1_SINGLE_ATTRIBUTED = "single_attributed"   # one constraint, span verified
    T2_MULTI_ATTRIBUTED  = "multi_attributed"    # 2-3 constraints, all attributed
    T3_CHAIN_PARTIAL     = "chain_partial"       # longer chain
    T4_UNATTRIBUTED      = "unattributed"        # at least one clue has no span
    T_STRUCTURAL         = "structural"          # only implicit domain rules


@dataclass
class ConstraintStep:
    """One step in a narrative chain. depth 0 = all vars hypothesis-known; depth N needs a
    depth-(N-1) step to fire first. is_contradiction True only for the final (failing) step."""
    constraint_ids: list
    variables_involved: list
    depth: int
    is_contradiction: bool
    evidence_spans: Optional[list]   # [[start, end], ...] in problem_text, or None


@dataclass
class WrongAnswerExplanation:
    answer: str
    query_type: str                  # "MUS_REFUTATION" | "COUNTEREXAMPLE"
    question_type: str               # could_be_true | must_be_true | must_be_false | could_be_false
    hypothesis_type: str             # always "proposition" in this system
    proposition_text: str            # human-readable answer choice

    # MUS_REFUTATION fields
    single_refutations: list         # list[ConstraintStep] (each a cardinality-1 MUS)
    all_mus: list                    # list[list[str]], preferred primary first
    narrative_chain: list            # list[ConstraintStep] for all_mus[0]
    mcs: list                        # minimal correction subset (clues to drop)
    mss: list                        # largest satisfiable subset

    tier: ExplanationTier

    # COUNTEREXAMPLE field
    counterexample_model: Optional[dict]


@dataclass
class CorrectAnswerExplanation:
    answer: str
    forced_bindings: list            # list[(var, value, [forcing_cids])]
    free_bindings: list              # list[(var, value)]
    narrative_order: list            # forcing cids in text-position order


@dataclass
class ExplanationStruct:
    question_index: int
    correct: list                    # list[CorrectAnswerExplanation] (usually 1)
    wrong: list                      # list[WrongAnswerExplanation]
    constraint_span_index: dict      # cid -> (start, end)
    problem_text: str


# ─────────────────────────────────────────────────────────────────────────────
# Question-type encoding (mirrors pipeline.z3_solve / outline §2.13)
# ─────────────────────────────────────────────────────────────────────────────

# Assert the proposition negated? (must_be_true / could_be_false assert the negation)
NEGATE = {
    "could_be_true":  False,
    "must_be_true":   True,
    "must_be_false":  False,
    "could_be_false": True,
}

# Is the choice verified (the correct answer) when the asserted hypothesis is SAT?
# could_be_true / could_be_false verify on SAT; must_be_true / must_be_false verify on UNSAT.
VERIFIED_ON_SAT = {
    "could_be_true":  True,
    "must_be_true":   False,
    "must_be_false":  False,
    "could_be_false": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Constraint -> Z3 variable names (mirrors the prefixes used in pipeline.encode)
# ─────────────────────────────────────────────────────────────────────────────

def constraint_variables(c) -> set:
    """Prefixed Z3 var names a constraint touches: slot_/group_/kk_. Recurses into wrappers."""
    t = c.type
    if t in ("before", "immediately_before", "adjacent"):
        return {f"slot_{c.left}", f"slot_{c.right}"}
    if t == "slot_fixed":
        return {f"slot_{c.entity}"}
    if t in ("is_truth_teller", "is_deceiver"):
        return {f"kk_{c.entity}"}
    if t in ("same_group", "different_group", "exactly_n"):
        return {f"group_{e}" for e in c.entities}
    if t == "is_in":
        return {f"group_{c.entity}"}
    if t == "not":
        return constraint_variables(c.claim)
    if t == "if_then":
        return constraint_variables(c.antecedent) | constraint_variables(c.consequent)
    if t in ("and", "or"):
        out = set()
        for cl in c.claims:
            out |= constraint_variables(cl)
        return out
    return set()


# ─────────────────────────────────────────────────────────────────────────────
# Solver context: vars + fully-named constraint set (incl. implicit structural rules)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverContext:
    zvars: dict                      # name -> Z3 var
    base_named: list                 # [(cid, expr)] implicit + global
    question_named: dict             # q_index -> [(cid, expr)]
    cid_vars: dict                   # cid -> set[str]
    cid_label: dict                  # cid -> human-readable label


def build_context(extracted) -> SolverContext:
    """Build the Z3 vars and the parallel named-constraint set the explanation engine reasons
    over. Implicit structural rules become real, named (implicit.*) constraints so a MUS can
    cite them — unlike pipeline.z3_solve which adds them untracked."""
    all_types = _used_types(extracted)
    ents = list(extracted.entities)
    zvars: dict = {}
    base_named: list = []
    cid_vars: dict = {}
    cid_label: dict = {}

    if all_types & ORDERING_TYPES:
        for e in ents:
            zvars[f"slot_{e}"] = Int(f"slot_{e}")
        slot_names = {f"slot_{e}" for e in ents}

        cid = "implicit.all_different"
        base_named.append((cid, Distinct([zvars[f"slot_{e}"] for e in ents])))
        cid_vars[cid] = set(slot_names)
        cid_label[cid] = "all entities occupy distinct slots [structural]"

        cid = "implicit.valid_range"
        rng = And(*[And(zvars[f"slot_{e}"] >= 1, zvars[f"slot_{e}"] <= extracted.num_slots)
                    for e in ents])
        base_named.append((cid, rng))
        cid_vars[cid] = set(slot_names)
        cid_label[cid] = f"each slot is between 1 and {extracted.num_slots} [structural]"

    if all_types & GROUPING_TYPES:
        for e in ents:
            zvars[f"group_{e}"] = Int(f"group_{e}")
        cid = "implicit.group_range"
        rng = And(*[And(zvars[f"group_{e}"] >= 1, zvars[f"group_{e}"] <= extracted.num_groups)
                    for e in ents])
        base_named.append((cid, rng))
        cid_vars[cid] = {f"group_{e}" for e in ents}
        cid_label[cid] = f"each entity is in group 1..{extracted.num_groups} [structural]"

    if all_types & KK_TYPES:
        for e in ents:
            zvars[f"kk_{e}"] = Bool(f"kk_{e}")

    # Global constraints -> c_{i} (lines up 1:1 with pipeline's track_c_{i} and attribution cids)
    for i, c in enumerate(extracted.constraints):
        cid = f"c_{i}"
        base_named.append((cid, encode(c, zvars)))
        cid_vars[cid] = constraint_variables(c)
        cid_label[cid] = format_constraint(c)

    # Question constraints -> q{n}.qc_{i}
    question_named: dict = {}
    for n, q in enumerate(extracted.questions):
        qn = []
        for i, c in enumerate(q.question_constraints):
            cid = f"q{n}.qc_{i}"
            qn.append((cid, encode(c, zvars)))
            cid_vars[cid] = constraint_variables(c)
            cid_label[cid] = format_constraint(c)
        question_named[n] = qn

    return SolverContext(zvars, base_named, question_named, cid_vars, cid_label)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis (answer-choice proposition) helpers
# ─────────────────────────────────────────────────────────────────────────────

def choice_hypothesis(choice, zvars):
    if not choice.constraints:
        return BoolVal(True)
    return And(*[encode(c, zvars) for c in choice.constraints])


def choice_vars(choice) -> set:
    out: set = set()
    for c in choice.constraints:
        out |= constraint_variables(c)
    return out


def proposition_text(choice) -> str:
    if not choice.constraints:
        return "(empty proposition)"
    return " and ".join(constraint_to_query(c.model_dump()) for c in choice.constraints)


def _check(named, hypothesis):
    s = Solver()
    s.add(hypothesis)
    for _, expr in named:
        s.add(expr)
    return s.check()


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 algorithms
# ─────────────────────────────────────────────────────────────────────────────

def find_all_single_refutations(named, hypothesis, span_index, cid_vars) -> list:
    """Every named constraint that, alone, is unsatisfiable with the hypothesis (depth-0)."""
    refs = []
    for cid, expr in named:
        s = Solver()
        s.add(hypothesis)
        s.add(expr)
        if s.check() == unsat:
            refs.append(ConstraintStep(
                constraint_ids=[cid],
                variables_involved=sorted(cid_vars.get(cid, set())),
                depth=0,
                is_contradiction=True,
                evidence_spans=[list(span_index[cid])] if cid in span_index else None,
            ))
    return refs


def marco_enumerate(named, hypothesis):
    """Enumerate every MUS and MSS of `named` w.r.t. the hard `hypothesis`, via the MARCO
    map-solver loop with one selector boolean per constraint. Returns (muses, msses) as lists
    of constraint-id lists. Works on cids (not Z3 objects) to avoid `in`-on-BoolRef pitfalls."""
    cids = [cid for cid, _ in named]
    sel = {cid: Bool(f"p_sel_{cid}") for cid in cids}
    sel_name_to_cid = {str(b): cid for cid, b in sel.items()}

    s = Solver()
    s.add(hypothesis)
    for cid, expr in named:
        s.add(Implies(sel[cid], expr))

    map_solver = Solver()
    muses, msses = [], []

    while map_solver.check() == sat:
        m = map_solver.model()
        seed = [cid for cid in cids if is_true(m.eval(sel[cid], model_completion=True))]

        if s.check([sel[c] for c in seed]) == sat:
            mss = _grow_to_mss(s, seed, cids, sel)
            msses.append(mss)
            mss_set = set(mss)
            comp = [sel[c] for c in cids if c not in mss_set]
            map_solver.add(Or(comp) if comp else BoolVal(False))
        else:
            core_cids = [sel_name_to_cid[str(b)] for b in s.unsat_core()
                         if str(b) in sel_name_to_cid]
            muses.append(core_cids)
            map_solver.add(Not(And([sel[c] for c in core_cids])))

    return muses, msses


def _grow_to_mss(s, seed, cids, sel) -> list:
    mss = list(seed)
    mss_set = set(seed)
    for c in cids:
        if c not in mss_set and s.check([sel[x] for x in mss] + [sel[c]]) == sat:
            mss.append(c)
            mss_set.add(c)
    return mss


def _dedup(muses) -> list:
    seen, out = set(), []
    for mus in muses:
        fs = frozenset(mus)
        if fs not in seen:
            seen.add(fs)
            out.append(mus)
    return out


def _text_pos(span_index, cid) -> int:
    sp = span_index.get(cid)
    return sp[0] if sp else 9999


def rank_muses(muses, span_index, text_positions) -> list:
    """Order MUSes so all_mus[0] is the preferred primary: fewest unattributed clues first,
    then earliest in the text, then smallest cardinality."""
    def key(mus):
        unattr = sum(1 for c in mus
                     if c not in span_index and not c.startswith("implicit."))
        min_pos = min((text_positions.get(c, 9999) for c in mus), default=9999)
        return (unattr, min_pos, len(mus))
    return sorted(muses, key=key)


def compute_mcs(all_cids, msses) -> list:
    """Minimal correction subset: clues to remove for the answer to become valid (complement
    of the largest MSS)."""
    if not msses:
        return list(all_cids)
    largest = max(msses, key=len)
    keep = set(largest)
    return [c for c in all_cids if c not in keep]


def build_narrative_chain(mus, hypothesis_vars, cid_vars, span_index, text_positions) -> list:
    """Topologically order a MUS so it reads causally: a clue fires once all its variables are
    known (hypothesis-fixed, or derived by an earlier clue). Ties broken by text position; the
    min-free-vars fallback handles partial-proposition chains where nothing is fully known yet."""
    known = set(hypothesis_vars)
    remaining = {cid: set(cid_vars.get(cid, set())) - known for cid in mus}
    ordered = []
    depth = 0

    while remaining:
        ready = [cid for cid, free in remaining.items() if not free]
        if not ready:
            min_free = min(len(free) for free in remaining.values())
            ready = [cid for cid, free in remaining.items() if len(free) == min_free]
        ready.sort(key=lambda c: text_positions.get(c, 9999))

        for cid in ready:
            ordered.append((depth, cid))
            known |= cid_vars.get(cid, set())
            del remaining[cid]

        remaining = {cid: free - known for cid, free in remaining.items()}
        depth += 1

    steps = []
    for i, (d, cid) in enumerate(ordered):
        steps.append(ConstraintStep(
            constraint_ids=[cid],
            variables_involved=sorted(cid_vars.get(cid, set())),
            depth=d,
            is_contradiction=(i == len(ordered) - 1),
            evidence_spans=[list(span_index[cid])] if cid in span_index else None,
        ))
    return steps


def compute_tier(query_type, single_refs, all_mus, span_index) -> ExplanationTier:
    if query_type == "COUNTEREXAMPLE":
        return ExplanationTier.T1_SINGLE_ATTRIBUTED   # witness model is always complete

    primary = ([s.constraint_ids[0] for s in single_refs] if single_refs
               else (all_mus[0] if all_mus else []))

    structural   = [c for c in primary if c.startswith("implicit.")]
    non_struct   = [c for c in primary if not c.startswith("implicit.")]
    unattributed = [c for c in non_struct if c not in span_index]

    if unattributed:
        return ExplanationTier.T4_UNATTRIBUTED
    if non_struct == [] and structural:
        return ExplanationTier.T_STRUCTURAL
    if len(primary) == 1 or single_refs:
        return ExplanationTier.T1_SINGLE_ATTRIBUTED
    if len(primary) <= 3:
        return ExplanationTier.T2_MULTI_ATTRIBUTED
    return ExplanationTier.T3_CHAIN_PARTIAL


# ─────────────────────────────────────────────────────────────────────────────
# §2.7 correct-answer / full-solution annotation (forced vs free bindings)
# ─────────────────────────────────────────────────────────────────────────────

def _py_val(model, zv):
    r = model.eval(zv, model_completion=True)
    if is_true(r):
        return True
    if is_false(r):
        return False
    return r.as_long()


def _neq_lit(zv, val):
    return zv != (BoolVal(val) if isinstance(val, bool) else IntVal(val))


def annotate_puzzle_solution(ctx: SolverContext, q_index, span_index):
    """For the puzzle's satisfying model (base + question constraints), classify each variable as
    forced (same in every model — with the minimal forcing clue set) or free. Returns
    (forced_bindings, free_bindings, narrative_order)."""
    named = ctx.base_named + ctx.question_named.get(q_index, [])

    s = Solver()
    for _, expr in named:
        s.add(expr)
    if s.check() != sat:
        return [], [], []
    model = s.model()

    var_to_cids = defaultdict(list)
    for cid, _ in named:
        for v in ctx.cid_vars.get(cid, set()):
            var_to_cids[v].append(cid)

    forced, free = [], []
    for vname, zv in ctx.zvars.items():
        val = _py_val(model, zv)
        touching = var_to_cids.get(vname, [])

        if len(touching) < 2:
            if len(touching) == 1:
                forced.append((vname, val, list(touching)))
            else:
                free.append((vname, val))
            continue

        s2 = Solver()
        trackers = {}
        for cid, expr in named:
            tracker = Bool(f"trk_{cid}")
            trackers[str(tracker)] = cid
            s2.assert_and_track(expr, tracker)
        s2.add(_neq_lit(zv, val))

        if s2.check() == unsat:
            core = [trackers[str(b)] for b in s2.unsat_core() if str(b) in trackers]
            forced.append((vname, val, core))
        else:
            free.append((vname, val))

    forced.sort(key=lambda t: min((_text_pos(span_index, c) for c in t[2]), default=9999))
    all_force = [c for _, _, cids in forced for c in cids]
    narrative_order = sorted(set(all_force), key=lambda c: _text_pos(span_index, c))
    return forced, free, narrative_order


def build_counterexample_model(asserted, named, zvars) -> Optional[dict]:
    s = Solver()
    s.add(asserted)
    for _, expr in named:
        s.add(expr)
    if s.check() != sat:
        return None
    m = s.model()
    return {name: _py_val(m, zv) for name, zv in zvars.items()}


# ─────────────────────────────────────────────────────────────────────────────
# REPL / single-binding query entry points
#
# These answer interactive "var = value" questions ("Why can't X=V?", "Can X=V?", "What forces X?")
# by reusing the exact Stage-1 primitives above, and return the SAME WrongAnswerExplanation /
# CorrectAnswerExplanation objects the verbalization adapters already consume — so the REPL needs no
# separate prose path. question_type is the sentinel "query" (not a multiple-choice modality).
# ─────────────────────────────────────────────────────────────────────────────

def prepare_query_context(extracted, problem_text):
    """Build (ctx, span_index) for one puzzle — the same setup explain_problem does internally,
    factored out so repl.py can reuse it. Spans come from attribution so cids line up with labels."""
    _methods, spans = build_attribution(extracted.model_dump(), problem_text)
    span_index = {cid: tuple(v) for cid, v in spans.items()}
    ctx = build_context(extracted)
    return ctx, span_index


def _binding_hypothesis(ctx: SolverContext, var: str, val):
    """Z3 literal asserting var == val (bool or int), mirroring _neq_lit."""
    lit = BoolVal(val) if isinstance(val, bool) else IntVal(val)
    return ctx.zvars[var] == lit


def _refute_binding(ctx, q_index, span_index, var, val) -> WrongAnswerExplanation:
    """Core of the REPL var=value query. If asserting var==val is UNSAT against the puzzle, return
    a MUS_REFUTATION (why it's impossible); if SAT, return a COUNTEREXAMPLE witness (it's possible,
    here is an arrangement). Reuses the same algorithms as explain_question's wrong-answer branch."""
    named = ctx.base_named + ctx.question_named.get(q_index, [])
    all_cids = [cid for cid, _ in named]
    text_positions = {cid: _text_pos(span_index, cid) for cid in all_cids}

    H = _binding_hypothesis(ctx, var, val)
    prop = f"{var} = {val}"
    status = _check(named, H)

    if status == unsat:
        single_refs = find_all_single_refutations(named, H, span_index, ctx.cid_vars)
        muses, msses = marco_enumerate(named, H)
        all_mus = rank_muses(_dedup(muses), span_index, text_positions)
        chain = (build_narrative_chain(all_mus[0], {var}, ctx.cid_vars, span_index, text_positions)
                 if all_mus else [])
        return WrongAnswerExplanation(
            answer=var, query_type="MUS_REFUTATION", question_type="query",
            hypothesis_type="binding", proposition_text=prop,
            single_refutations=single_refs, all_mus=all_mus, narrative_chain=chain,
            mcs=compute_mcs(all_cids, msses), mss=max(msses, key=len) if msses else [],
            tier=compute_tier("MUS_REFUTATION", single_refs, all_mus, span_index),
            counterexample_model=None,
        )

    return WrongAnswerExplanation(
        answer=var, query_type="COUNTEREXAMPLE", question_type="query",
        hypothesis_type="binding", proposition_text=prop,
        single_refutations=[], all_mus=[], narrative_chain=[], mcs=[], mss=[],
        tier=compute_tier("COUNTEREXAMPLE", [], [], span_index),
        counterexample_model=build_counterexample_model(H, named, ctx.zvars),
    )


def query_why_not(ctx, q_index, span_index, var, val) -> WrongAnswerExplanation:
    """'Why can't X = V?' — expects UNSAT and returns the MUS refutation; if it's actually possible,
    returns a COUNTEREXAMPLE witness instead (informing the user that the binding can hold)."""
    return _refute_binding(ctx, q_index, span_index, var, val)


def query_can(ctx, q_index, span_index, var, val) -> WrongAnswerExplanation:
    """'Can X = V?' — SAT returns a COUNTEREXAMPLE witness (yes, here is an arrangement); UNSAT
    returns the MUS refutation (no, and here is why). Same machinery as query_why_not."""
    return _refute_binding(ctx, q_index, span_index, var, val)


def query_what_forces(ctx, q_index, span_index, var) -> Optional[CorrectAnswerExplanation]:
    """'What forces X?' — classify var via annotate_puzzle_solution. Returns a CorrectAnswerExplanation
    carrying just that variable's forced binding (with its minimal forcing clue set) or its free
    binding. Returns None if the puzzle is UNSAT or var is not in the solution."""
    forced, free, _order = annotate_puzzle_solution(ctx, q_index, span_index)
    for v, val, cids in forced:
        if v == var:
            return CorrectAnswerExplanation(answer=var, forced_bindings=[(var, val, cids)],
                                            free_bindings=[], narrative_order=list(cids))
    for v, val in free:
        if v == var:
            return CorrectAnswerExplanation(answer=var, forced_bindings=[],
                                            free_bindings=[(var, val)], narrative_order=[])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Top level
# ─────────────────────────────────────────────────────────────────────────────

def explain_question(extracted, q_index, ctx, span_index, problem_text) -> ExplanationStruct:
    q = extracted.questions[q_index]
    named = ctx.base_named + ctx.question_named.get(q_index, [])
    all_cids = [cid for cid, _ in named]
    text_positions = {cid: _text_pos(span_index, cid) for cid in all_cids}

    wrong, verified_labels = [], []

    for choice in q.answer_choices:
        H = choice_hypothesis(choice, ctx.zvars)
        asserted = Not(H) if NEGATE[choice.type] else H
        status = _check(named, asserted)
        verified = (status == sat) == VERIFIED_ON_SAT[choice.type]

        if verified:
            verified_labels.append(choice.label)
            continue

        prop = proposition_text(choice)

        if status == unsat:
            single_refs = find_all_single_refutations(named, asserted, span_index, ctx.cid_vars)
            muses, msses = marco_enumerate(named, asserted)
            all_mus = rank_muses(_dedup(muses), span_index, text_positions)
            chain = (build_narrative_chain(all_mus[0], choice_vars(choice),
                                           ctx.cid_vars, span_index, text_positions)
                     if all_mus else [])
            wrong.append(WrongAnswerExplanation(
                answer=choice.label, query_type="MUS_REFUTATION",
                question_type=choice.type, hypothesis_type="proposition",
                proposition_text=prop,
                single_refutations=single_refs, all_mus=all_mus, narrative_chain=chain,
                mcs=compute_mcs(all_cids, msses),
                mss=max(msses, key=len) if msses else [],
                tier=compute_tier("MUS_REFUTATION", single_refs, all_mus, span_index),
                counterexample_model=None,
            ))
        else:
            wrong.append(WrongAnswerExplanation(
                answer=choice.label, query_type="COUNTEREXAMPLE",
                question_type=choice.type, hypothesis_type="proposition",
                proposition_text=prop,
                single_refutations=[], all_mus=[], narrative_chain=[], mcs=[], mss=[],
                tier=compute_tier("COUNTEREXAMPLE", [], [], span_index),
                counterexample_model=build_counterexample_model(asserted, named, ctx.zvars),
            ))

    correct = []
    if verified_labels:
        forced, free, order = annotate_puzzle_solution(ctx, q_index, span_index)
        for lbl in verified_labels:
            correct.append(CorrectAnswerExplanation(lbl, forced, free, order))

    return ExplanationStruct(
        question_index=q_index,
        correct=correct,
        wrong=wrong,
        constraint_span_index={cid: tuple(span_index[cid]) for cid in span_index},
        problem_text=problem_text,
    )


def explain_problem(extracted, problem_text) -> list:
    """Run the engine over every question in an extracted LogicProblem. Spans are recomputed for
    this extraction (gold or model) via attribution.build_attribution, so cids/spans line up."""
    _methods, spans = build_attribution(extracted.model_dump(), problem_text)
    span_index = {cid: tuple(v) for cid, v in spans.items()}
    ctx = build_context(extracted)
    return [explain_question(extracted, n, ctx, span_index, problem_text)
            for n in range(len(extracted.questions))]


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization
# ─────────────────────────────────────────────────────────────────────────────

def _jsonify(o):
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    return o


def explanation_to_dict(struct: ExplanationStruct) -> dict:
    """JSON-safe dict for one question's ExplanationStruct (Enum -> value, tuples -> lists)."""
    return _jsonify(asdict(struct))


# ─────────────────────────────────────────────────────────────────────────────
# Built-in smoke test (no database required)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    import json
    from prompts import EXAMPLE_JSONS
    from validators import build_hybrid_schema

    for domains_fs, ex in EXAMPLE_JSONS.items():
        domains = list(domains_fs)
        lp = build_hybrid_schema(domains)(**ex)

        # Synthesize a problem_text from the global clues so attribution has something to match.
        problem_text = ". ".join(constraint_to_query(c) for c in ex["constraints"]) + "."

        print("=" * 78)
        print(f"DOMAINS: {domains}")
        structs = explain_problem(lp, problem_text)

        for st in structs:
            payload = explanation_to_dict(st)
            json.dumps(payload)  # must be serializable
            print(f"  Q{st.question_index}: "
                  f"correct={[c.answer for c in st.correct]}  "
                  f"wrong={[w.answer for w in st.wrong]}")
            for w in st.wrong:
                if w.query_type == "MUS_REFUTATION":
                    assert w.all_mus, f"MUS_REFUTATION {w.answer} produced no MUS"
                    print(f"    [{w.answer}] {w.query_type} tier={w.tier.value} "
                          f"MUSes={w.all_mus} chain_depth={[s.depth for s in w.narrative_chain]} "
                          f"mcs={w.mcs}")
                else:
                    assert w.counterexample_model is not None, \
                        f"COUNTEREXAMPLE {w.answer} produced no witness"
                    print(f"    [{w.answer}] {w.query_type} tier={w.tier.value} "
                          f"witness={w.counterexample_model}")
            for c in st.correct:
                print(f"    [{c.answer}] forced={[(v, val) for v, val, _ in c.forced_bindings]} "
                      f"free={c.free_bindings}")

    print("=" * 78)
    print("smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
