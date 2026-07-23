"""
Verbalization (Stage 2 — the formatted-output step) for the neurosymbolic logic-puzzle pipeline.

Turns the Z3-derived explanation structures from explanation.py into human-readable prose using
templates *only* — no LLM. This is the single prose path: both the primary multiple-choice
explanation (pipeline STEP 4 / run.py, via `verbalize_struct`) and the interactive REPL
(repl.py, via the `vinput_for_*` builders) converge on one `render(VerbalizationInput)`.

Design constraints:
  - This module must NOT import `explanation` or `pipeline` at module load. explanation.py imports
    pipeline, and pipeline.run_ns_pipeline imports this module (locally) for STEP 4; keeping this
    file dependency-light avoids an import cycle. It reads explanation objects (WrongAnswerExplanation,
    CorrectAnswerExplanation, ConstraintStep) duck-typed, and takes a cid->label dict from the caller.
  - The renderer invents no reasoning. Every clue label, MUS, MCS and binding it prints comes
    straight from the ExplanationStruct (or a REPL query result built from the same primitives).

Standalone smoke test (no model / DB required):

    python verbalization.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data contract — one VerbalizationInput -> VerbOutput path for every caller
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepInput:
    """One clue in a narrative chain, flattened for rendering."""
    cid: str                 # constraint id, kept for hover-highlighting in a UI
    label: str               # human label resolved from the caller's cid_label dict
    depth: int               # causal depth (0 = fires from the hypothesis alone)
    is_contradiction: bool   # True only for the final, failing clue of a MUS
    contradiction_str: str   # what that final clue is incompatible with (the tested claim)


@dataclass
class VerbalizationInput:
    opening_frame: str                              # framing sentence, computed by the builder
    query_type: str                                 # MUS_REFUTATION | COUNTEREXAMPLE | CORRECT | FORCED | FREE
    steps: list = field(default_factory=list)       # list[StepInput]
    mcs_labels: list = field(default_factory=list)  # human labels of MCS clues (drop-to-fix set)
    additional_mus_labels: list = field(default_factory=list)  # list[list[str]] alternative MUSes
    counterexample_model: Optional[dict] = None     # var -> value witness (COUNTEREXAMPLE)
    counterexample_lead: str = "For example:"        # lead-in for the witness (set by builder)
    forced_bindings: list = field(default_factory=list)  # (var, value, [clue_labels])
    free_bindings: list = field(default_factory=list)    # (var, value)
    answer: Optional[str] = None                    # answer label / variable, for headers


@dataclass
class VerbOutput:
    prose: str
    cids_in_order: list      # [s.cid for s in steps] — kept for hover-highlighting
    mcs_prose: str


# ─────────────────────────────────────────────────────────────────────────────
# Framing + small formatters (own the question-type frames; mirror explanation_debug)
# ─────────────────────────────────────────────────────────────────────────────

QTYPE_FRAME = {
    ("could_be_true",  "MUS_REFUTATION"): "is IMPOSSIBLE",
    ("could_be_false", "MUS_REFUTATION"): "must ALWAYS be true (cannot be false)",
    ("must_be_true",   "COUNTEREXAMPLE"): "is NOT necessarily true",
    ("must_be_false",  "COUNTEREXAMPLE"): "is NOT necessarily false (it can hold)",
}


def describe_binding(var: str, val) -> str:
    """A prefixed Z3 var name + value as a readable clause: slot_Alice=2 -> 'Alice is in slot 2'."""
    if "_" not in var:
        return f"{var} = {val}"
    prefix, entity = var.split("_", 1)
    if prefix == "slot":
        return f"{entity} is in slot {val}"
    if prefix == "group":
        return f"{entity} is in group {val}"
    if prefix == "kk":
        return f"{entity} is a truth-teller" if val else f"{entity} is a deceiver"
    return f"{var} = {val}"


def format_mcs(mcs_labels: list) -> str:
    if not mcs_labels:
        return ""
    return "This would become possible if you dropped: " + ", ".join(mcs_labels) + "."


def frame_for_choice(question_type: str, query_type: str, proposition_text: str) -> str:
    """Opening sentence for a refuted/counter-exampled claim. Falls back to a generic frame for
    REPL queries (whose question_type is the sentinel 'query', not a multiple-choice modality)."""
    frame = QTYPE_FRAME.get((question_type, query_type))
    if frame is not None:
        suffix = ", because:" if query_type == "MUS_REFUTATION" else "."
        return f'The claim "{proposition_text}" {frame}{suffix}'
    if query_type == "MUS_REFUTATION":
        return f'"{proposition_text}" is impossible, because:'
    return f'"{proposition_text}" is possible.'


def _label(cid: str, labels: dict) -> str:
    return labels.get(cid, cid)


# ─────────────────────────────────────────────────────────────────────────────
# The single prose path
# ─────────────────────────────────────────────────────────────────────────────

def render(vinput: VerbalizationInput) -> VerbOutput:
    """Template-only rendering of any VerbalizationInput. The one and only verbalization path."""
    mcs_prose = format_mcs(vinput.mcs_labels)
    parts = [vinput.opening_frame] if vinput.opening_frame else []

    if vinput.query_type == "COUNTEREXAMPLE":
        if vinput.counterexample_model:
            assignments = "; ".join(
                describe_binding(k, v) for k, v in sorted(vinput.counterexample_model.items())
            )
            parts.append(f"{vinput.counterexample_lead} {assignments}.")
        return VerbOutput(prose=" ".join(parts), cids_in_order=[], mcs_prose=mcs_prose)

    if vinput.query_type in ("CORRECT", "FORCED", "FREE"):
        for var, val, clue_labels in vinput.forced_bindings:
            by = f", forced by {', '.join(clue_labels)}" if clue_labels else ""
            parts.append(f"{describe_binding(var, val)}{by}.")
        if vinput.free_bindings:
            if vinput.query_type == "FREE":
                for var, val in vinput.free_bindings:
                    parts.append(
                        f"{describe_binding(var, val)}, but this is not forced — "
                        f"other assignments are possible."
                    )
            else:
                frees = "; ".join(describe_binding(var, val) for var, val in vinput.free_bindings)
                parts.append(f"Consistent but not uniquely determined: {frees}.")
        return VerbOutput(prose=" ".join(parts), cids_in_order=[], mcs_prose=mcs_prose)

    # MUS_REFUTATION (default)
    for step in vinput.steps:
        if step.is_contradiction:
            parts.append(f"{step.label}, which is incompatible with \"{step.contradiction_str}\".")
        else:
            parts.append(f"{step.label}.")

    for mus_labels in vinput.additional_mus_labels:
        parts.append("Independently, it also fails because: " + "; ".join(mus_labels) + ".")

    return VerbOutput(
        prose=" ".join(parts),
        cids_in_order=[s.cid for s in vinput.steps],
        mcs_prose=mcs_prose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Builders: explanation objects (or REPL query results) -> VerbalizationInput
# ─────────────────────────────────────────────────────────────────────────────

def _step_from(cs, labels: dict, proposition_text: str) -> StepInput:
    cid = cs.constraint_ids[0] if cs.constraint_ids else ""
    return StepInput(
        cid=cid,
        label=_label(cid, labels),
        depth=cs.depth,
        is_contradiction=cs.is_contradiction,
        contradiction_str=proposition_text,
    )


def vinput_for_wrong(w, labels: dict, proposition_override: Optional[str] = None) -> VerbalizationInput:
    """A WrongAnswerExplanation (MC choice or REPL why-not/can result) -> VerbalizationInput."""
    prop = proposition_override if proposition_override is not None else w.proposition_text
    frame = frame_for_choice(w.question_type, w.query_type, prop)

    if w.query_type == "COUNTEREXAMPLE":
        # The witness is a model of what was *asserted*: for must_be_true that means the claim
        # FAILS in it (proving it's not necessary); for must_be_false it HOLDS; for a REPL binding
        # query the binding holds (it's satisfiable). Word the lead-in accordingly.
        if w.question_type == "must_be_true":
            lead = "One valid arrangement where it does not hold:"
        elif w.question_type == "must_be_false":
            lead = "One valid arrangement where it does hold:"
        else:
            lead = "For example:"
        return VerbalizationInput(
            opening_frame=frame,
            query_type="COUNTEREXAMPLE",
            counterexample_model=w.counterexample_model,
            counterexample_lead=lead,
            answer=w.answer,
        )

    chain = w.narrative_chain or w.single_refutations
    steps = [_step_from(cs, labels, prop) for cs in chain]
    extra = [[_label(cid, labels) for cid in mus] for mus in (w.all_mus[1:] if w.all_mus else [])]
    return VerbalizationInput(
        opening_frame=frame,
        query_type="MUS_REFUTATION",
        steps=steps,
        mcs_labels=[_label(cid, labels) for cid in w.mcs],
        additional_mus_labels=extra,
        answer=w.answer,
    )


def vinput_for_correct(c, labels: dict) -> VerbalizationInput:
    """A primary CorrectAnswerExplanation (full forced/free annotation) -> VerbalizationInput."""
    forced = [(var, val, [_label(cid, labels) for cid in cids])
              for var, val, cids in c.forced_bindings]
    return VerbalizationInput(
        opening_frame=f"Answer {c.answer} is correct.",
        query_type="CORRECT",
        forced_bindings=forced,
        free_bindings=[(var, val) for var, val in c.free_bindings],
        answer=c.answer,
    )


def vinput_for_forced(var: str, val, clue_labels: list) -> VerbalizationInput:
    """REPL 'what forces X?' — the variable is uniquely determined."""
    return VerbalizationInput(
        opening_frame="",
        query_type="FORCED",
        forced_bindings=[(var, val, list(clue_labels))],
        answer=var,
    )


def vinput_for_free(var: str, val) -> VerbalizationInput:
    """REPL 'what forces X?' — the variable is free (this value is just one consistent choice)."""
    return VerbalizationInput(
        opening_frame="",
        query_type="FREE",
        free_bindings=[(var, val)],
        answer=var,
    )


def verbalize_struct(structs, labels: dict) -> str:
    """Full prose for a list of ExplanationStructs (one per question). Used by pipeline STEP 4 and
    run.py. `labels` is build_context(extracted).cid_label."""
    blocks = []
    for st in structs:
        lines = [f"Question {st.question_index}:"]
        for c in st.correct:
            vo = render(vinput_for_correct(c, labels))
            lines.append(f"  [{c.answer}] {vo.prose}")
        for w in st.wrong:
            vo = render(vinput_for_wrong(w, labels))
            line = f"  [{w.answer}] {vo.prose}"
            if vo.mcs_prose:
                line += " " + vo.mcs_prose
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in smoke test (no database / model required)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    # Local imports so the module itself stays free of explanation/pipeline at load time.
    from prompts import EXAMPLE_JSONS
    from validators import build_hybrid_schema
    from explanation import explain_problem, build_context
    from attribution import constraint_to_query

    for domains_fs, ex in EXAMPLE_JSONS.items():
        domains = list(domains_fs)
        lp = build_hybrid_schema(domains)(**ex)
        problem_text = ". ".join(constraint_to_query(c) for c in ex["constraints"]) + "."

        structs = explain_problem(lp, problem_text)
        labels = build_context(lp).cid_label

        print("=" * 78)
        print(f"DOMAINS: {domains}")
        print(verbalize_struct(structs, labels))
        print()

    print("=" * 78)
    print("verbalization smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
