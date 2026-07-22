from __future__ import annotations
from pydantic import BaseModel, create_model, Field, ConfigDict
from typing import Annotated, Union, Optional, Literal

# --- Leaf constraint classes (split by field shape; no forward refs, safe at module level) ---


# Grouped by field shape, not by individual `type` nor by domain:
#   - by type   -> near-duplicate classes (before/immediately_before/adjacent are
#                  field-identical); bloats the schema and duplicates every field.
#   - by domain -> one kitchen-sink class must union every field any type in the domain
#                  uses, most inapplicable per type. Because all fields are then declared,
#                  extra="forbid" can't reject wrong-field combos, and constrained decoding
#                  can emit nonsense (e.g. a slot_fixed carrying left/right).
# By shape, each class declares exactly its fields, so extra="forbid" rejects anything
# extra and the grammar can't represent invalid type/field combos -- limiting what the LLM
# can emit wrong. The `type` Literal still enumerates each value, so per-type dispatch on
# the discriminated union is unchanged.


# evidence_text: the exact text clause in problem_text this constraint was extracted from.
# The LLM emits the text; attribution.py converts it
# to a [start, end] span later by substring-matching against problem_text. Optional (default
# None) because nested children of logical wrappers always keep None (the wrapper carries the clue's sentence)
# and in case the LLM fails to extract it

class BinaryOrdering(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["before", "immediately_before", "adjacent"]
    left: str
    right: str
    evidence_text: Optional[str] = None

class SlotFixed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["slot_fixed"]
    entity: str
    slot: int
    evidence_text: Optional[str] = None

class KKConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["is_truth_teller", "is_deceiver"]
    entity: str
    evidence_text: Optional[str] = None

class GroupRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["same_group", "different_group"]
    entities: list[str]
    evidence_text: Optional[str] = None

class ExactlyN(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["exactly_n"]
    entities: list[str]
    n: int
    group: Optional[int] = None
    evidence_text: Optional[str] = None

class IsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["is_in"]
    entity: str
    group: int
    evidence_text: Optional[str] = None


# LEFT OFF HERE
# --- Registry: domain -> LIST of leaf classes ---

DOMAIN_CONSTRAINT_CLASSES = {
    "ordering":           [BinaryOrdering, SlotFixed],
    "knights_and_knaves": [KKConstraint],
    "grouping":           [GroupRelation, ExactlyN, IsIn],
}

DOMAIN_LP_FIELDS = {
    "ordering":           {"num_slots":   (int, ...)},
    "grouping":           {"num_groups":  (int, ...)},
    "knights_and_knaves": {},
}


# --- Hybrid builder: recursive classes built FRESH per call (no shared/stale state) ---
#
def build_hybrid_schema(active_domains: list[str]) -> type:
    leaf_classes = [cls for d in active_domains for cls in DOMAIN_CONSTRAINT_CLASSES[d]]

    # wrappers reference HybridConstraint recursively -> must be fresh per call
    # evidence_text on wrappers too: a wrapper is one clue, so it carries the clue's
    # source sentence; nested children keep evidence_text=None (no sub-sentence recursion).
    IfThen = create_model("IfThen", __config__=ConfigDict(extra="forbid"),
        type=(Literal["if_then"], ...),
        antecedent=("HybridConstraint", ...),
        consequent=("HybridConstraint", ...),
        evidence_text=(Optional[str], None))
    NotC = create_model("NotC", __config__=ConfigDict(extra="forbid"),
        type=(Literal["not"], ...),
        claim=("HybridConstraint", ...),
        evidence_text=(Optional[str], None))
    AndOr = create_model("AndOr", __config__=ConfigDict(extra="forbid"),
        type=(Literal["and", "or"], ...),
        claims=(list["HybridConstraint"], ...),
        evidence_text=(Optional[str], None))

    members = tuple(leaf_classes) + (IfThen, NotC, AndOr)

    # LLM only allowed to fill in fields associated with the leaf class containing whatever type is chosen
    HybridConstraint = Annotated[Union[members], Field(discriminator="type")]

    namespace = {"HybridConstraint": HybridConstraint}
    for w in (IfThen, NotC, AndOr):
        w.model_rebuild(_types_namespace=namespace)

    AnswerChoice = create_model("AnswerChoice",
        label=(str, ...),
        type=(Literal["must_be_true", "could_be_true", "must_be_false", "could_be_false"], ...),
        constraints=(list[HybridConstraint], ...))
    Question = create_model("Question",
        question_constraints=(list[HybridConstraint], ...),
        answer_choices=(list[AnswerChoice], ...))

    base_fields = {
        "entities":    (list[str], ...),
        "constraints": (list[HybridConstraint], ...),
        "questions":   (list[Question], ...),
    }
    for d in active_domains:
        base_fields.update(DOMAIN_LP_FIELDS[d])

    return create_model("LogicProblem", **base_fields)