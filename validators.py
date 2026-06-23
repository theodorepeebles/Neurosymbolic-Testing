from __future__ import annotations
from pydantic import BaseModel, create_model, Field, ConfigDict
from typing import Annotated, Union, Optional, Literal

# --- Leaf constraint classes (split by field shape; no forward refs, safe at module level) ---

class BinaryOrdering(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["before", "immediately_before", "not_adjacent"]
    left: str
    right: str

class SlotFixed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["slot_fixed"]
    entity: str
    slot: int

class KKConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["is_truth_teller", "is_deceiver"]
    entity: str

class GroupRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["same_group", "different_group"]
    entities: list[str]

class ExactlyN(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["exactly_n"]
    entities: list[str]
    n: int
    group: Optional[int] = None


# --- Registry: domain -> LIST of leaf classes ---

DOMAIN_CONSTRAINT_CLASSES = {
    "ordering":           [BinaryOrdering, SlotFixed],
    "knights_and_knaves": [KKConstraint],
    "grouping":           [GroupRelation, ExactlyN],
}

DOMAIN_LP_FIELDS = {
    "ordering":           {"num_slots":   (int, ...)},
    "grouping":           {"num_groups":  (int, ...)},
    "knights_and_knaves": {},
}


# --- Hybrid builder: recursive classes built FRESH per call (no shared/stale state) ---

def build_hybrid_schema(active_domains: list[str]) -> type:
    leaf_classes = [cls for d in active_domains for cls in DOMAIN_CONSTRAINT_CLASSES[d]]

    # wrappers reference HybridConstraint recursively -> must be fresh per call
    IfThen = create_model("IfThen", __config__=ConfigDict(extra="forbid"),
        type=(Literal["if_then"], ...),
        antecedent=("HybridConstraint", ...),
        consequent=("HybridConstraint", ...))
    NotC = create_model("NotC", __config__=ConfigDict(extra="forbid"),
        type=(Literal["not"], ...),
        claim=("HybridConstraint", ...))
    AndOr = create_model("AndOr", __config__=ConfigDict(extra="forbid"),
        type=(Literal["and", "or"], ...),
        claims=(list["HybridConstraint"], ...))

    members = tuple(leaf_classes) + (IfThen, NotC, AndOr)
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