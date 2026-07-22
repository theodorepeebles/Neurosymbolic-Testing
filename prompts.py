import json

# All LLM-facing text lives here — system prompts and the non-finetuned
# extraction prompt scaffolding. pipeline.py holds the mechanics; this file
# holds the words we send the model.


# --- Classifier ---

CLASSIFIER_SYSTEM = """You are a logic puzzle domain classifier. Your only job is to identify which domains a puzzle uses. Do NOT attempt to solve it.

Available domains:
  ordering           - entities occupy ordered positions or slots (first, last, before, after)
  knights_and_knaves - characters make statements; some always lie, some always tell the truth
  grouping           - entities are assigned to categories or groups

Reason through which domains the puzzle requires, then output your final answer on the last line in exactly this format:
ACTIVE_DOMAINS: <comma-separated, no spaces>

Example: ACTIVE_DOMAINS: ordering,knights_and_knaves"""


# --- Extraction system prompts ---

FT_EXTRACTION_SYSTEM = "Extract logic puzzles into JSON. Return ONLY a JSON object, no explanation."


# --- Baseline solve ---

BASELINE_LOGIC_SYSTEM = (
    "Solve the logic puzzle. Show your reasoning if needed, "
    "but your final answer must appear on the last line in this exact format:\n"
    "ANSWER: <label>\n"
    "Example: ANSWER: A"
)


# --- Example JSONs for non-finetuned models ---

EXAMPLE_JSONS = {
    frozenset(["ordering"]): {
        "entities": ["Alice", "Bob", "Carol"],
        "num_slots": 3,
        "constraints": [
            {"type": "before", "left": "Bob", "right": "Carol"},
            # "Alice is not before Bob" — no primitive for this, so use not wrapper
            {
                "type": "not",
                "claim": {"type": "before", "left": "Alice", "right": "Bob"}
            }
        ],
        "questions": [{
            "question_constraints": [],
            "answer_choices": [
                {
                    "label": "A",
                    "type": "must_be_true",
                    "constraints": [{"type": "slot_fixed", "entity": "Bob", "slot": 2}]
                },
                {
                    "label": "B",
                    "type": "could_be_true",
                    "constraints": [{"type": "immediately_before", "left": "Alice", "right": "Carol"}]
                }
            ]
        }]
    },

    frozenset(["knights_and_knaves"]): {
        "entities": ["Alice", "Bob", "Carol"],
        "constraints": [
            # Carol makes no statement — appears as a direct ground constraint
            {"type": "is_truth_teller", "entity": "Carol"},
            # Alice says "Bob is a truth-teller" — encoded as if_then pair
            {
                "type": "if_then",
                "antecedent": {"type": "is_truth_teller", "entity": "Alice"},
                "consequent": {"type": "is_truth_teller", "entity": "Bob"}
            },
            {
                "type": "if_then",
                "antecedent": {"type": "is_deceiver", "entity": "Alice"},
                # deceiver case: negate the statement with not wrapper
                "consequent": {
                    "type": "not",
                    "claim": {"type": "is_truth_teller", "entity": "Bob"}
                }
            }
        ],
        "questions": [{
            "question_constraints": [],
            "answer_choices": [
                {
                    "label": "A",
                    "type": "must_be_true",
                    "constraints": [{"type": "is_truth_teller", "entity": "Alice"}]
                },
                {
                    "label": "B",
                    "type": "must_be_true",
                    "constraints": [{"type": "is_deceiver", "entity": "Alice"}]
                }
            ]
        }]
    },
    #
    frozenset(["grouping"]): {
        "entities": ["Alice", "Bob", "Carol", "Dave"],
        "num_groups": 2,
        "constraints": [
            # group sizes encoded as exactly_n, not as a group_sizes field
            {"type": "exactly_n", "entities": ["Alice", "Bob", "Carol", "Dave"], "n": 2, "group": 1},
            {"type": "exactly_n", "entities": ["Alice", "Bob", "Carol", "Dave"], "n": 2, "group": 2},
            {"type": "different_group", "entities": ["Alice", "Bob"]},
            # disjunctive constraint — either pairing is acceptable
            {
                "type": "or",
                "claims": [
                    {"type": "same_group", "entities": ["Alice", "Carol"]},
                    {"type": "same_group", "entities": ["Bob", "Dave"]}
                ]
            }
        ],
        "questions": [{
            "question_constraints": [],
            "answer_choices": [
                {
                    "label": "A",
                    "type": "must_be_true",
                    "constraints": [{"type": "same_group", "entities": ["Alice", "Carol"]}]
                },
                {
                    "label": "B",
                    "type": "could_be_true",
                    "constraints": [{"type": "different_group", "entities": ["Bob", "Carol"]}]
                }
            ]
        }]
    }
}


# --- Rules ---

LOGICAL_WRAPPER_RULES = """\
Logical wrapper — use to combine or negate any constraint:
  Valid types: if_then, not, and, or
  - if_then : requires "antecedent" and "consequent" (each a full constraint object)
  - not     : requires "claim" (a single constraint object)
  - and/or  : require "claims" (a list of constraint objects)
  - Wrappers can reference constraints from any active domain, including mixing domains"""

DOMAIN_RULES = {
    "ordering": """\
Ordering constraints:
  Valid types: before, immediately_before, adjacent, slot_fixed
  - before / immediately_before / adjacent : require "left" and "right" (entity names)
  - slot_fixed                             : requires "entity" and "slot" (1-indexed integer)""",

    "knights_and_knaves": """\
Knights and Knaves constraints:
  Valid types: is_truth_teller, is_deceiver
  - Both require "entity" (entity name)
  - An entity that makes no statement can appear as a bare is_truth_teller or is_deceiver constraint
  - If an entity makes a statement, encode it as an if_then PAIR:
      if speaker is_truth_teller → the statement as a constraint
      if speaker is_deceiver     → the negation of the statement (wrap consequent with "not")""",

    "grouping": """\
Grouping constraints:
  Valid types: same_group, different_group, exactly_n, is_in
  - same_group / different_group : require "entities" (list of entity names)
  - exactly_n                    : requires "entities", "n" (integer), and optionally "group" (1-indexed)
  - is_in                        : requires "entity" (name) and "group" (1-indexed integer)""",
}


# --- Builder ---
# only used for the non-finetuned models
def build_extraction_prompt(active_domains: list[str]) -> str:
    key = frozenset(active_domains)

    if key in EXAMPLE_JSONS:
        # exact match — single example
        examples = [EXAMPLE_JSONS[key]]
    else:
        # no hybrid example — show one primitive per active domain
        # every constraint type and wrapper gets demonstrated from whichever
        # single-domain examples are relevant; rules carry the composition logic
        examples = [
            EXAMPLE_JSONS[frozenset([d])]
            for d in active_domains
            if frozenset([d]) in EXAMPLE_JSONS
        ]

    examples_block = "\n\n".join(
        f"Example {i + 1}:\n{json.dumps(ex, indent=2)}"
        for i, ex in enumerate(examples)
    )

    domain_rules = "\n\n".join(DOMAIN_RULES[d] for d in active_domains)

    return (
        "Extract logic puzzles into JSON. Return ONLY a JSON object with no explanation.\n\n"
        f"Use this exact structure:\n{examples_block}\n\n"
        "Rules:\n\n"
        f"{LOGICAL_WRAPPER_RULES}\n\n"
        f"{domain_rules}"
    )
