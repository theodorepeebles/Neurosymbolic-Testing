import requests
import json
import time
from z3 import *
import re
from collections import Counter
from validators import DOMAIN_CONSTRAINT_CLASSES, build_hybrid_schema


def ask_llm(prompt: str, system: str, fmt: str | None = None, model: str = "qwen3:8b", think: bool = False) -> str:
    """
    POSTs prompt to Ollama's local HTTP server.
    Returns:
        the stripped LLM response.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": think
    }
    if fmt:
        body["format"] = fmt

    try:
        response = requests.post("http://localhost:11434/api/generate", json=body, timeout=150)
        
        # This forces an exception if Ollama returns a 500 Internal Server Error
        response.raise_for_status() 
        
        return response.json()["response"].strip()
        
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama API timed out after 150 seconds. The server might be hung.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Did the server crash?")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


CLASSIFIER_SYSTEM = """You are a logic puzzle domain classifier. Your only job is to identify which domains a puzzle uses. Do NOT attempt to solve it.

Available domains:
  ordering           - entities occupy ordered positions or slots (first, last, before, after)
  knights_and_knaves - characters make statements; some always lie, some always tell the truth
  grouping           - entities are assigned to categories or groups

Reason through which domains the puzzle requires, then output your final answer on the last line in exactly this format:
ACTIVE_DOMAINS: <comma-separated, no spaces>

Example: ACTIVE_DOMAINS: ordering,knights_and_knaves"""

def classify_domains(problem_text: str, model: str = "qwen3:8b") -> list[str]:
    response = ask_llm(prompt=problem_text, system=CLASSIFIER_SYSTEM, think=False, model=model)

    # Scan from the end — guards against "ACTIVE_DOMAINS" appearing in reasoning trace
    domains = None
    for line in reversed(response.splitlines()):
        line = line.strip()
        if line.startswith("ACTIVE_DOMAINS:"):
            raw = line.split(":", 1)[1].strip()
            domains = [d.strip() for d in raw.split(",") if d.strip()]
            break

    if domains is None:
        raise RuntimeError(
            f"Classifier did not emit ACTIVE_DOMAINS tag.\nResponse:\n{response}"
        )

    unknown = [d for d in domains if d not in DOMAIN_CONSTRAINT_CLASSES]
    if unknown:
        raise ValueError(
            f"Classifier returned unknown domains: {unknown}. "
            f"Known: {list(DOMAIN_CONSTRAINT_CLASSES)}"
        )

    return domains


FT_EXTRACTION_SYSTEM = "Extract logic puzzles into JSON. Return ONLY a JSON object, no explanation."


def extract_finetuned(problem_text, active_domains, LogicProblem, unsat_context=None):
    base_prompt = (f"Active domains: {', '.join(active_domains)}\n\n"
                   f"Extract this logic puzzle:\n\n{problem_text}")
    prompt = f"{unsat_context}\n\n{base_prompt}" if unsat_context else base_prompt
    schema = LogicProblem.model_json_schema()

    unmatched_errors, attempts_used = [], 0
    for attempt in range(MAX_ATTEMPTS):
        raw = ask_llm(prompt=prompt, system=FT_EXTRACTION_SYSTEM,
                      fmt=schema, model="qwen3-ns")   # <-- the fine-tune
        attempts_used += 1
        try:
            return LogicProblem(**json.loads(raw)), unmatched_errors, attempts_used
        except Exception as e:
            unmatched_errors.append({"attempt": attempt + 1,
                                     "error_type": type(e).__name__, "error_msg": str(e)})
            prompt = (f"{base_prompt}\n\nYour previous JSON:\n{raw}\n\n"
                      f"It failed: {e}\n\nReturn corrected JSON.")
    return None, unmatched_errors, attempts_used






# For LLM retry upon schema validation failure
MAX_ATTEMPTS = 3


# LogicProblem is a parameter instead of a fixed import — it's built
# dynamically by build_hybrid_schema() so it can't be referenced at module level
def extract_logic_problem(
    problem_text: str,
    active_domains,
    LogicProblem: type,
    unsat_context: str = None,
    model: str = "qwen3:8b"
) -> tuple:



    unmatched_errors = []
    attempts_used = 0

    extraction_system = build_extraction_prompt(active_domains)
    base_prompt = f"Extract this logic puzzle:\n\n{problem_text}"
    prompt = f"{unsat_context}\n\n{base_prompt}" if unsat_context else base_prompt

    schema = LogicProblem.model_json_schema()
    active_hints = set()

    for attempt in range(MAX_ATTEMPTS):

        # fmt=schema: for local models this enforces grammar-based constrained generation
        # (every token hard-filtered, raw JSON guaranteed). For cloud-routed models
        # (e.g. gemma4:31b-cloud) Ollama proxies to a remote API that doesn't support
        # grammar constraints — the schema is passed as a soft hint only, so the model
        # may still wrap output in markdown fences or add prose. Strip fences before
        # json.loads (see `cleaned` below). This is not documented by Ollama explicitly;
        # confirmed empirically by observing fence-wrapped responses from cloud models.
        raw = ask_llm(prompt=prompt, system=extraction_system, fmt=schema, model=model)
        attempts_used += 1

        try:
            cleaned = re.sub(r"^```(json)?\s*|```$", "", raw.strip(), flags=re.M).strip()
            parsed = json.loads(cleaned)

            logic_prob = LogicProblem(**parsed)

            # CHANGED: validate_math_logic removed — stub this out until you
            # discover what semantic validation logic puzzles actually need
            # validate_logic(...) goes here when known
            #

            return logic_prob, unmatched_errors, attempts_used

        except Exception as e:
            error_msg = str(e)
            hint = getattr(type(e), "hint", "")   # UNCHANGED: hint mechanism stays,
            if hint:                                # but no hints are populated yet —
                active_hints.add(hint)              # you'll add them as real failures surface

            hints_block = "\n".join(f"- {h}" for h in active_hints)

            if not hint:
                unmatched_errors.append({
                    "attempt": attempt + 1,
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                })

            print(f"  [Parse failed attempt {attempt+1}/{MAX_ATTEMPTS}: {e}]")

            prompt = (
                f"Extract this logic puzzle:\n\n{problem_text}\n\n"  
                f"Your previous attempt returned this JSON:\n{raw}\n\n"
                f"It failed validation with this error: {error_msg}\n\n"
                f"Rules to remember:\n{hints_block}\n\n"
                f"Fix the error and return the corrected JSON."
            )

    return None, unmatched_errors, attempts_used


# --- Example JSONs ---

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

    frozenset(["grouping"]): {
        "entities": ["Alice", "Bob", "Carol", "Dave"],
        "num_groups": 2,
        "group_sizes": [2, 2],
        "constraints": [
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
  Valid types: before, immediately_before, not_adjacent, slot_fixed
  - before / immediately_before / not_adjacent : require "left" and "right" (entity names)
  - slot_fixed                                  : requires "entity" and "slot" (1-indexed integer)""",

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
  Valid types: same_group, different_group, exactly_n
  - same_group / different_group : require "entities" (list of entity names)
  - exactly_n                    : requires "entities", "n" (integer), and optionally "group" (1-indexed)""",
}


# --- Builder ---

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




BASELINE_LOGIC_SYSTEM = (
    "Solve the logic puzzle. Show your reasoning if needed, "
    "but your final answer must appear on the last line in this exact format:\n"
    "ANSWER: <label>\n"
    "Example: ANSWER: A"
)

def baseline_llm_solve(problem: str, model: str = "qwen3:8b", think: bool = False) -> tuple[str | None, str | None, str]:
    """
    Sends the problem directly to the LLM and parses an answer choice label.
    Returns (label, None, raw) on success, (None, error_string, raw) on parse failure.
    model and think params allow ablation testing against different configurations.
    """
    try:
        # CHANGED: system prompt reference updated
        raw = ask_llm(prompt=problem, system=BASELINE_LOGIC_SYSTEM, model=model, think=think)
    except RuntimeError as e:
        return None, f"TIMEOUT_OR_CONNECTION_ERROR: {e}", ""

    cleaned = raw.strip()

    # Look for the marker first
    # CHANGED: pattern looks for a label (one or more word chars) instead of a number
    match = re.search(r"ANSWER:\s*([A-D])\b", cleaned, re.IGNORECASE)
    if match:
        # CHANGED: return uppercased label string, not float
        return match.group(1).upper(), None, raw

    # Fallback: last standalone uppercase letter in the response
    # CHANGED: was "last number", now looks for a bare label like A/B/C/D
    matches = re.findall(r"\b([A-D])\b", cleaned)
    if matches:
        return matches[-1].upper(), "marker missing, used last uppercase label", raw

    return None, f"No answer label found in: '{raw}'", raw




def encode(c, vars: dict):
    """
    Recursively encodes a HybridConstraint into a Z3 expression.
    vars is a flat dict keyed by prefixed entity name: slot_Alice, group_Alice, kk_Alice.
    Separated from z3_solve so it can be tested and reused independently.
    """
    t = c.type

    # Ordering
    if   t == "before":             return vars[f"slot_{c.left}"] < vars[f"slot_{c.right}"]
    elif t == "immediately_before": return vars[f"slot_{c.left}"] + 1 == vars[f"slot_{c.right}"]
    elif t == "not_adjacent":       return Abs(vars[f"slot_{c.left}"] - vars[f"slot_{c.right}"]) > 1
    elif t == "slot_fixed":         return vars[f"slot_{c.entity}"] == c.slot

    # Knights and Knaves
    elif t == "is_truth_teller":    return vars[f"kk_{c.entity}"]
    elif t == "is_deceiver":        return Not(vars[f"kk_{c.entity}"])

    # Grouping
    elif t == "same_group":
        first = vars[f"group_{c.entities[0]}"]
        return And(*[vars[f"group_{e}"] == first for e in c.entities[1:]])
    elif t == "different_group":
        return Distinct([vars[f"group_{e}"] for e in c.entities])
    elif t == "exactly_n":
        return Sum([If(vars[f"group_{e}"] == c.group, 1, 0) for e in c.entities]) == c.n

    # Logical wrappers
    elif t == "if_then": return Implies(encode(c.antecedent, vars), encode(c.consequent, vars))
    elif t == "not":     return Not(encode(c.claim, vars))
    elif t == "and":     return And(*[encode(cl, vars) for cl in c.claims])
    elif t == "or":      return Or(*[encode(cl, vars) for cl in c.claims])

    else:
        raise ValueError(f"Unknown constraint type: {t}")


def _collect_types(c) -> set[str]:
    t = c.type
    types = {t}
    if   t == "if_then":     types |= _collect_types(c.antecedent) | _collect_types(c.consequent)
    elif t == "not":         types |= _collect_types(c.claim)
    elif t in ("and", "or"): 
        for cl in c.claims:  types |= _collect_types(cl)
    return types

def _used_types(extracted) -> set[str]:
    types = set()
    for c in extracted.constraints:
        types |= _collect_types(c)
    for q in extracted.questions:
        for c in q.question_constraints:
            types |= _collect_types(c)
        for ch in q.answer_choices:
            for c in ch.constraints:
                types |= _collect_types(c)
    return types

def _count_types(c, counter: Counter) -> None:
    counter[c.type] += 1
    t = c.type
    if   t == "if_then":     _count_types(c.antecedent, counter); _count_types(c.consequent, counter)
    elif t == "not":         _count_types(c.claim, counter)
    elif t in ("and", "or"):
        for cl in c.claims:  _count_types(cl, counter)

def constraint_type_counts(extracted) -> dict:
    counter = Counter()
    for c in extracted.constraints:
        _count_types(c, counter)
    for q in extracted.questions:
        for c in q.question_constraints:
            _count_types(c, counter)
        for ch in q.answer_choices:
            for c in ch.constraints:
                _count_types(c, counter)
    return dict(counter)


ORDERING_TYPES = {"before", "immediately_before", "not_adjacent", "slot_fixed"}
GROUPING_TYPES = {"same_group", "different_group", "exactly_n"}
KK_TYPES       = {"is_truth_teller", "is_deceiver"}

def z3_solve(extracted) -> dict:
    solver = Solver()

    # -------------------------------------------------------------------------
    # 1. Build vars from types actually present in extracted constraints
    #    — not from active_domains, guarding against classifier hallucination
    # -------------------------------------------------------------------------
    all_types = _used_types(extracted)
    vars = {}

    if all_types & ORDERING_TYPES:
        vars.update({f"slot_{e}": Int(f"slot_{e}") for e in extracted.entities})
        solver.add(Distinct([vars[f"slot_{e}"] for e in extracted.entities]))
        for e in extracted.entities:
            solver.add(vars[f"slot_{e}"] >= 1, vars[f"slot_{e}"] <= extracted.num_slots)

    if all_types & GROUPING_TYPES:
        vars.update({f"group_{e}": Int(f"group_{e}") for e in extracted.entities})
        for e in extracted.entities:
            solver.add(vars[f"group_{e}"] >= 1, vars[f"group_{e}"] <= extracted.num_groups)
        for g_idx, size in enumerate(extracted.group_sizes):
            g = g_idx + 1
            solver.add(
                Sum([If(vars[f"group_{e}"] == g, 1, 0) for e in extracted.entities]) == size
            )

    if all_types & KK_TYPES:
        vars.update({f"kk_{e}": Bool(f"kk_{e}") for e in extracted.entities})

    # -------------------------------------------------------------------------
    # 2. Add problem-level constraints with tracking for unsat core
    # -------------------------------------------------------------------------
    trackers = {}
    for i, c in enumerate(extracted.constraints):
        tracker = Bool(f"track_c_{i}")
        trackers[tracker] = format_constraint(c)
        solver.assert_and_track(encode(c, vars), tracker)

    # -------------------------------------------------------------------------
    # 3. Check base problem level satisfiability
    # -------------------------------------------------------------------------
    status = solver.check()
    if status == unsat:
        core = solver.unsat_core()
        core_descriptions = [trackers[b] for b in core if b in trackers]
        return {"status": "unsat", "unsat_core": core_descriptions}
    if status != sat:
        return {"status": "unknown", "answer": None}

    # -------------------------------------------------------------------------
    # 4. Verify each question independently via push/pop
    # CHANGED: was a single question, now loops over extracted.questions
    # push/pop means base constraints are built once and reused across all questions
    # -------------------------------------------------------------------------
    question_results = []

    for question in extracted.questions:
        solver.push()
        for qc in question.question_constraints:    # narrows solution space for this question only
            solver.add(encode(qc, vars))            # popped with outer solver.pop() after choices

        choice_results = {}
        for choice in question.answer_choices:
            solver.push()

            choice_expr = (
                And(*[encode(c, vars) for c in choice.constraints])
                if choice.constraints
                else BoolVal(True)
            )

            if   choice.type == "must_be_true":   solver.add(Not(choice_expr));  verified = solver.check() == unsat
            elif choice.type == "could_be_true":  solver.add(choice_expr);       verified = solver.check() == sat
            elif choice.type == "must_be_false":  solver.add(choice_expr);       verified = solver.check() == unsat
            elif choice.type == "could_be_false": solver.add(Not(choice_expr));  verified = solver.check() == sat
            else: verified = False

            choice_results[choice.label] = verified
            solver.pop()

        correct = [label for label, verified in choice_results.items() if verified]
        question_results.append(choice_results) #there is 1 choice_results per question

        solver.pop()

    return {
        "status":           "sat",
        "question_results": question_results
    }


def format_constraint(c) -> str:
    t = c.type
    if   t == "before":             return f"before({c.left}, {c.right})"
    elif t == "immediately_before": return f"immediately_before({c.left}, {c.right})"
    elif t == "not_adjacent":       return f"not_adjacent({c.left}, {c.right})"
    elif t == "slot_fixed":         return f"slot_fixed({c.entity}, slot={c.slot})"
    elif t == "is_truth_teller":    return f"is_truth_teller({c.entity})"
    elif t == "is_deceiver":        return f"is_deceiver({c.entity})"
    elif t == "same_group":         return f"same_group({', '.join(c.entities)})"
    elif t == "different_group":    return f"different_group({', '.join(c.entities)})"
    elif t == "exactly_n":          return f"exactly_n({', '.join(c.entities)}, n={c.n}, group={c.group})"
    elif t == "if_then":            return f"if_then({format_constraint(c.antecedent)}, {format_constraint(c.consequent)})"
    elif t == "not":                return f"not({format_constraint(c.claim)})"
    elif t == "and":                return f"and({', '.join(format_constraint(cl) for cl in c.claims)})"
    elif t == "or":                 return f"or({', '.join(format_constraint(cl) for cl in c.claims)})"
    else:                           return f"unknown({t})"

def _handle_unsat_retry(
    problem: str,
    active_domains,
    extracted,
    z3_result: dict,
    extract_fn
) -> tuple:

    print(f"  [UNSAT] Retrying extraction...")

    entity_summary = "  " + ", ".join(extracted.entities)

    # all three constraint levels included — unsat can originate from any layer
    constraint_summary = "Problem constraints:\n" + "\n".join(
        f"  {format_constraint(c)}" for c in extracted.constraints
    )
    for i, q in enumerate(extracted.questions):
        constraint_summary += f"\n\nQuestion {i+1} constraints:\n" + "\n".join(
            f"  {format_constraint(c)}" for c in q.question_constraints
        )
        constraint_summary += f"\n  Answer choices:\n" + "\n".join(
            f"    {ch.label}: " + ", ".join(format_constraint(c) for c in ch.constraints)
            for ch in q.answer_choices
        )

    core_info = "\n".join(f"  {d}" for d in z3_result.get("unsat_core", [])) # info about the minimal unsatisfing set

    unsat_context = (
        f"Your previous extraction produced constraints that Z3 found unsatisfiable.\n\n"
        f"Entities:\n{entity_summary}\n\n"
        f"Extracted constraints:\n{constraint_summary}\n\n"
        f"Z3 identified these specific constraints as contradictory:\n{core_info}\n\n"
        f"Re-read the problem carefully and fix the extraction."
    )

    extracted_retry, retry_unmatched, retry_calls = extract_fn(
        problem, active_domains, type(extracted), unsat_context=unsat_context
    )

    if extracted_retry is not None:
        z3_result_retry = z3_solve(extracted_retry)
        if z3_result_retry["status"] == "unsat":
            print(f"  [UNSAT occurred twice]")
        return extracted_retry, z3_result_retry, retry_calls, retry_unmatched
    else:
        return extracted, {"status": "unsat", "answer": None}, retry_calls, retry_unmatched
    


def run_ns_pipeline(problem: str, extract_fn) -> dict:
    # TODO: add model params (classify_model, extract_model, format_model) —
    # currently relies on qwen3:8b defaults in classify_domains and extract_logic_problem
    result = {
        "extracted":        None,
        "question_results": None,
        "z3_status":        None,
        "formatted_output": None,
        "llm_calls":        0,
        "unmatched_errors": []
    }
    try:
        # STEP 1 — classify domains
        print(f"  Classifying domains...")
        t_cls_start = time.time()
        active_domains = classify_domains(problem)
        t_cls_end = time.time()
        print(f"  Domains: {active_domains} ({t_cls_end - t_cls_start:.2f}s)")
        result["llm_calls"] += 1

        result["active_domains"] = active_domains

        LogicProblem      = build_hybrid_schema(active_domains)

        # STEP 2 — extract
        print(f"  Waiting for LLM extraction...")
        t_ext_start = time.time()
        extracted, unmatched_errors, ext_calls = extract_fn(
            problem, active_domains, LogicProblem
        )
        t_ext_end = time.time()
        print(f"  Got extraction response ({t_ext_end - t_ext_start:.2f}s)")

        result["llm_calls"]       += ext_calls
        result["unmatched_errors"] = unmatched_errors

        if extracted is None:
            print(f"  [!] LLM Extraction Failed after {MAX_ATTEMPTS} attempts")
            result["formatted_output"] = f"ERROR: extraction failed after {MAX_ATTEMPTS} attempts"
            return result

        result["extracted"] = extracted

        # STEP 3 — Z3 solve, with one UNSAT retry
        try:
            z3_result = z3_solve(extracted)
            if z3_result["status"] == "unsat":
                extracted, z3_result, retry_calls, retry_unmatched = _handle_unsat_retry(
                    problem, active_domains, extracted, z3_result, extract_fn
                )
                result["llm_calls"]             += retry_calls
                result["unmatched_errors"].extend(retry_unmatched)
                result["extracted"]              = extracted
        except Exception as e:
            result["formatted_output"] = f"ERROR: {type(e).__name__}: {e}"
            return result

        result["z3_status"]        = z3_result["status"]
        result["question_results"] = z3_result.get("question_results")

        print(f"  Z3 status         : {z3_result['status']}")
        print(f"  Question results  : {z3_result.get('question_results')}")

        if z3_result["status"] != "sat":
            result["formatted_output"] = f"ERROR: Z3 returned {z3_result['status']}"
            return result

        # STEP 4 — format
        # LLM explains the answer given Z3's verified result
        # placeholder for richer explanation — currently just states the correct label
        print(f"  Waiting for LLM formatting...")
        t_fmt_start = time.time()
        formatted = ask_llm(
            prompt=(
            f"Answer this logic puzzle in one sentence. State only the correct answer label and what it means, no working or reasoning. The answer is {z3_result.get('question_results')}.\nLogic puzzle: {problem}"
            ),
            system="You answer logic puzzles in one sentence. State only the correct answer label and what it means."
        )
        t_fmt_end = time.time()
        print(f"  Got formatting response ({t_fmt_end - t_fmt_start:.2f}s)")

        result["llm_calls"]        += 1
        result["formatted_output"]  = formatted

    except RuntimeError as e:
        print(f"  [!] NS Pipeline failed due to timeout/connection error: {e}")
        result["formatted_output"] = f"ERROR: TIMEOUT_OR_CONNECTION_ERROR"
        return result

    return result