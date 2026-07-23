import requests
import json
import time
from z3 import *
import re
from collections import Counter
from validators import DOMAIN_CONSTRAINT_CLASSES, build_hybrid_schema
from prompts import (
    CLASSIFIER_SYSTEM,
    FT_EXTRACTION_SYSTEM,
    BASELINE_LOGIC_SYSTEM,
    build_extraction_prompt,
)


def ask_llm(prompt: str, system: str, model: str, fmt: str | None = None, think: bool = False, timeout: int = 150, is_extraction: bool = False) -> str:
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
        "think": think,
    }
    # not sure how redundant this body options stuff is - need to do direct comparison testing with and without it on a large dataset - on 30 examples it performed exactly the same
    # definitely redundant for my fine tuned models but not redundant for testing on regular qwen or other non fine tuned models
    if is_extraction:
        body["options"] = {
            "temperature": 0,
            "top_k": 1,
            "num_predict": 1024,
            "num_ctx": 4096,
        }
    if fmt:
        body["format"] = fmt
        
    try:
        response = requests.post("http://localhost:11434/api/generate", json=body, timeout=timeout)

        # This forces an exception if Ollama returns a 500 Internal Server Error
        response.raise_for_status()

        return response.json()["response"].strip()

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama API timed out after {timeout} seconds. The server might be hung.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Did the server crash?")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


def classify_domains(problem_text: str, model: str) -> list[str]:
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



# For LLM retry upon schema validation failure
MAX_ATTEMPTS = 1


# LogicProblem is a parameter instead of a fixed import — it's built
# dynamically by build_hybrid_schema() so it can't be referenced at module level
def extract_logic_problem(
    problem_text: str,
    active_domains,
    LogicProblem: type,
    model: str,
    unsat_context: str = None,
    finetuned: bool = False,
) -> tuple:
    """
    Extract a logic puzzle into a LogicProblem via the LLM, with schema-validation retry.

    finetuned=True routes to a fine-tuned extraction model: the minimal
    FT_EXTRACTION_SYSTEM system prompt, active domains prepended to the user prompt,
    and raw json.loads (no fence-stripping — the FT model runs under fmt=schema grammar
    constraints that force output to start with '{', so fences/preamble can't occur).
    finetuned=False (default) uses the rich domain-specific prompt with examples and
    strips markdown fences before parsing (cloud-routed models may wrap output).
    """

    unmatched_errors = []
    attempts_used = 0

    if finetuned:
        extraction_system = FT_EXTRACTION_SYSTEM
        base_prompt = (f"Active domains: {', '.join(active_domains)}\n\n"
                       f"Extract this logic puzzle:\n\n{problem_text}")
    else:
        extraction_system = build_extraction_prompt(active_domains)
        base_prompt = f"Extract this logic puzzle:\n\n{problem_text}"
    prompt = f"{unsat_context}\n\n{base_prompt}" if unsat_context else base_prompt

    schema = LogicProblem.model_json_schema()
    active_hints = set()

    last_raw = ""
    for attempt in range(MAX_ATTEMPTS):

        # fmt=schema: for local models this enforces grammar-based constrained generation
        # (every token hard-filtered, raw JSON guaranteed). UNSURE ABT THIS: For cloud-routed models
        # (e.g. gemma4:31b-cloud) Ollama proxies to a remote API that doesn't support
        # grammar constraints — the schema is passed as a soft hint only, so the model
        # may still wrap output in markdown fences or add prose. Strip fences before
        # json.loads (see `cleaned` below). This is not documented by Ollama explicitly;
        # confirmed empirically by observing fence-wrapped responses from cloud models.
        raw = ask_llm(prompt=prompt, system=extraction_system, fmt=schema, model=model, is_extraction=True)
        last_raw = raw
        attempts_used += 1

        try:
            if finetuned:
                # FT model runs under grammar constraints → raw JSON, no fences to strip
                parsed = json.loads(raw)
            else:
                cleaned = re.sub(r"^```(json)?\s*|```$", "", raw.strip(), flags=re.M).strip()
                parsed = json.loads(cleaned)

            logic_prob = LogicProblem(**parsed)

            # CHANGED: validate_math_logic removed — stub this out until you
            # discover what semantic validation logic puzzles actually need
            # validate_logic(...) goes here when known

            return logic_prob, unmatched_errors, attempts_used, raw

        except Exception as e:
            error_msg = str(e)
            # NOTE: this hint / unmatched_errors triage is inert in this domain. With a
            # closed, grammar-constrained constraint vocabulary, structural validity ==
            # semantic legality, so no semantic validator runs — no hints ever fire and
            # nothing reaches unmatched_errors. This machinery only earns its keep where
            # there's a gap between structural validity and semantic legality (math:
            # nonlinear/self-referential exprs; contracts: dangling clause/party refs),
            # where a semantic validator raises *recoverable* errors to triage into
            # known (hint) vs novel (unmatched).
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
                f"{base_prompt}\n\n"
                f"Your previous attempt returned this JSON:\n{raw}\n\n"
                f"It failed validation with this error: {error_msg}\n\n"
                f"Rules to remember:\n{hints_block}\n\n"
                f"Fix the error and return the corrected JSON."
            )

    return None, unmatched_errors, attempts_used, last_raw




def baseline_llm_solve(problem: str, model: str, think: bool = False) -> tuple[str | None, str | None, str]:
    """
    Sends the problem directly to the LLM and parses an answer choice label.
    Returns (label, None, raw) on success, (None, error_string, raw) on parse failure.
    model and think params allow ablation testing against different configurations.
    """
    try:
        raw = ask_llm(prompt=problem, system=BASELINE_LOGIC_SYSTEM, model=model, think=think)
    except RuntimeError as e:
        return None, f"TIMEOUT_OR_CONNECTION_ERROR: {e}", ""

    cleaned = raw.strip()

    # Look for the marker first
    match = re.search(r"ANSWER:\s*([A-D])\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper(), None, raw

    # Fallback: last standalone uppercase letter in the response
    matches = re.findall(r"\b([A-D])\b", cleaned)
    if matches:
        return matches[-1].upper(), "marker missing, used last uppercase label", raw

    return None, f"No answer label found in: '{raw}'", raw




def encode(c, vars: dict):
    """
    Encodes the formal logic of each constraint.
    Recursively encodes a HybridConstraint into a Z3 expression.
    vars is a flat dict keyed by prefixed entity name: slot_Alice, group_Alice, kk_Alice.
    """
    t = c.type

    # Ordering
    if   t == "before":             return vars[f"slot_{c.left}"] < vars[f"slot_{c.right}"]
    elif t == "immediately_before": return vars[f"slot_{c.left}"] + 1 == vars[f"slot_{c.right}"]
    elif t == "adjacent":           return Abs(vars[f"slot_{c.left}"] - vars[f"slot_{c.right}"]) == 1
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
    elif t == "is_in":
        return vars[f"group_{c.entity}"] == c.group

    # Logical wrappers
    elif t == "if_then": return Implies(encode(c.antecedent, vars), encode(c.consequent, vars))
    elif t == "not":     return Not(encode(c.claim, vars))
    elif t == "and":     return And(*[encode(cl, vars) for cl in c.claims])
    elif t == "or":      return Or(*[encode(cl, vars) for cl in c.claims])

    else:
        raise ValueError(f"Unknown constraint type: {t}")


# Recursively gather the distinct type tags in one constraint tree -> set[str].
def _collect_types(c) -> set[str]:
    t = c.type
    types = {t}
    if   t == "if_then":     types |= _collect_types(c.antecedent) | _collect_types(c.consequent)
    elif t == "not":         types |= _collect_types(c.claim)
    elif t in ("and", "or"): 
        for cl in c.claims:  types |= _collect_types(cl)
    return types

# Gather the set of distinct constraint types used across a whole extraction -> set[str].
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

# Recursively tally each type tag in one constraint tree into `counter` returns None (mutates).
def _count_types(c, counter: Counter) -> None:
    counter[c.type] += 1
    t = c.type
    if   t == "if_then":     _count_types(c.antecedent, counter); _count_types(c.consequent, counter)
    elif t == "not":         _count_types(c.claim, counter)
    elif t in ("and", "or"):
        for cl in c.claims:  _count_types(cl, counter)

# Count occurrences of every constraint type across a whole extraction -> dict[str, int].
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


ORDERING_TYPES = {"before", "immediately_before", "adjacent", "slot_fixed"}
GROUPING_TYPES = {"same_group", "different_group", "exactly_n", "is_in"}
KK_TYPES       = {"is_truth_teller", "is_deceiver"}


def z3_solve(extracted) -> dict:
    solver = Solver()

    # -------------------------------------------------------------------------
    # 1. Build vars from types actually present in extracted constraints
    #    — not from active_domains, guarding against classifier hallucination
    # -------------------------------------------------------------------------
    all_types = _used_types(extracted)
    vars = {}


    #encode basic domain axioms 

    if all_types & ORDERING_TYPES:
        vars.update({f"slot_{e}": Int(f"slot_{e}") for e in extracted.entities})
        solver.add(Distinct([vars[f"slot_{e}"] for e in extracted.entities]))
        for e in extracted.entities:
            solver.add(vars[f"slot_{e}"] >= 1, vars[f"slot_{e}"] <= extracted.num_slots)

    if all_types & GROUPING_TYPES:
        vars.update({f"group_{e}": Int(f"group_{e}") for e in extracted.entities})
        for e in extracted.entities:
            solver.add(vars[f"group_{e}"] >= 1, vars[f"group_{e}"] <= extracted.num_groups)
        # group sizes are encoded as exactly_n constraints in the constraint list

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

        question_results.append(choice_results) # there is 1 choice_results per question

        solver.pop()

    return {
        "status":           "sat",
        "question_results": question_results
    }


def format_constraint(c) -> str:
    # Serializes a structured constraint into a readable function-call-style string
    # (e.g. before(A, B)); recurses into nested constraints for the logical
    # connectives (if_then, not, and, or). Used across the project for:
    #   - labeling Z3 solver trackers for debugging / unsat-core inspection
    #     (encode at :347, plus explanation.py and ns_training.py)
    #   - rendering extracted puzzles into readable text (:434, :438, :441)
    #   - displaying a choice's constraints in the chatbox UI (chatbox.py)
    t = c.type
    if   t == "before":             return f"before({c.left}, {c.right})"
    elif t == "immediately_before": return f"immediately_before({c.left}, {c.right})"
    elif t == "adjacent":           return f"adjacent({c.left}, {c.right})"
    elif t == "slot_fixed":         return f"slot_fixed({c.entity}, slot={c.slot})"
    elif t == "is_truth_teller":    return f"is_truth_teller({c.entity})"
    elif t == "is_deceiver":        return f"is_deceiver({c.entity})"
    elif t == "same_group":         return f"same_group({', '.join(c.entities)})"
    elif t == "different_group":    return f"different_group({', '.join(c.entities)})"
    elif t == "exactly_n":          return f"exactly_n({', '.join(c.entities)}, n={c.n}, group={c.group})"
    elif t == "is_in":              return f"is_in({c.entity}, group={c.group})"
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
    """
    CURRENTLY UNUSED. The only call site is commented out in run_ns_pipeline
    (the '*** UNSAT RETRY DISABLED ***' block), and the live extraction call
    never passes unsat_context — so nothing reaches this function today.

    Recovery path for a base-problem Z3 unsat: it feeds the contradictory
    constraints plus Z3's unsat core back to the extractor as unsat_context and
    re-solves. It's left disabled because this "explain the error and ask the
    model to fix it" strategy is a poor fit for the small fine-tuned extraction
    model, which was never trained on correction prompts and treats that input as
    out-of-distribution.

    Retained for a possible future redesign.
    """

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

    extracted_retry, retry_unmatched, retry_calls, _retry_raw = extract_fn(
        problem, active_domains, type(extracted), unsat_context=unsat_context
    )

    if extracted_retry is not None:
        z3_result_retry = z3_solve(extracted_retry)
        if z3_result_retry["status"] == "unsat":
            print(f"  [UNSAT occurred twice]")
        return extracted_retry, z3_result_retry, retry_calls, retry_unmatched
    else:
        return extracted, {"status": "unsat", "answer": None}, retry_calls, retry_unmatched
    


def run_ns_pipeline(problem: str, extract_fn, classifier_model: str, formatter_model: str,
                    active_domains: list[str] | None = None,
                    use_ground_truth_domains: bool = True,
                    verbalize: bool = False) -> dict:
    result = {
        "extracted":        None,
        "extraction_raw":   None,
        "question_results": None,
        "z3_status":        None,
        "formatted_output": None,
        "llm_calls":        0,
        "unmatched_errors": []
    }
    try:
        # STEP 1 — determine active domains
        # use_ground_truth_domains=True skips the LLM classifier and trusts the
        # active_domains injected by the caller (read from each sft_test.jsonl row
        # in run.py), so each problem gets its ground-truth domains. Set to False
        # to instead predict domains with the classifier model (e.g. qwen3:8b).
        if use_ground_truth_domains:
            if active_domains is None:
                raise ValueError(
                    "use_ground_truth_domains=True but no ground-truth active_domains "
                    "were provided for this problem. This happens on a novel problem that "
                    "isn't in the dataset, so it has no ground-truth domains to look up. "
                    "Set use_ground_truth_domains=False to classify domains at runtime instead."
                )
        else:
            print(f"  Classifying domains...")
            t_cls_start = time.time()
            active_domains = classify_domains(problem, model=classifier_model)
            t_cls_end = time.time()
            print(f"  Domains: {active_domains} ({t_cls_end - t_cls_start:.2f}s)")
            result["llm_calls"] += 1

        result["active_domains"] = active_domains

        LogicProblem      = build_hybrid_schema(active_domains)

        # STEP 2 — extract
        print(f"  Waiting for LLM extraction...")
        t_ext_start = time.time()
        extracted, unmatched_errors, ext_calls, extraction_raw = extract_fn(
            problem, active_domains, LogicProblem
        )
        t_ext_end = time.time()
        print(f"  Got extraction response ({t_ext_end - t_ext_start:.2f}s)")
        print(f"  Raw extraction response:\n{extraction_raw}")

        result["extraction_raw"]   = extraction_raw
        result["llm_calls"]       += ext_calls
        result["unmatched_errors"] = unmatched_errors

        if extracted is None:
            print(f"  [!] LLM Extraction Failed after {MAX_ATTEMPTS} attempts")
            result["formatted_output"] = f"ERROR: extraction failed after {MAX_ATTEMPTS} attempts"
            return result

        result["extracted"] = extracted

        # STEP 3 — Z3 solve
        # *** UNSAT RETRY DISABLED — comment back in to re-enable ***
        try:
            z3_result = z3_solve(extracted)
            # if z3_result["status"] == "unsat":
            #     extracted, z3_result, retry_calls, retry_unmatched = _handle_unsat_retry(
            #         problem, active_domains, extracted, z3_result, extract_fn
            #     )
            #     result["llm_calls"]             += retry_calls
            #     result["unmatched_errors"].extend(retry_unmatched)
            #     result["extracted"]              = extracted
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

        # STEP 4 — format (verbalization)
        # Deterministic, template-only prose for the verified Z3 result via the symbolic
        # explanation engine — no LLM. Gated behind `verbalize` (default off) so generation
        # callers (e.g. ollama_auto_sft) don't pay the MARCO enumeration cost. Imports are
        # local to avoid the explanation -> pipeline -> verbalization import cycle.
        if verbalize:
            try:
                from explanation import explain_problem, build_context
                import verbalization
                structs = explain_problem(extracted, problem)
                labels = build_context(extracted).cid_label
                result["formatted_output"] = verbalization.verbalize_struct(structs, labels)
            except Exception as e:
                result["formatted_output"] = f"ERROR: verbalization failed: {type(e).__name__}: {e}"
        else:
            result["formatted_output"] = None

    except RuntimeError as e:
        print(f"  [!] NS Pipeline failed due to timeout/connection error: {e}")
        result["formatted_output"] = f"ERROR: TIMEOUT_OR_CONNECTION_ERROR"
        return result

    return result