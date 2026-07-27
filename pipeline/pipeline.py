from z3 import *
import requests
import json
import time
import operator
import re
from validators import MathProblem, validate_math_logic, OutputVariableMissing, UndefinedConstraintReference, OrphanedNullVariable, SelfReferentialConstraint

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

# For LLM retry upon schema validation failure
MAX_ATTEMPTS = 3

EXAMPLE_JSON = {
  "domain": "arithmetic", #doesnt do much but could be useful metadata with expansion
  "objective": "Find the total pay earned",  #doesnt really do anyhting at all
  "variables": [
    {"name": "hourly_rate", "value": 18, "unit": "dollars"},
    {"name": "hours_per_day", "value": 8, "unit": "hours"},
    {"name": "days", "value": 5, "unit": "days"},
    {"name": "daily_pay", "value": None, "unit": "dollars"},
    {"name": "total_pay", "value": None, "unit": "dollars"}
  ],
  "constraints": [
    {"target_variable": "daily_pay", "operand_1": "hourly_rate", "operator": "*", "operand_2": "hours_per_day"},
    {"target_variable": "total_pay", "operand_1": "daily_pay", "operator": "*", "operand_2": "days"}
  ],
  "output_variable": "total_pay"
}

# System prompt given to LLM
EXTRACTION_SYSTEM = f"""Extract math word problems into JSON. Return ONLY a JSON object with no explanation.

Use this exact structure:
{json.dumps(EXAMPLE_JSON, indent=2)}

Rules:
- Extract values EXACTLY as stated in the problem. Do NOT compute or simplify anything yourself — leave all calculation to the constraints
- Variable names must be snake_case
- Set value to null for ALL intermediate and output variables, not just the final one
- output_variable must exactly match one of the variable names
- operand_1 and operand_2 can be a variable name (string) or a number
- Operators: + - * / only
- CRITICAL: Never use a variable as both target_variable and an operand in the same constraint.
  Z3 solves all constraints at once as simultaneous equations, not step-by-step.
  Each variable may only appear as target_variable ONCE across all constraints.
  
  WRONG (will fail):
    {{"target_variable": "total", "operand_1": "servers", "operator": "*", "operand_2": "rps"}},
    {{"target_variable": "total", "operand_1": "total",   "operator": "*", "operand_2": "days"}}
  
  RIGHT (use intermediate variables):
    {{"target_variable": "total_per_second", "operand_1": "servers", "operator": "*", "operand_2": "rps"}},
    {{"target_variable": "total_with_days",  "operand_1": "total_per_second", "operator": "*", "operand_2": "days"}}"""



def extract_problem(word_problem: str, unsat_context: str = None) -> tuple[MathProblem | None, list, int]:
    """
    Calls ask_llm until LLM returns a valid MathProblem. Each reprompt is given a descriptive hint.
    Args:
        word_problem: ...
        unsat_context: the set of constraints that caused Z3 to deem the MathProblem unsat, formatted for LLM reprompt
    Returns:
        The extracted MathProblem or none, the errors for which no hint is written yet, # of attempts
    """
    unmatched_errors = []
    attempts_used = 0

    base_prompt = f"Extract this word problem:\n\n{word_problem}"
    prompt = f"{unsat_context}\n\n{base_prompt}" if unsat_context else base_prompt

    #for constrained output
    schema = MathProblem.model_json_schema()
    active_hints = set()

    for attempt in range(MAX_ATTEMPTS):

        raw = ask_llm(prompt=prompt, system=EXTRACTION_SYSTEM, fmt=schema)
        attempts_used += 1

        try:
            parsed = json.loads(raw)

            # Let custom logic check validate extracted problem
            math_prob = MathProblem(**parsed)
            validate_math_logic(math_prob)

            return math_prob, unmatched_errors, attempts_used
        
        except Exception as e:
            error_msg = str(e)
            hint = getattr(type(e), "hint", "")
            if hint:
                active_hints.add(hint) # Add to the growing list of hints based on what LLM has failed

            # Format all active hints into a single block
            hints_block = "\n".join(f"- {h}" for h in active_hints)

            #add unmatched hints to list
            if not hint:
                unmatched_errors.append({
                    "attempt": attempt + 1,
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                })
            
            print(f"  [Parse failed attempt {attempt+1}/{MAX_ATTEMPTS}: {e}]")
            
            # Surgical reprompting
            prompt = (
                f"Extract this word problem:\n\n{word_problem}\n\n"
                f"Your previous attempt returned this JSON:\n{raw}\n\n"
                f"It failed validation with this error: {error_msg}\n\n"
                f"Rules to remember:\n{hints_block}\n\n"
                f"Fix the error and return the corrected JSON."
            )

    return None, unmatched_errors, attempts_used
        

# Baseline LLM prompt
BASELINE_SYSTEM = (
    "Solve the math word problem. Show your work if needed, "
    "but your final answer must appear on the last line in this exact format:\n"
    "ANSWER: <number>\n"
    "Example: ANSWER: 42"
)

def baseline_llm_solve(problem: str, model: str = "qwen3:8b", think: bool = False) -> tuple[float | None, str | None, str]:
    """
    Sends the problem directly to the LLM and parses a numeric answer.
    Returns (answer, None) on success, (None, error_string) on parse failure.
    model and think params allow ablation testing against different configurations.
    """
    try:
        raw = ask_llm(prompt=problem, system=BASELINE_SYSTEM, model=model, think=think)
    except RuntimeError as e:
        # Catch the timeout right here
        return None, f"TIMEOUT_OR_CONNECTION_ERROR: {e}", ""
    cleaned = raw.replace("$", "").replace(",", "")


    # Look for the marker first
    match = re.search(r"ANSWER:\s*\$?([\d,]+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if match:
        return float(match.group(1)), None, raw
    
    # Fallback: last number in the response
    matches = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", cleaned)
    if matches:
        return float(matches[-1].replace(",", "")), "marker missing, used last number", raw
    return None, f"No number found in: '{raw}'", raw
    
# ── DISPATCH: route operation to the right Z3 verifier ────────────────────────

def z3_solve(extracted: MathProblem) -> dict:
    solver = Solver()

    # 1. Create a Z3 variable for every extracted variable
    z3_vars = {v.name: Real(v.name) for v in extracted.variables}

    # 2. Pin the known values
    for v in extracted.variables:
        if v.value is not None:
            solver.add(z3_vars[v.name] == v.value)

    # 3. Map string operators to safe Python math operations
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }

    # 4. Process the 2-operand constraints, tracked for unsat explanation potential
    constraint_labels = {}

    for i, c in enumerate(extracted.constraints):
        # Check if the operand is a variable name (string) or a hardcoded number (float)
        val1 = z3_vars[c.operand_1] if isinstance(c.operand_1, str) else RealVal(c.operand_1)
        val2 = z3_vars[c.operand_2] if isinstance(c.operand_2, str) else RealVal(c.operand_2)
        
        # Safely do the math: e.g., operator.add(val1, val2)
        math_result = ops[c.operator](val1, val2)

        label = Bool(f"c_{i}_{c.target_variable}")
        constraint_labels[str(label)] = c          # store label → constraint for lookup later

        
        # Add it to the solver: e.g., total = a + b
        solver.assert_and_track(z3_vars[c.target_variable] == math_result, label)

    status = solver.check()

    if status == sat:
        model = solver.model()
        z3_ans = model.eval(z3_vars[extracted.output_variable])

        # Safely extract the exact number depending on how Z3 stored it
        try:
            answer = float(z3_ans.as_fraction())
        except (AttributeError, TypeError):
            answer = float(z3_ans.as_decimal(10).replace("?", "")) #irrational numbers case
        
        return {"status": "sat", "answer": answer}
    elif status == unsat:
        core = solver.unsat_core()
        conflicting = [constraint_labels[str(label)] for label in core if str(label) in constraint_labels]
        core_summary = "\n".join(
            f"  {c.target_variable} = {c.operand_1} {c.operator} {c.operand_2}"
            for c in conflicting
        )
        return {"status": "unsat", "answer": None, "unsat_core": core_summary}   # escalate to unsat recheck
    else:
        return {"status": "unknown", "answer": None} # Z3 timed out
    
def _handle_unsat_retry(problem: str, extracted: MathProblem, z3_result: dict) -> tuple[MathProblem, dict, int, list]:
    """
    Called when Z3 returns unsat. Builds diagnostic context and retries extraction.
    Returns (extracted, z3_result, llm_calls_used, unmatched_errors)
    """
    print(f"  [UNSAT] Retrying extraction...")

    var_summary = "\n".join(
        f"  {v.name} = {v.value}" for v in extracted.variables
    )
    constraint_summary = "\n".join(
        f"  {c.target_variable} = {c.operand_1} {c.operator} {c.operand_2}"
        for c in extracted.constraints
    )

    core_info = z3_result.get("unsat_core", "")

    unsat_context = (
        f"Your previous extraction produced constraints that Z3 found unsatisfiable.\n\n"
        f"Variables:\n{var_summary}\n\n"
        f"Constraints:\n{constraint_summary}\n\n"
        f"Z3 identified these specific constraints as contradictory:\n{core_info}\n\n"
        f"Re-read the problem carefully and fix the extraction."
    )

    extracted_retry, retry_unmatched, retry_calls = extract_problem(problem, unsat_context=unsat_context)

    if extracted_retry is not None:
        z3_result_retry = z3_solve(extracted_retry)
        if z3_result_retry["status"] == "unsat":
            print(f"  [UNSAT occurred twice]")
        return extracted_retry, z3_result_retry, retry_calls, retry_unmatched
    else:
        # retry extraction failed entirely — return original extracted with unsat result
        return extracted, {"status": "unsat", "answer": None}, retry_calls, retry_unmatched

def run_ns_pipeline(problem: str) -> dict:
    """
    Runs the full NS pipeline for a single problem.
    Returns a dict with: extracted, z3_answer, z3_status, formatted_output, llm_calls, unmatched_errors
    """
    result = {
        "extracted": None,
        "z3_answer": None,
        "z3_status": None,
        "formatted_output": None,
        "llm_calls": 0,
        "unmatched_errors": []
    }
    try:
        # STEP 1 — extract
        print(f"  Waiting for LLM extraction...")
        t_ext_start = time.time()
        extracted, unmatched_errors, ext_calls = extract_problem(problem)
        t_ext_end = time.time()
        print(f"  Got extraction response ({t_ext_end - t_ext_start:.2f}s)")

        result["llm_calls"] += ext_calls
        result["unmatched_errors"] = unmatched_errors

        if extracted is None:
            print(f"  [!] LLM Extraction Failed after {MAX_ATTEMPTS} attempts")
            result["formatted_output"] = f"ERROR: extraction failed after {MAX_ATTEMPTS} attempts"
            return result
        
        result["extracted"] = extracted

        # STEP 2 — Z3 solve, with one UNSAT retry
        try:
            z3_result = z3_solve(extracted)
            if z3_result["status"] == "unsat":
                extracted, z3_result, retry_calls, retry_unmatched = _handle_unsat_retry(problem, extracted, z3_result)
                result["llm_calls"] += retry_calls
                result["unmatched_errors"].extend(retry_unmatched)
                result["extracted"] = extracted
        except Exception as e:
            result["formatted_output"] = f"ERROR: {type(e).__name__}: {e}"
            return result

        result["z3_answer"] = z3_result["answer"]
        result["z3_status"] = z3_result["status"]
        
        print(f"  Raw Z3 Answer     : {z3_result['answer']}")

        if z3_result["status"] != "sat":
            result["formatted_output"] = f"ERROR: Z3 returned {z3_result['status']}"
            return result

        # STEP 3 — format
        print(f"  Waiting for LLM formatting...")
        t_fmt_start = time.time()
        formatted = ask_llm(
            prompt=f"Answer this word problem in one sentence. State only the final answer, no working or reasoning. The answer is {z3_result['answer']}.\nWord problem: {problem}",
            system="You answer math word problems in one sentence. State only the final answer with appropriate units."
        )
        t_fmt_end = time.time()
        print(f"  Got formatting response ({t_fmt_end - t_fmt_start:.2f}s)")

        result["llm_calls"] += 1
        result["formatted_output"] = formatted

    # CATCH ANY LLM TIMEOUTS FROM STEP 1, 2, OR 3 HERE
    except RuntimeError as e:
        print(f"  [!] NS Pipeline failed due to timeout/connection error: {e}")
        result["formatted_output"] = f"ERROR: TIMEOUT_OR_CONNECTION_ERROR"
        return result
    
    return result
