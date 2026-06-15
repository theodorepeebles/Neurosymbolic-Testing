"""ollama_auto_sft.py — generate → run_ns_pipeline → Z3-verify → append, all via Ollama.
   Run: python ollama_auto_sft.py --target 2000"""
import json, re, time, uuid, argparse, hashlib, random
from datetime import datetime, timezone
from time import perf_counter
from pipeline import ask_llm, extract_logic_problem, classify_domains, z3_solve, build_hybrid_schema

MODEL      = "gpt-oss:120b-cloud"
SFT_OUT    = "../data/sft_positives.jsonl"
BATCH_SIZE = 5

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
         "Ivan", "Judy", "Karl", "Liam", "Mona", "Nina", "Omar", "Priya",
         "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
         "Yara", "Zane", "Aria", "Theo", "Lena", "Cyrus"]

GEN_SYSTEM = """You are generating a benchmark of logic puzzles for an automated solver. The solver
encodes each puzzle into a fixed constraint vocabulary and verifies it with the Z3 SMT
solver. Therefore EVERY puzzle you generate MUST be fully expressible using ONLY the
primitives below. Do not invent relationships outside this vocabulary. 

=== DOMAINS AND EXACT VOCABULARY ===

ORDERING — entities occupy distinct positions in slots numbered 1..N (1-INDEXED):
  - before(X, Y): X is in an earlier slot than Y
  - immediately_before(X, Y): X is exactly one slot before Y
  - not_adjacent(X, Y): |slot(X) - slot(Y)| > 1
  - slot_fixed(X, k): X is in slot k (k is 1-indexed, 1..N)

KNIGHTS_AND_KNAVES — each character is a truth-teller (always true statements) or a
deceiver (always false statements):
  - is_truth_teller(X)
  - is_deceiver(X)
  - A character's STATEMENT is encoded as a biconditional: if the speaker is a
    truth-teller the statement is true; if a deceiver, false.

GROUPING — entities are partitioned into G groups of fixed sizes (groups 1-INDEXED):
  - same_group(X, Y, ...): listed entities share a group
  - different_group(X, Y, ...): listed entities are all in distinct groups
  - exactly_n(entities, n, group): exactly n of the listed entities are in group k

LOGICAL WRAPPERS — combine or negate any of the above:
  - not(claim)
  - and(claims), or(claims)
  - if_then(antecedent, consequent)

=== ANSWER CHOICE SEMANTICS ===
Each answer choice is a claim about the solution, tagged with one modality:
  - must_be_true   (true in EVERY valid solution)
  - could_be_true  (true in AT LEAST ONE valid solution)
  - must_be_false  (false in EVERY valid solution)
  - could_be_false (false in AT LEAST ONE valid solution)
Across the batch, vary which modalities appear.

=== HARD RULES (every puzzle must satisfy ALL) ===
1. Exactly ONE question per puzzle, with 3-4 labeled answer choices (A, B, C, D).
2. Exactly ONE answer choice is correct. The correct choice must be definitively
   resolvable — never "cannot be determined."
3. The puzzle constraints must be SATISFIABLE (at least one valid arrangement exists).
4. Every constraint in the puzzle must map to a primitive above. If you cannot express
   something with the vocabulary, do not use it.
5. Before outputting, mentally solve the puzzle to verify exactly one choice is correct.
   Vary WHICH letter is correct across puzzles — do not default to A.

=== DIFFICULTY ===
Generate only easy and medium puzzles.
  - Easy: 3 entities, 2-3 flat constraints, single domain.
  - Medium: 4-5 entities, 4-6 constraints, occasional not/or wrapper, single domain.

=== OUTPUT FORMAT ===
Return ONLY a JSON array, no prose. Each element:
{
  "problem": "<full natural-language puzzle text including the question and the labeled answer choices>",
  "answer": "<correct choice label, e.g. 'C'>",
  "domains": ["<one or more of: ordering, knights_and_knaves, grouping>"],
  "difficulty": "<easy|medium>"
}"""


def call_gen(names):
    """Generate a batch of puzzles via Ollama. System carries the full instructions;
    prompt supplies the per-batch seed so name variety scatters the puzzle space.

    fmt="json" = Ollama's basic JSON mode: guarantees the response is valid JSON,
    but does NOT enforce any particular schema — we just need a well-formed array here.
    Contrast with extraction inside run_ns_pipeline, where fmt=LogicProblem.model_json_schema()
    passes the full Pydantic schema so Ollama is constrained to that exact structure."""
    prompt = (f"Generate {BATCH_SIZE} puzzles now. Keep them solvable and rule-compliant."
              + build_seed(names))
    raw = ask_llm(prompt=prompt, system=GEN_SYSTEM, fmt="json", model=MODEL, think=False)
    return parse(raw)


def parse(text):
    return json.loads(re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip())


def fp(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def build_seed(names):
    return (f"\n\nFor THIS batch: draw entity names only from {names}. "
            f"You may use 2-6 entities per puzzle. "
            f"Vary which letter is correct and the domain across the {BATCH_SIZE} puzzles.")


def verify_one(puzzle, res):
    """Returns a ready-to-write dict if the pipeline result is a valid SFT positive,
    else None. Checks: sat, exactly one verified choice, matches labeled answer."""
    if res["z3_status"] != "sat":
        return None
    qr = res.get("question_results")
    if not qr:
        return None
    verified = [l for l, v in qr[0].items() if v]
    if len(verified) != 1 or verified[0].upper() != puzzle["answer"].upper():
        return None
    extracted = res["extracted"]
    return {
        "run_id":         uuid.uuid4().hex,
        "problem_text":   puzzle["problem"],
        "active_domains": json.dumps(res.get("active_domains", [])),
        "extracted_json": json.dumps(extracted.model_dump()),
        "model_name":     MODEL,
        "timestamp":      datetime.now(timezone.utc).isoformat()
    }


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def run_extract_pipeline(problem_text):
    """Classify → extract → Z3-verify using MODEL. No formatting, no UNSAT retry."""
    try:
        print(f"  Classifying domains...")
        t0 = perf_counter()
        active_domains = classify_domains(problem_text, model=MODEL)
        print(f"  Domains: {active_domains} ({perf_counter() - t0:.2f}s)")

        LogicProblem = build_hybrid_schema(active_domains)

        print(f"  Waiting for LLM extraction...")
        t1 = perf_counter()
        extracted, _, _ = extract_logic_problem(problem_text, active_domains, LogicProblem, model=MODEL)
        print(f"  Got extraction response ({perf_counter() - t1:.2f}s)")

        if extracted is None:
            return {"z3_status": None, "question_results": None, "extracted": None, "active_domains": active_domains}

        z3_result = z3_solve(extracted)
        print(f"  Z3 status         : {z3_result['status']}")
        print(f"  Question results  : {z3_result.get('question_results')}")

        return {
            "z3_status":        z3_result["status"],
            "question_results": z3_result.get("question_results"),
            "extracted":        extracted,
            "active_domains":   active_domains,
        }
    except RuntimeError as e:
        print(f"  [!] Pipeline error: {e}")
        return {"z3_status": None, "question_results": None, "extracted": None, "active_domains": []}


def main(target):
    seen = set()
    try:
        for line in open(SFT_OUT):
            seen.add(fp(json.loads(line)["problem_text"]))
    except FileNotFoundError:
        pass

    kept       = len(seen)
    start_time = perf_counter()
    print(f"Starting — batch size: {BATCH_SIZE} | already saved: {kept} | target: {target}")

    while kept < target:
        names       = random.sample(NAMES, k=6)
        batch_start = perf_counter()

        try:
            puzzles = call_gen(names)
            time.sleep(2)
        except Exception as err:
            print(f"Generation failed ({type(err).__name__}: {err}) — skipping. "
                  f"[{fmt_elapsed(perf_counter() - start_time)} elapsed]")
            time.sleep(5)
            continue

        kept_before = kept
        for puzzle in puzzles:
            problem_text = puzzle.get("problem", "") #probably means the other stuff generated by the gen prompt is not used here
            h = fp(problem_text)
            if h in seen:
                continue
            try:
                result = run_extract_pipeline(problem_text)
                out    = verify_one(puzzle, result)
            except Exception:
                out = None
            if out:
                with open(SFT_OUT, "a") as f:
                    f.write(json.dumps(out) + "\n")
                seen.add(h)
                kept += 1

        batch_kept    = kept - kept_before
        elapsed       = fmt_elapsed(perf_counter() - start_time)
        batch_elapsed = fmt_elapsed(perf_counter() - batch_start)
        print(f"batch: {batch_kept}/{BATCH_SIZE} kept  |  total: {kept}/{target}  |  "
              f"batch {batch_elapsed}  |  elapsed {elapsed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    main(ap.parse_args().target)
