# ============================================================================
# SUPERSEDED — no longer used since the switch to algorithmic SFT generation.
# This is the old LLM-first approach: ask a model to invent a puzzle, then
# Z3-verify it. Replaced by algorithmic_sft_generator.py, which builds puzzles
# Z3-first (correct by construction) and only calls an LLM to paraphrase.
# Kept for reference only; not part of the active pipeline.
# ============================================================================
"""groq_auto_sft.py — generate → extract → Z3-verify → append, via Groq API.
   Run: python groq_auto_sft.py --target 2000

=== CONSTRAINED OUTPUT ATTEMPTS (all failed, documented for future reference) ===

PROBLEM: model intermittently returns empty responses, causing:
  JSONDecodeError: Expecting value: line 1 column 1 (char 0)

ROOT CAUSE (confirmed): max_tokens=16384 was being counted as a TPM reservation by Groq
  upfront, exhausting the free-tier 6000 TPM limit before any tokens were generated.
  Fix: reduced to max_tokens=4096 (gen) and max_tokens=8192 (extract).
  After this fix, llama-3.3-70b-versatile ran cleanly in testing with BATCH_SIZE=1.

ATTEMPT 1 — response_format={"type": "json_object"}
  Requires top-level output to be {}. Our prompts output [], so Groq's validator
  rejected it: BadRequestError 400 json_validate_failed, failed_generation: ''

ATTEMPT 2 — response_format={"type": "json_schema", "json_schema": ...} strict=True
  Switched model to openai/gpt-oss-120b (Groq lists GPT-OSS 120B as supporting
  strict json_schema). Changed both prompts to wrap output in {"items": [...]} and
  used schema {"type":"object","properties":{"items":{"type":"array"}},...}.
  Same error: BadRequestError 400 json_validate_failed, failed_generation: ''
  Likely causes: (a) "items" clashes with the JSON Schema "items" keyword, or
  (b) openai/gpt-oss-120b is not the correct Groq model ID — verify with:
    curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
  If retrying: use a non-conflicting key like "results" instead of "items".

ATTEMPT 3 — retry on empty response inside call() without response_format
  Removed response_format. Added retry loop (3 attempts, 5s sleep) checking if
  response content is empty. Avoids 400 errors but doesn't prevent empty responses.
  User reverted (git restore) — this is the current state.

CUSTOM SCHEMA NOTE: build_hybrid_schema() (HybridConstraint Pydantic schema) is NOT
  viable for json_schema mode — it uses anyOf (discriminated unions) and recursive
  $ref types, both unsupported by Groq strict mode.
"""
import json, re, time, uuid, argparse, hashlib, os, random
from datetime import datetime, timezone
from time import perf_counter
from openai import OpenAI
from validators import build_hybrid_schema
from pipeline import z3_solve

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

GENERATION_MODEL = "openai/gpt-oss-120b"
EXTRACTION_MODEL = "openai/gpt-oss-120b"
SFT_OUT          = "../data/sft_dataset.jsonl"
INCLUDE_EASY     = True
INCLUDE_MEDIUM   = True
INCLUDE_HARD     = True

_DIFFICULTY_SPECS = {
    "easy":   "  - Easy:   3 entities, 2-3 flat constraints, 1 domain.",
    "medium": "  - Medium: 4-5 entities, 4-6 constraints, occasional not/or wrapper, 1-2 domains.",
    "hard":   "  - Hard:   5-6 entities, 6-9 constraints, not/or/if_then wrappers used freely, 2-3 domains.",
}
BATCH_SIZE = 3

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
         "Ivan", "Judy", "Karl", "Liam", "Mona", "Nina", "Omar", "Priya",
         "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
         "Yara", "Zane", "Aria", "Theo", "Lena", "Cyrus"]

GEN_SYSTEM = f"""You are generating a benchmark of logic puzzles for an automated solver. The solver
encodes each puzzle into a fixed constraint vocabulary and verifies it with the Z3 SMT
solver. Therefore EVERY puzzle you generate MUST be fully expressible using ONLY the
primitives below. Do not invent relationships outside this vocabulary.

=== DOMAINS AND EXACT VOCABULARY ===

ORDERING — entities occupy distinct positions in slots numbered 1..N (1-INDEXED):
  - before(X, Y): X is in an earlier slot than Y
  - immediately_before(X, Y): X is exactly one slot before Y
  - adjacent(X, Y): |slot(X) - slot(Y)| == 1
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
  - is_in(X, k): X is in group k (1-indexed)

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
Generate a mix of {", ".join(d for d, on in [("easy", INCLUDE_EASY), ("medium", INCLUDE_MEDIUM), ("hard", INCLUDE_HARD)] if on)} puzzles.
{chr(10).join(_DIFFICULTY_SPECS[d] for d, on in [("easy", INCLUDE_EASY), ("medium", INCLUDE_MEDIUM), ("hard", INCLUDE_HARD)] if on)}

=== OUTPUT FORMAT ===
Return ONLY a JSON array, no prose. Each element:
{{
  "problem": "<full natural-language puzzle text including the question and the labeled answer choices>",
  "answer": "<correct choice label, e.g. 'C'>",
  "domains": ["<one or more of: ordering, knights_and_knaves, grouping>"],
  "difficulty": "<{'|'.join(d for d, on in [('easy', INCLUDE_EASY), ('medium', INCLUDE_MEDIUM), ('hard', INCLUDE_HARD)] if on)}>"
}}"""

EXTRACT_SYSTEM = """You are a logic puzzle constraint extractor. I will give you logic puzzles as a JSON array. For each puzzle, produce one JSON object. Output ONLY a JSON array of all results — no prose, no markdown, no code blocks.

=== OUTPUT FORMAT ===
Each element of the output array must be:
{
  "problem_text": "<exact problem string from input, including answer choices>",
  "answer": "<correct label from input, e.g. \"C\">",
  "active_domains": ["domain1"],
  "extracted_json": { ...see schema below... },
  "model_name": "groq"
}

active_domains and extracted_json are REAL objects/arrays, NOT strings.

=== EXTRACTION SCHEMA ===
extracted_json must follow this exact structure:
{
  "entities": ["name1", "name2", ...],
  "constraints": [<list of HybridConstraint>],
  "questions": [{
    "question_constraints": [],
    "answer_choices": [
      {"label": "A", "type": "<modality>", "constraints": [<list of HybridConstraint>]},
      ...
    ]
  }],
  "num_slots": <integer, only if domain includes ordering>,
  "num_groups": <integer, only if domain includes grouping>,
  "group_sizes": [<int>, ...], only if domain includes grouping>
}

=== CONSTRAINT TYPES — USE THESE EXACT type STRINGS ===

ORDERING:
  {"type": "before", "left": "X", "right": "Y"}
  {"type": "immediately_before", "left": "X", "right": "Y"}
  {"type": "adjacent", "left": "X", "right": "Y"}
  {"type": "slot_fixed", "entity": "X", "slot": <integer>}
  *** SLOTS ARE 1-INDEXED. slot 1 = first position. NEVER use slot 0. NEVER use strings for slot values. ***
  *** type string is "slot_fixed" — NEVER "slot_filled" or anything else ***

KNIGHTS AND KNAVES:
  {"type": "is_truth_teller", "entity": "X"}
  {"type": "is_deceiver", "entity": "X"}
  Every statement by a character MUST become exactly TWO if_then constraints:
    {"type": "if_then",
     "antecedent": {"type": "is_truth_teller", "entity": "Speaker"},
     "consequent": <the statement as a constraint>}
    {"type": "if_then",
     "antecedent": {"type": "is_deceiver", "entity": "Speaker"},
     "consequent": {"type": "not", "claim": <the statement as a constraint>}}

GROUPING:
  {"type": "same_group", "entities": ["X", "Y"]}
  {"type": "different_group", "entities": ["X", "Y"]}
  {"type": "exactly_n", "entities": [...], "n": <int>, "group": <int>}
  {"type": "is_in", "entity": "X", "group": <int>}
  *** GROUPS ARE 1-INDEXED. NEVER use 0. ***

LOGICAL WRAPPERS:
  {"type": "not", "claim": <constraint>}
  {"type": "and", "claims": [<constraint>, ...]}
  {"type": "or", "claims": [<constraint>, ...]}
  {"type": "if_then", "antecedent": <constraint>, "consequent": <constraint>}

ANSWER CHOICE MODALITIES — exactly one per choice:
  "must_be_true"   — true in every valid solution
  "could_be_true"  — true in at least one valid solution
  "must_be_false"  — false in every valid solution
  "could_be_false" — false in at least one valid solution

=== EXAMPLE — match this structure exactly ===
Input puzzle:
{"problem": "Alice, Bob, and Carol are standing in a line. Alice is before Bob. Bob is before Carol. Who is first?\\nA) Alice must be first  B) Bob must be first  C) Carol must be first", "answer": "A", "domains": ["ordering"]}

Correct output element:
{
  "problem_text": "Alice, Bob, and Carol are standing in a line. Alice is before Bob. Bob is before Carol. Who is first?\\nA) Alice must be first  B) Bob must be first  C) Carol must be first",
  "answer": "A",
  "active_domains": ["ordering"],
  "extracted_json": {
    "entities": ["Alice", "Bob", "Carol"],
    "constraints": [
      {"type": "before", "left": "Alice", "right": "Bob"},
      {"type": "before", "left": "Bob", "right": "Carol"}
    ],
    "questions": [{
      "question_constraints": [],
      "answer_choices": [
        {"label": "A", "type": "must_be_true", "constraints": [{"type": "slot_fixed", "entity": "Alice", "slot": 1}]},
        {"label": "B", "type": "must_be_true", "constraints": [{"type": "slot_fixed", "entity": "Bob", "slot": 1}]},
        {"label": "C", "type": "must_be_true", "constraints": [{"type": "slot_fixed", "entity": "Carol", "slot": 1}]}
      ]
    }],
    "num_slots": 3
  },
  "model_name": "groq"
}

=== RULES ===
1. Output ONLY a JSON array. No prose, no markdown fences, no explanation.
2. Include ALL answer choices from the problem — never drop any.
3. Every KK character statement produces exactly two if_then constraints.
4. Slots and groups are always integers, always 1-indexed, never 0, never strings.
5. Do not invent constraint types outside the vocabulary above.
6. num_slots, num_groups, group_sizes: include only for the relevant domain.

Output COMPACT minified JSON on a single line — no newlines, no indentation.

=== INPUT PUZZLES ==="""


def call(system, user, model, temp, max_tokens):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def parse(text):
    return json.loads(re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip())


def fp(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def is_quota_error(err):
    msg = str(err).upper()
    return "429" in msg or "RATE_LIMIT" in msg or "RESOURCE_EXHAUSTED" in msg


def build_seed(names):
    return (f"\n\nFor THIS batch: draw entity names only from {names}. "
            f"You may use 2-6 entities per puzzle. "
            f"Vary which letter is correct and the domain across the {BATCH_SIZE} puzzles.")


def call_gen(names):
    prompt = (f"Generate {BATCH_SIZE} puzzles now. Keep them solvable and rule-compliant."
              + build_seed(names))
    return parse(call(GEN_SYSTEM, prompt, model=GENERATION_MODEL, temp=0.7, max_tokens=4096))


def call_extract(puzzles):
    return parse(call(EXTRACT_SYSTEM, json.dumps(puzzles), model=EXTRACTION_MODEL, temp=0.0, max_tokens=8192))


def verify_one(row):
    domains   = row["active_domains"] if isinstance(row["active_domains"], list) else json.loads(row["active_domains"])
    ext_obj   = row["extracted_json"] if isinstance(row["extracted_json"], dict) else json.loads(row["extracted_json"])
    extracted = build_hybrid_schema(domains)(**ext_obj)
    res       = z3_solve(extracted)
    if res["status"] != "sat": return None
    verified = [l for l, v in res["question_results"][0].items() if v]
    if len(verified) != 1 or verified[0].upper() != row["answer"].upper(): return None
    return {"run_id": uuid.uuid4().hex, "problem_text": row["problem_text"],
            "active_domains": json.dumps(domains), "extracted_json": json.dumps(ext_obj),
            "model_name": row.get("model_name", "groq"),
            "timestamp": datetime.now(timezone.utc).isoformat()}


def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main(target):
    seen = set()
    try:
        for line in open(SFT_OUT):
            seen.add(fp(json.loads(line)["problem_text"]))
    except FileNotFoundError:
        pass

    kept          = len(seen)
    quota_strikes = 0
    start_time    = perf_counter()
    print(f"Starting — batch size: {BATCH_SIZE} | already saved: {kept} | target: {target}")

    while kept < target:
        names       = random.sample(NAMES, k=6)
        batch_start = perf_counter()
        try:
            puzzles       = call_gen(names);          time.sleep(2)
            extracted     = call_extract(puzzles);    time.sleep(2)
            quota_strikes = 0
        except Exception as err:
            if is_quota_error(err):
                quota_strikes += 1
                if quota_strikes >= 5:
                    print(f"Quota likely exhausted. {kept} saved. Re-run later.")
                    break
                print(f"Rate limited ({quota_strikes}/5) — waiting 60s. [{fmt_elapsed(perf_counter() - start_time)} elapsed]")
                print(f"({type(err).__name__}): {err}")
                time.sleep(60)
            else:
                print(f"Batch failed ({type(err).__name__}: {err}) — skipping. [{fmt_elapsed(perf_counter() - start_time)} elapsed]")
                time.sleep(5)
            continue

        kept_before = kept
        for row in extracted:
            h = fp(row.get("problem_text", ""))
            if h in seen:
                continue
            try:
                out = verify_one(row)
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
        print(f"batch: {batch_kept}/{BATCH_SIZE} kept  |  total: {kept}/{target}  |  batch {batch_elapsed}  |  elapsed {elapsed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    main(ap.parse_args().target)
