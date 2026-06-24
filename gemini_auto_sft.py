"""auto_sft.py — generate → extract → Z3-verify → append, all via Gemini API.
   Run: python auto_sft.py --target 2000"""
import json, re, time, uuid, argparse, hashlib, os, random
from datetime import datetime, timezone
from time import perf_counter
from google import genai
from google.genai import types
from validators import build_hybrid_schema
from pipeline import z3_solve

client           = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GENERATION_MODEL = "gemini-3.5-flash"
EXTRACTION_MODEL = "gemini-3.1-flash-lite"
SFT_OUT          = "../data/sft_positives.jsonl"  # output file; also doubles as the resume checkpoint
BATCH_SIZE     = 3
INCLUDE_EASY   = False
INCLUDE_MEDIUM = True
INCLUDE_HARD   = False

_DIFFICULTY_SPECS = {
    "easy":   "  - Easy:   3 entities, 2-3 flat constraints, 1 domain.",
    "medium": "  - Medium: 4-5 entities, 4-6 constraints, occasional not/or wrapper, 1-2 domains.",
    "hard":   "  - Hard:   5-6 entities, 6-9 constraints, not/or/if_then wrappers used freely, 2-3 domains.",
}

# Pool of entity names. We sample from this each batch so the model doesn't
# default to Alice/Bob/Carol every time — surface-form variety helps the
# extraction model generalize instead of memorizing specific names.
NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
         "Ivan", "Judy", "Karl", "Liam", "Mona", "Nina", "Omar", "Priya",
         "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
         "Yara", "Zane", "Aria", "Theo", "Lena", "Cyrus"]

GEN_PROMPT = f"""You are generating a benchmark of logic puzzles for an automated solver. The solver
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
}}

Generate {BATCH_SIZE} puzzles now. Keep them solvable and rule-compliant."""

EXTRACT_PROMPT = """You are a logic puzzle constraint extractor. I will give you logic puzzles as a JSON array. For each puzzle, produce one JSON object. Output ONLY a JSON array of all results — no prose, no markdown, no code blocks.

=== OUTPUT FORMAT ===
Each element of the output array must be:
{
  "problem_text": "<exact problem string from input, including answer choices>",
  "answer": "<correct label from input, e.g. \"C\">",
  "active_domains": ["domain1"],
  "extracted_json": { ...see schema below... },
  "model_name": "gemini"
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
{"problem": "Alice, Bob, and Carol are standing in a line. Alice is before Bob. Bob is before Carol. Who is first?\nA) Alice must be first  B) Bob must be first  C) Carol must be first", "answer": "A", "domains": ["ordering"]}

Correct output element:
{
  "problem_text": "Alice, Bob, and Carol are standing in a line. Alice is before Bob. Bob is before Carol. Who is first?\nA) Alice must be first  B) Bob must be first  C) Carol must be first",
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
  "model_name": "gemini"
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


def call(prompt, model):
    """Send one prompt to Gemini and return the raw text response.
    max_output_tokens is set high so a full batch's JSON isn't truncated
    mid-array — truncation would make parse() throw and lose the whole batch."""
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=64000,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            response_mime_type="application/json"
        )
    ).text

def parse(text):
    """Turn the model's text response into Python objects.
    The regex strips any ```json ... ``` markdown fences the model adds
    despite being told not to — json.loads() would choke on the backticks.
    Then we parse the cleaned string into a list/dict."""
    return json.loads(re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip())

def fp(text):
    """Fingerprint: a short normalized hash of a problem's text, used as a
    fast O(1) key for duplicate detection in the `seen` set. Lowercasing and
    collapsing whitespace first means trivial formatting differences don't
    register as distinct problems. This is the hard guarantee that no exact
    duplicate ever gets written to the output file."""
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()

def is_quota_error(err):
    """True if the exception looks like a rate-limit / quota error (HTTP 429).
    Used to decide whether to wait-and-retry vs. just skip the batch."""
    msg = str(err).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg

def build_seed(names):
    """Build the per-batch suffix appended to the generation prompt.
    Randomized names scatter each batch into a different region of the puzzle
    space, reducing wasted duplicate generation without growing the prompt."""
    return (f"\n\nFor THIS batch: draw entity names only from {names}. "
            f"You may use 2-6 entities per puzzle. "
            f"Vary which letter is correct and the domain across the {BATCH_SIZE} puzzles.")

def verify_one(row):
    """Z3 filter — the source of truth. Returns a ready-to-write dict if the
    extraction is valid, else None. Checks three things:
      1. the constraints are satisfiable (status == sat),
      2. exactly one answer choice resolves true,
      3. that choice matches the answer label the generator claimed.
    Anything failing these is dropped — bad extractions never reach the file."""
    domains   = row["active_domains"] if isinstance(row["active_domains"], list) else json.loads(row["active_domains"])
    ext_obj   = row["extracted_json"] if isinstance(row["extracted_json"], dict) else json.loads(row["extracted_json"])
    extracted = build_hybrid_schema(domains)(**ext_obj)  # build per-domain Pydantic schema, then validate
    res       = z3_solve(extracted)
    if res["status"] != "sat": return None
    verified = [l for l, v in res["question_results"][0].items() if v]
    if len(verified) != 1 or verified[0].upper() != row["answer"].upper(): return None
    return {"run_id": uuid.uuid4().hex, "problem_text": row["problem_text"],
            "active_domains": json.dumps(domains), "extracted_json": json.dumps(ext_obj),
            "model_name": row.get("model_name", "gemini"),
            "timestamp": datetime.now(timezone.utc).isoformat()}

def fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"

def main(target):
    # `seen` = hashes of every problem already saved (dedup guarantee).
    seen = set()

    # Resume support: reload hashes from any existing output so a re-run
    # continues where it left off instead of regenerating from scratch.
    try:
        for line in open(SFT_OUT):
            seen.add(fp(json.loads(line)["problem_text"]))
    except FileNotFoundError:
        pass

    kept          = len(seen)  # count toward target includes already-saved rows
    quota_strikes = 0          # consecutive rate-limit hits; 5 in a row => stop
    start_time    = perf_counter()

    print(f"Starting — batch size: {BATCH_SIZE} | already saved: {kept} | target: {target}")

    while kept < target:
        names = random.sample(NAMES, k=6)
        batch_start = perf_counter()
        gen_prompt = GEN_PROMPT + build_seed(names)
        try:
            puzzles        = parse(call(gen_prompt, GENERATION_MODEL));               time.sleep(5)   # step 1: generate NL puzzles
            extract_prompt = EXTRACT_PROMPT + "\n" + json.dumps(puzzles)                               # puzzles appended after the header
            extracted      = parse(call(extract_prompt, EXTRACTION_MODEL));            time.sleep(5)   # step 2: extract to constraint JSON
            quota_strikes  = 0                                                            # clean batch — reset strike counter
        except Exception as err:
            # Rate-limit: wait and retry. A per-minute limit clears in 60s;
            # 5 strikes in a row means the daily quota is gone, so stop cleanly.
            if is_quota_error(err):
                quota_strikes += 1
                if quota_strikes >= 5:
                    print(f"Daily quota likely exhausted. {kept} saved. Re-run tomorrow.")
                    break
                print(f"Rate limited ({quota_strikes}/5) — waiting 60s. [{fmt_elapsed(perf_counter() - start_time)} elapsed]")
                time.sleep(60)
            else:
                # Any other error (bad JSON, truncation, network) — skip this batch.
                print(f"Batch failed ({type(err).__name__}: {err}) — skipping. [{fmt_elapsed(perf_counter() - start_time)} elapsed]")
                time.sleep(5)
            continue

        # Verify and append each extracted puzzle individually.
        kept_before = kept
        for row in extracted:
            h = fp(row.get("problem_text", ""))
            if h in seen:               # already have this exact problem — skip
                continue
            try:
                out = verify_one(row)   # Z3 check
            except Exception:
                out = None              # malformed row — treat as a drop
            if out:
                with open(SFT_OUT, "a") as f:  # append-only; never rewrites the file
                    f.write(json.dumps(out) + "\n")
                seen.add(h)
                kept += 1
        batch_kept = kept - kept_before
        elapsed = fmt_elapsed(perf_counter() - start_time)
        batch_elapsed = fmt_elapsed(perf_counter() - batch_start)
        print(f"batch: {batch_kept}/{BATCH_SIZE} kept  |  total: {kept}/{target}  |  batch {batch_elapsed}  |  elapsed {elapsed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    main(ap.parse_args().target)