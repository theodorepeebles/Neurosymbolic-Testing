"""
Constraint -> source-text attribution for extracted logic problems.

Maps each extracted constraint to the span of problem_text it came from, using a
two-tier strategy:

  Option A (primary)  : the extraction LLM emits `evidence_text` per constraint (the
                        exact source sentence). Verified as a real substring of
                        problem_text and converted to its [start, end] span here. Fast,
                        no extra inference.
  Option B (fallback) : when evidence_text is missing/None or fails substring
                        verification, run entity-filtered BM25 over the problem's
                        sentences. Deterministic, no LLM.

Granularity matches the z3 unsat-core trackers (`track_c_{i}` in pipeline.z3_solve):
attribution is per TOP-LEVEL constraint object. A logical wrapper (if_then/and/or/not)
is one clue -> one span; its nested children are NOT attributed separately.

Constraint IDs (keys in the returned dicts):
  global  constraints[i]                          -> "c_{i}"        (matches track_c_{i})
  question[n].question_constraints[i]             -> "q{n}.qc_{i}"
  question[n].answer_choices[*].constraints[j]    -> "q{n}.{label}_{j}"

Operates on constraint *dicts* (LogicProblem.model_dump() output), like eval_metrics.
"""

import re

from rank_bm25 import BM25Okapi


# --- Constraint -> query / entities (type-dispatch, mirrors pipeline/eval_metrics) ---

def constraint_to_query(c: dict) -> str:
    """Render a constraint dict as the search query most likely to overlap its source
    sentence: entity names + relation vocabulary.

    Keep in sync with algorithmic_sft_generator._constraint_to_english — same wording so
    BM25 term overlap matches the vocabulary the paraphrase LLM was shown. (Copied rather
    than imported: the generator pulls heavy gemini/genai deps we don't want at test time.)
    """
    t = c.get("type", "")
    if   t == "slot_fixed":          return f"{c['entity']} is in slot {c['slot']}"
    elif t == "before":              return f"{c['left']} is before {c['right']}"
    elif t == "immediately_before":  return f"{c['left']} is immediately before {c['right']}"
    elif t == "adjacent":            return f"{c['left']} and {c['right']} are adjacent"
    elif t == "is_in":               return f"{c['entity']} is in group {c['group']}"
    elif t == "is_truth_teller":     return f"{c['entity']} is a truth-teller (knight)"
    elif t == "is_deceiver":         return f"{c['entity']} is a deceiver (knave)"
    elif t == "same_group":          return f"{' and '.join(c['entities'])} are in the same group"
    elif t == "different_group":     return f"{' and '.join(c['entities'])} are in different groups"
    elif t == "exactly_n":
        ents = ", ".join(c["entities"])
        return f"exactly {c['n']} of [{ents}] are in group {c.get('group', '?')}"
    elif t == "not":
        return f"it is not the case that: {constraint_to_query(c['claim'])}"
    elif t == "if_then":
        return (f"if {constraint_to_query(c['antecedent'])}, "
                f"then {constraint_to_query(c['consequent'])}")
    elif t in ("and", "or"):
        sep   = " and " if t == "and" else " or "
        parts = sep.join(constraint_to_query(cl) for cl in c["claims"])
        return f"({parts})"
    return str(c)


def constraint_entities(c: dict) -> list[str]:
    """All entity names referenced by a constraint, recursing into logical wrappers.
    Used as a hard filter before BM25 (only sentences containing every entity score)."""
    if not isinstance(c, dict):
        return []
    t = c.get("type", "")
    if t in ("before", "immediately_before", "adjacent"):
        return [c["left"], c["right"]]
    if t in ("slot_fixed", "is_truth_teller", "is_deceiver", "is_in"):
        return [c["entity"]]
    if t in ("same_group", "different_group", "exactly_n"):
        return list(c.get("entities", []))
    if t == "not":
        return constraint_entities(c.get("claim", {}))
    if t == "if_then":
        return constraint_entities(c.get("antecedent", {})) + constraint_entities(c.get("consequent", {}))
    if t in ("and", "or"):
        out = []
        for cl in c.get("claims", []) or []:
            out.extend(constraint_entities(cl))
        return out
    return []


# --- Sentence segmentation ---

# Split on sentence boundaries (. ! ?), semicolons, and ", and" joins of compound clauses.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=;)\s*|(?<=,)\s+(?=and\s)')


def segment_sentences(problem_text: str) -> list[tuple[str, int, int]]:
    """Split problem_text into (sentence_text, char_start, char_end) tuples.
    Spans index into the ORIGINAL problem_text; empty/whitespace fragments dropped."""
    text = problem_text or ""
    spans, last = [], 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((text[last:m.start()], last, m.start()))
        last = m.end()
    spans.append((text[last:], last, len(text)))
    # strip() may trim leading whitespace; recompute start so the span still aligns
    out = []
    for s, start, end in spans:
        stripped = s.strip()
        if not stripped:
            continue
        offset = s.index(stripped)
        out.append((stripped, start + offset, start + offset + len(stripped)))
    return out


# --- Attribution ---

def attribute_constraint(
    c: dict,
    problem_text: str,
    sentences: list[tuple[str, int, int]],
    bm25_threshold: float = 0.3,
) -> tuple[tuple[int, int] | None, str]:
    """Attribute one constraint to a (start, end) span of problem_text.

    Returns (span_or_None, method) where method is one of:
      "option_a"      : LLM-supplied evidence_text verified as a substring
      "bm25_fallback" : entity-filtered BM25 match over sentences
      "unattributed"  : nothing scored above threshold
    """
    # Option A: verify the LLM-supplied evidence_text is a real substring, returning its span.
    evidence = c.get("evidence_text")
    if evidence:
        needle = evidence.lower().strip(".,; ")
        idx = problem_text.lower().find(needle)
        if idx != -1 and needle:
            return (idx, idx + len(needle)), "option_a"

    # Option B: entity filter + BM25.
    if not sentences:
        return None, "unattributed"

    def span(i):
        _, start, end = sentences[i]
        return (start, end), "bm25_fallback"

    # Hard filter: sentences containing every entity in the constraint.
    entity_names = [n.lower() for n in constraint_entities(c)]
    candidate_idx = [
        i for i, (text, _, _) in enumerate(sentences)
        if all(name in text.lower() for name in entity_names)
    ]

    # A unique entity match is unambiguous on its own — take it without a BM25 threshold.
    # (BM25 IDF is degenerate on a 1-document corpus, so don't gate the obvious case on it.)
    if len(candidate_idx) == 1:
        return span(candidate_idx[0])

    # Otherwise let BM25 discriminate. Score over the FULL sentence corpus so IDF is stable,
    # then pick the best-scoring sentence among the entity candidates (or all, if none match).
    query_tokens = constraint_to_query(c).lower().split()
    corpus = [text.lower().split() for (text, _, _) in sentences]
    scores = BM25Okapi(corpus).get_scores(query_tokens)

    pool = candidate_idx or list(range(len(sentences)))
    best = max(pool, key=lambda i: scores[i])
    if scores[best] >= bm25_threshold:
        return span(best)
    return None, "unattributed"


def build_attribution(extracted_dict: dict, problem_text: str) -> tuple[dict, dict]:
    """Attribute every constraint in an extracted LogicProblem dict.

    Returns (methods, spans):
      methods : {cid: "option_a"|"bm25_fallback"|"unattributed"}  (always populated)
      spans   : {cid: [start, end]}                                (only when a span found)

    All regions are attributed in one pass; keys are namespaced by region (see module
    docstring). Global clue ids (c_{i}) line up 1:1 with z3's track_c_{i}.
    """
    sentences = segment_sentences(problem_text)
    methods: dict[str, str] = {}
    spans: dict[str, list[int]] = {}

    def record(cid: str, c: dict) -> None:
        span, method = attribute_constraint(c, problem_text, sentences)
        methods[cid] = method
        if span is not None:
            spans[cid] = [span[0], span[1]]

    for i, c in enumerate(extracted_dict.get("constraints", []) or []):
        record(f"c_{i}", c)

    for n, q in enumerate(extracted_dict.get("questions", []) or []):
        for i, c in enumerate(q.get("question_constraints", []) or []):
            record(f"q{n}.qc_{i}", c)
        for ch in q.get("answer_choices", []) or []:
            label = ch.get("label", "?")
            for j, c in enumerate(ch.get("constraints", []) or []):
                record(f"q{n}.{label}_{j}", c)

    return methods, spans
