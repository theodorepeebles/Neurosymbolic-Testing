"""
Interactive REPL for the neurosymbolic logic-puzzle pipeline.

Loads one puzzle (by run_id from sft_test.db, gold extraction by default) and answers three
query types about it via the Stage-1 Z3 explanation primitives, rendered through verbalization:

  whynot <var> <val>   "Why can't X = V?"  -> assert X=V, expect UNSAT -> MUS refutation
  can    <var> <val>   "Can X = V?"        -> assert X=V, SAT -> witness / UNSAT -> refutation
  forces <var>         "What forces X?"    -> report X's forced/free binding in the solution

Variables are prefixed Z3 names: slot_<entity> (ordering), group_<entity> (grouping),
kk_<entity> (knights & knaves). Values are validated against each variable's domain.

Two interactive input modes, toggled at runtime with `mode`:
  structured (default) — type the commands above directly, or `form` for a guided prompt. No LLM.
  free-text (--freetext) — type a plain-English question and qwen3:8b (via Ollama) maps it to
      {variable, value, query_type}; the REPL echoes the interpreted command, then answers, or
      says "Cannot answer question" if it doesn't fit. This is the Phase 2 seam (parse_freetext),
      using the BASE model — the fine-tuned extractor (PHASE2_REPL_FREETEXT.txt) isn't built yet.

Usage:
    python logic/repl.py <run_id>                # interactive, structured mode (random run_id if omitted)
    python logic/repl.py <run_id> --freetext     # interactive, free-text (LLM) mode
    python logic/repl.py <run_id> --source extracted                          # query model's extraction
    python logic/repl.py <run_id> --var slot_Alice --value 2 --query whynot   # one-shot, no stdin
"""

import argparse
import json
import sys

# Reuse the battle-tested DB loaders from the explanation debugger.
from explanation_debug import fetch_run, pick_random_run_id, _loads, DEFAULT_DB
from validators import build_hybrid_schema
from pipeline import ask_llm
from prompts import FREETEXT_REPL_SYSTEM, build_freetext_repl_prompt
import explanation
import verbalization

QUERY_TYPES = ("whynot", "can", "forces")

# Base model used for free-text mode (no fine-tuned REPL extractor yet — see module docstring).
# Mirrors run.py's BASELINE_MODEL; served locally by Ollama.
FREETEXT_MODEL = "qwen3:8b"

# Sentinel: parse_freetext returns this when the model reports the question doesn't fit
# (query_type=cannot_answer) or the mapping fails validation — distinct from None (not free text)
# and from a (var, val, qtype) tuple (a usable query).
CANNOT_ANSWER = object()


# ─────────────────────────────────────────────────────────────────────────────
# Loading + variable/value handling
# ─────────────────────────────────────────────────────────────────────────────

def load_puzzle(db_path: str, run_id: str, source: str = "gold"):
    """Rebuild the LogicProblem + problem_text for a stored run (gold or model extraction)."""
    row = fetch_run(db_path, run_id)
    problem_text = row["problem_text"]
    active_domains = _loads(row["active_domains"])
    col = "expected_json" if source == "gold" else "extracted_json"
    source_json = _loads(row[col])
    if source_json is None:
        sys.exit(f"Row has no {source} extraction (column is NULL) for run_id={run_id}")
    lp = build_hybrid_schema(active_domains)(**source_json)
    return lp, problem_text


def _domain_high(var: str, lp):
    prefix = var.split("_", 1)[0]
    if prefix == "slot":
        return getattr(lp, "num_slots", None)
    if prefix == "group":
        return getattr(lp, "num_groups", None)
    return None


def parse_value(var: str, raw: str, lp):
    """Parse + validate a value for a variable against its domain. kk_* -> bool, slot/group -> int."""
    prefix = var.split("_", 1)[0]
    if prefix == "kk":
        low = raw.strip().lower()
        if low in ("true", "t", "1", "truth-teller", "knight", "yes"):
            return True
        if low in ("false", "f", "0", "deceiver", "knave", "no"):
            return False
        raise ValueError(f"{var} is a knights/knaves variable — value must be true/false")
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{var} takes an integer, got {raw!r}")
    hi = _domain_high(var, lp)
    if hi is not None and not (1 <= n <= hi):
        raise ValueError(f"{var} must be in 1..{hi}")
    return n


def list_variables(ctx, lp) -> str:
    by_prefix = {"slot": [], "group": [], "kk": []}
    for name in ctx.zvars:
        by_prefix.setdefault(name.split("_", 1)[0], []).append(name)
    lines = []
    if by_prefix.get("slot"):
        lines.append(f"  ordering   (slot,  1..{getattr(lp, 'num_slots', '?')}) : "
                     + ", ".join(sorted(by_prefix["slot"])))
    if by_prefix.get("group"):
        lines.append(f"  grouping   (group, 1..{getattr(lp, 'num_groups', '?')}) : "
                     + ", ".join(sorted(by_prefix["group"])))
    if by_prefix.get("kk"):
        lines.append("  knaves     (kk,    true/false) : " + ", ".join(sorted(by_prefix["kk"])))
    return "\n".join(lines) or "  (no variables)"


# ─────────────────────────────────────────────────────────────────────────────
# Query dispatch -> verbalization (the single prose path)
# ─────────────────────────────────────────────────────────────────────────────

def run_query(ctx, q_index, span_index, labels, var, val, query_type):
    """Dispatch one structured query and return (prose, mcs_prose)."""
    if query_type in ("whynot", "can"):
        fn = explanation.query_why_not if query_type == "whynot" else explanation.query_can
        w = fn(ctx, q_index, span_index, var, val)
        vinput = verbalization.vinput_for_wrong(
            w, labels, proposition_override=verbalization.describe_binding(var, val)
        )
    elif query_type == "forces":
        c = explanation.query_what_forces(ctx, q_index, span_index, var)
        if c is None:
            return (f"{var} does not appear in this question's solution "
                    f"(or the puzzle is unsatisfiable)."), ""
        if c.forced_bindings:
            v, fval, cids = c.forced_bindings[0]
            vinput = verbalization.vinput_for_forced(v, fval, [labels.get(cid, cid) for cid in cids])
        else:
            v, fval = c.free_bindings[0]
            vinput = verbalization.vinput_for_free(v, fval)
    else:
        raise ValueError(f"query_type must be one of {QUERY_TYPES}")

    vo = verbalization.render(vinput)
    return vo.prose, vo.mcs_prose


# ─────────────────────────────────────────────────────────────────────────────
# Input parsing — Phase 1 structured form + Phase 2 seam
# ─────────────────────────────────────────────────────────────────────────────

def _freetext_schema(ctx) -> dict:
    """JSON schema handed to Ollama as a decoding grammar (fmt=schema). Constraining
    `variable` to this puzzle's real variable names makes hallucinated names impossible;
    `cannot_answer` is the explicit escape hatch the grammar would otherwise deny."""
    return {
        "type": "object",
        "properties": {
            "query_type": {"type": "string", "enum": list(QUERY_TYPES) + ["cannot_answer"]},
            "variable": {"type": "string", "enum": sorted(ctx.zvars)},
            "value": {"type": ["string", "null"]},
        },
        "required": ["query_type", "variable", "value"],
    }


def parse_freetext(text: str, ctx, lp):
    """Phase 2: map a free-text question -> (variable, value, query_type) with the base
    qwen3:8b model under a JSON-schema grammar (no fine-tuned extractor yet).

    Returns:
      (var, val, qtype)  a validated, runnable query
      CANNOT_ANSWER      the model declined (query_type=cannot_answer) or the mapping
                         failed validation (bad variable / out-of-range value / bad JSON)
    Raises RuntimeError if Ollama is unreachable (surfaced by the interactive loop)."""
    variables_block = list_variables(ctx, lp)
    prompt = build_freetext_repl_prompt(text, variables_block, text)
    raw = ask_llm(
        prompt=prompt, system=FREETEXT_REPL_SYSTEM,
        fmt=_freetext_schema(ctx), model=FREETEXT_MODEL, is_extraction=True,
    )

    try:
        obj = json.loads(raw)
        qtype = obj["query_type"]
        var = obj["variable"]
        if qtype not in QUERY_TYPES or var not in ctx.zvars:
            return CANNOT_ANSWER  # cannot_answer, or a value the grammar somehow let slip through
        if qtype == "forces":
            return var, None, qtype
        rawval = obj.get("value")
        if rawval is None:
            return CANNOT_ANSWER  # whynot/can need a value
        return var, parse_value(var, str(rawval), lp), qtype
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        # parse_value raises ValueError on an out-of-range/ill-typed value; treat any
        # mapping/validation failure as "cannot answer" rather than crashing the REPL.
        return CANNOT_ANSWER


def format_structured(var: str, val, qtype: str) -> str:
    """Render a parsed query back as its structured-command form, e.g. 'whynot slot_Alice 2'."""
    if qtype == "forces":
        return f"forces {var}"
    return f"{qtype} {var} {val}"


def parse_command(raw: str, ctx, lp):
    """Parse an inline command: 'whynot <var> <val>' | 'can <var> <val>' | 'forces <var>'.
    Returns (var, val, query_type) or None if it isn't a recognised command. May raise ValueError
    for a recognised command with a bad variable/value."""
    toks = raw.split()
    if not toks or toks[0].lower() not in QUERY_TYPES:
        return None
    qtype = toks[0].lower()

    if qtype == "forces":
        if len(toks) != 2:
            raise ValueError("usage: forces <var>")
        var = toks[1]
        if var not in ctx.zvars:
            raise ValueError(f"unknown variable: {var}")
        return var, None, "forces"

    if len(toks) != 3:
        raise ValueError(f"usage: {qtype} <var> <val>")
    var, rawval = toks[1], toks[2]
    if var not in ctx.zvars:
        raise ValueError(f"unknown variable: {var}")
    return var, parse_value(var, rawval, lp), qtype


def guided_form(ctx, lp):
    """The Phase-1 'structured form' rendered in a terminal: prompt for variable, query type, value.
    Returns (var, val, query_type) or None to cancel. May raise ValueError on a bad value."""
    print("  (blank line cancels)")
    var = input("  Variable: ").strip()
    if not var:
        return None
    if var not in ctx.zvars:
        raise ValueError(f"unknown variable: {var}")
    qtype = input(f"  Query [{'/'.join(QUERY_TYPES)}]: ").strip().lower()
    if qtype not in QUERY_TYPES:
        raise ValueError(f"query must be one of {QUERY_TYPES}")
    val = None
    if qtype != "forces":
        val = parse_value(var, input("  Value: ").strip(), lp)
    return var, val, qtype


HELP = """Commands:
  whynot <var> <val>   why X cannot equal V (expects it to be impossible)   [structured mode]
  can    <var> <val>   whether X can equal V (yes -> a witness, no -> why not) [structured mode]
  forces <var>         what forces X to its value (or reports it is free)    [structured mode]
  <plain question>     ask in plain English, e.g. "why can't Alice be second?" [free-text mode]
  mode                 toggle between structured and free-text (LLM) input
  form                 fill the query out as a guided form
  vars                 list the puzzle's variables and their domains
  help                 show this help
  quit                 exit"""


def _mode_name(freetext_mode: bool) -> str:
    return f"free-text (LLM: {FREETEXT_MODEL})" if freetext_mode else "structured"


# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop + main
# ─────────────────────────────────────────────────────────────────────────────

def interactive(ctx, q_index, span_index, labels, lp, problem_text, freetext_mode: bool = False):
    print("=" * 78)
    print(f"Puzzle: {problem_text}")
    print("-" * 78)
    print("Variables:")
    print(list_variables(ctx, lp))
    print("-" * 78)
    print(HELP)
    print(f"\nMode: {_mode_name(freetext_mode)}  (toggle with 'mode')")

    while True:
        try:
            raw = input("\nquery> ").lstrip("﻿").strip()  # lstrip guards a piped-stdin BOM
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("quit", "exit", "q"):
            break
        if low in ("help", "?"):
            print(HELP)
            continue
        if low in ("vars", "variables"):
            print(list_variables(ctx, lp))
            continue
        if low in ("mode", "toggle", "llm", "structured"):
            freetext_mode = not freetext_mode
            print(f"  Mode: {_mode_name(freetext_mode)}")
            continue

        try:
            if low in ("form", "ask"):
                parsed = guided_form(ctx, lp)
            elif freetext_mode:
                # Free-text only: map the whole line via the LLM (no whynot/can/forces grammar).
                parsed = parse_freetext(raw, ctx, lp)
                if parsed is CANNOT_ANSWER:
                    print("  Cannot answer question (doesn't map to whynot/can/forces on this puzzle).")
                    continue
            else:
                # Structured mode: the deterministic command grammar, no LLM.
                parsed = parse_command(raw, ctx, lp)
            if parsed is None:
                print("  Could not parse. Try 'help', 'mode' for free-text, or 'form' for a guided prompt.")
                continue
            var, val, qtype = parsed
            if freetext_mode:
                print(f"  Interpreted as: {format_structured(var, val, qtype)}")
            prose, mcs = run_query(ctx, q_index, span_index, labels, var, val, qtype)
        except ValueError as e:
            print(f"  [invalid] {e}")
            continue
        except Exception as e:  # noqa: BLE001 — surface engine errors without killing the REPL
            print(f"  [error] {type(e).__name__}: {e}")
            continue

        print("\n" + prose)
        if mcs:
            print(mcs)


def main():
    ap = argparse.ArgumentParser(description="Interactive REPL over a stored logic puzzle.")
    ap.add_argument("run_id", nargs="?", default=None,
                    help="run_id from extraction_attempts (omit to pick a random row)")
    ap.add_argument("--db", default=DEFAULT_DB, help="path to sft_test.db")
    ap.add_argument("--source", choices=["gold", "extracted"], default="gold",
                    help="which extraction to query (default: gold)")
    ap.add_argument("--question", type=int, default=0, help="question index to query (default: 0)")
    ap.add_argument("--freetext", action="store_true",
                    help="start interactive session in free-text (LLM) mode; toggle in-session with 'mode'")
    # one-shot (non-interactive) mode — handy for tests / scripting (no stdin needed)
    ap.add_argument("--var", help="variable to query (one-shot mode)")
    ap.add_argument("--value", help="value for whynot/can (one-shot mode)")
    ap.add_argument("--query", choices=list(QUERY_TYPES), help="query type (one-shot mode)")
    args = ap.parse_args()

    run_id = args.run_id
    if run_id is None:
        run_id = pick_random_run_id(args.db)
        print(f"No run_id given - picked a random one: {run_id}\n")

    lp, problem_text = load_puzzle(args.db, run_id, args.source)
    ctx, span_index = explanation.prepare_query_context(lp, problem_text)
    labels = ctx.cid_label
    q_index = args.question

    # One-shot mode
    if args.var and args.query:
        if args.var not in ctx.zvars:
            sys.exit(f"unknown variable: {args.var}\nVariables:\n{list_variables(ctx, lp)}")
        val = None
        if args.query != "forces":
            if args.value is None:
                sys.exit("--value is required for --query whynot/can")
            try:
                val = parse_value(args.var, args.value, lp)
            except ValueError as e:
                sys.exit(f"invalid value: {e}")
        prose, mcs = run_query(ctx, q_index, span_index, labels, args.var, val, args.query)
        print(prose)
        if mcs:
            print(mcs)
        return

    interactive(ctx, q_index, span_index, labels, lp, problem_text, freetext_mode=args.freetext)


if __name__ == "__main__":
    main()
