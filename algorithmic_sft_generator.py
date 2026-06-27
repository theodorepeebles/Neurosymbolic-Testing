"""
algorithmic_sft_generator.py — Algorithmic Logic Puzzle Generator for SFT Data

This module generates synthetic training data for the neurosymbolic extraction
model using a Z3-first (fix-then-relax) approach. Unlike LLM-first generation
(gemini_auto_sft.py, ollama_auto_sft.py), which asks a model to invent a valid
puzzle and then Z3-verifies it, this pipeline inverts the order:

  Fix a random ground truth assignment -> derive all true statements from it
  -> prune to a target clue count while preserving >=2 valid solutions
  -> let Z3 determine each answer choice's correctness independently
  -> call the LLM only to paraphrase the formal structure into natural language.

Correctness is guaranteed by construction. The LLM cannot introduce logical
errors because it only touches surface prose.

The training target is problem_text -> extracted_json, consistent with the
existing pipeline. The extraction model never predicts answers directly -- Z3
computes them from extracted_json at inference time. The correct answer label is
therefore NOT stored on the row; the testing harness (run.py) re-derives ground
truth by Z3-solving the gold extraction.

Output appends only to sft_dataset.jsonl (the full dataset). It is decoupled from
the SQLite DB: split_dataset.py later carves this file into sft_train.jsonl /
sft_test.jsonl, and sft_test.db is populated only by run.py during evaluation.
"""

import json
import random
import argparse
import hashlib
import math
import os
import re
import uuid
from itertools import combinations
from datetime import datetime, timezone
from types import SimpleNamespace

from z3 import (
    Solver, Int, Bool, And, Or, Not, Distinct, Sum, If, Implies, Abs,
    sat, unsat
)

from validators import build_hybrid_schema, DOMAIN_LP_FIELDS
from pipeline import encode, ask_llm

try:
    from google import genai
    from google.genai import types as gtypes
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ==============================================================================
# PHASE 0 -- GLOBAL CONFIGURATION
# All bounds defined once as named constants. Every randomised draw in Phase 1
# pulls from these ranges.
# ==============================================================================
#
ENTITY_COUNT_MIN         = 2
ENTITY_COUNT_MAX         = 6
NUM_GROUPS_MIN           = 2      # only used when grouping is active
NUM_GROUPS_MAX           = 3
WRAPPER_PCT_MIN          = 0.0
WRAPPER_PCT_MAX          = 0.20   # at most 20% as many wrappers as primitives
PRUNE_TARGET_MIN         = 3
PRUNE_TARGET_MAX         = 6     # was 8 for first 100-1500 rows
ANSWER_CHOICES_MIN       = 2
ANSWER_CHOICES_MAX       = 4
QUESTION_TYPES           = ["must_be_true", "must_be_false", "could_be_true", "could_be_false"]
QUESTION_CONSTRAINTS_MIN = 0     # 0 -> bare "which must be true?"; 1 -> "given that X, ..."
QUESTION_CONSTRAINTS_MAX = 1
MAX_PARAPHRASE_RETRIES   = 3
MAX_QUESTION_RETRIES     = 10

SFT_OUT              = "../data/sft_dataset.jsonl"  # the full dataset; split later by split_dataset.py

PARAPHRASE_BACKEND   = "gemini"            # "gemini" | "ollama"
PARAPHRASE_MODEL     = "gemini-3.1-flash-lite"  # for ollama backend use: "gpt-oss:120b-cloud" or "gpt-oss:20b-cloud"
                                                # for gemini use "gemini-3.1-flash-lite"

KK_SPEECH_ACT_MAX    = 2     # max speech act PAIRS added post-pruning (= 4 constraints total)
CROSS_DOMAIN_BIAS    = 0.6   # extra cross-domain if_then budget as fraction of primitive_pool_size; 0 disables; was 8 for first 100-1500 rows
UNIQUE_SOLUTION_PROB = 0.10  # fraction of puzzles generated with exactly 1 solution
#
NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Liam", "Mona", "Nina", "Omar", "Priya",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zane", "Aria", "Theo", "Lena", "Cyrus", "Dara", "Finn",
    "Gwen", "Hugo", "Iris", "Juno", "Knox", "Luna", "Mars", "Nova",
    "Orla", "Penn", "Reed", "Sage", "Troy", "Vega", "Wade", "Zara",
]
# optionally expand pool via `names` package; exclude names that double as
# common English words (single-word names only)
try:
    import names as _names_pkg
    _extra = [_names_pkg.get_first_name() for _ in range(100)]
    NAMES = list(set(NAMES + [n for n in _extra if n.isalpha() and len(n) > 2]))
except ImportError:
    pass

ALL_DOMAINS = ["ordering", "knights_and_knaves", "grouping"]


# -- Utilities -----------------------------------------------------------------

def fp(text: str) -> str:
    """SHA-1 fingerprint of normalised problem text; used for duplicate detection."""
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def dict_to_ns(obj):
    """Recursively convert a nested dict/list into SimpleNamespace objects.
    encode() from pipeline.py accesses constraint fields as attributes; this
    bridges plain dicts (generated in Phase 3) and the attribute-style access
    encode() expects."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_ns(i) for i in obj]
    return obj


def load_seen_hashes(path: str) -> set:
    seen = set()
    try:
        for line in open(path):
            seen.add(fp(json.loads(line).get("problem_text", "")))
    except FileNotFoundError:
        pass
    return seen


# ==============================================================================
# PHASE 1 -- DRAW PUZZLE PARAMETERS  (pure Python random)
# ==============================================================================

def draw_puzzle_params() -> dict:
    """Sample all top-level puzzle parameters randomly.

    num_question_constraints and num_answer_choices are drawn here alongside
    all other parameters for organisational clarity, but are inert until
    Phase 5. Phases 2, 3, and 4 are completely agnostic to question shape.
    """
    entity_count = random.randint(ENTITY_COUNT_MIN, ENTITY_COUNT_MAX)

    # exclude grouping if entity_count < NUM_GROUPS_MIN -- can't form enough groups
    available_domains = [d for d in ALL_DOMAINS
                         if d != "grouping" or entity_count >= NUM_GROUPS_MIN]
    k = random.randint(1, max(1, len(available_domains)))
    active_domains = random.sample(available_domains, k)

    return {
        "entity_count":             entity_count,
        "active_domains":           active_domains,
        "wrapper_pct":              random.uniform(WRAPPER_PCT_MIN, WRAPPER_PCT_MAX),
        "prune_target":             random.randint(PRUNE_TARGET_MIN, PRUNE_TARGET_MAX),
        "num_answer_choices":       random.randint(ANSWER_CHOICES_MIN, ANSWER_CHOICES_MAX),
        "num_question_constraints": random.randint(QUESTION_CONSTRAINTS_MIN, QUESTION_CONSTRAINTS_MAX),
        "unique_solution":          random.random() < UNIQUE_SOLUTION_PROB,
    }


# ==============================================================================
# PHASE 2 -- DRAW DOMAIN AXIOMS + FIX GROUND TRUTH ASSIGNMENT  (pure Python)
# ==============================================================================

def _random_partition(total: int, num_parts: int) -> list:
    """Random integer partition of `total` into `num_parts` positive integers."""
    if num_parts == 1:
        return [total]
    available = list(range(1, total))
    if len(available) < num_parts - 1:
        # fall back to as-equal-as-possible split when total < num_parts is infeasible
        base, rem = divmod(total, num_parts)
        return [base + (1 if i < rem else 0) for i in range(num_parts)]
    cuts = sorted(random.sample(available, num_parts - 1))
    return [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [total - cuts[-1]]


def draw_domain_axioms_and_ground_truth(params: dict) -> dict:
    """Draw domain axiom info then fix the ground truth assignment for every
    entity in every active domain. This is the source of truth -- all downstream
    phases derive from it. No LLM, no Z3.

    Domain axioms (num_groups, num_slots) are drawn immediately
    before entity assignment so they are available for the grouping step.
    Only drawn for active domains.

    Note: all entities are assigned in every active domain. Partial assignment
    (entities absent from a domain) is not yet supported by the schema or Z3 path.
    """
    entity_count   = params["entity_count"]
    active_domains = params["active_domains"]

    entity_names = random.sample(NAMES, entity_count)

    # -- domain axiom info (only for active domains) --
    num_slots   = None
    num_groups  = None
    group_sizes = None

    if "ordering" in active_domains:
        # temporarily: num_slots always equals entity_count
        num_slots = entity_count

    if "grouping" in active_domains:
        # temporarily: group_sizes always partitions ALL entities exactly,
        # i.e. sum(group_sizes) == entity_count
        num_groups  = random.randint(NUM_GROUPS_MIN, min(NUM_GROUPS_MAX, entity_count))
        group_sizes = _random_partition(entity_count, num_groups)

    # -- ground truth assignment --
    truth = {}

    if "ordering" in active_domains:
        slots = list(range(1, entity_count + 1))
        random.shuffle(slots)
        truth["slots"] = dict(zip(entity_names, slots))  # entity -> 1-indexed slot

    if "knights_and_knaves" in active_domains:
        truth["kk"] = {e: random.choice(["knight", "knave"]) for e in entity_names}

    if "grouping" in active_domains:
        # distribute entities across groups respecting group_sizes (1-indexed)
        assignment = []
        for g_idx, size in enumerate(group_sizes):
            assignment.extend([g_idx + 1] * size)
        random.shuffle(assignment)
        truth["groups"] = dict(zip(entity_names, assignment))

    return {
        "entity_names": entity_names,
        "num_slots":    num_slots,
        "num_groups":   num_groups,
        "group_sizes":  group_sizes,
        "truth":        truth,
    }


# ==============================================================================
# PHASE 3 -- GENERATE CANDIDATE CLUE POOL  (pure Python, no Z3)
#
# Runtime: Phase 3a is O(N^2) in entity_count for ordering and grouping pair
# generation. For N<=6 this is at most ~30 primitives per domain -- negligible.
# Wrapper generation (Phase 3c) iterates pairs of primitives: O(P^2) where
# P=primitive_pool_size; at N=6 this is at most a few hundred pairs, still fast.
# ==============================================================================

def _gen_ordering_primitives(entity_names: list, truth: dict) -> tuple:
    """Generate true and false primitives for the ordering domain."""
    slots = truth["slots"]
    n     = len(entity_names)
    true_p, false_p = [], []

    # slot_fixed -- one per entity, always TRUE by construction
    for e in entity_names:
        true_p.append({"type": "slot_fixed", "entity": e, "slot": slots[e]})
        for s in range(1, n + 1):
            if s != slots[e]:
                false_p.append({"type": "slot_fixed", "entity": e, "slot": s})

    # before -- ALL ordered pairs, including transitive ones
    # "Alice before Carol" is valid even when Bob sits between them;
    # transitivity is a feature, not a reason to omit -- pruning discards redundant ones
    for x, y in combinations(entity_names, 2):
        if slots[x] < slots[y]:
            true_p.append({"type": "before", "left": x, "right": y})
            false_p.append({"type": "before", "left": y, "right": x})
        else:
            true_p.append({"type": "before", "left": y, "right": x})
            false_p.append({"type": "before", "left": x, "right": y})

    # immediately_before -- strict adjacency only
    for x in entity_names:
        for y in entity_names:
            if x == y:
                continue
            if slots[x] + 1 == slots[y]:
                true_p.append({"type": "immediately_before", "left": x, "right": y})
            else:
                false_p.append({"type": "immediately_before", "left": x, "right": y})

    # adjacent -- unordered pairs where gap == 1; not_adjacent formed by not(adjacent(...))
    for x, y in combinations(entity_names, 2):
        if abs(slots[x] - slots[y]) == 1:
            true_p.append({"type": "adjacent", "left": x, "right": y})
        else:
            false_p.append({"type": "adjacent", "left": x, "right": y})

    return true_p, false_p


def _gen_kk_primitives(entity_names: list, truth: dict) -> tuple:
    """Generate true and false primitives for the knights_and_knaves domain.

    Only direct type-assignment constraints are generated here. KK speech act
    constructs (e.g. "X says Y is a knight") are if_then wrappers assembled
    in Phase 3c from pairs of these primitives.
    """
    kk = truth["kk"]
    true_p, false_p = [], []
    for e in entity_names:
        if kk[e] == "knight":
            true_p.append({"type": "is_truth_teller", "entity": e})
            false_p.append({"type": "is_deceiver",     "entity": e})
        else:
            true_p.append({"type": "is_deceiver",     "entity": e})
            false_p.append({"type": "is_truth_teller", "entity": e})
    return true_p, false_p


def _gen_grouping_primitives(entity_names: list, truth: dict, num_groups: int) -> tuple:
    """Generate true and false primitives for the grouping domain."""
    groups = truth["groups"]
    true_p, false_p = [], []

    # same_group / different_group -- unordered pairs
    for x, y in combinations(entity_names, 2):
        if groups[x] == groups[y]:
            true_p.append({"type": "same_group",      "entities": [x, y]})
            false_p.append({"type": "different_group", "entities": [x, y]})
        else:
            true_p.append({"type": "different_group", "entities": [x, y]})
            false_p.append({"type": "same_group",     "entities": [x, y]})

    # is_in -- one per entity per group
    for e in entity_names:
        for g in range(1, num_groups + 1):
            if groups[e] == g:
                true_p.append({"type": "is_in", "entity": e, "group": g})
            else:
                false_p.append({"type": "is_in", "entity": e, "group": g})

    # exactly_n (full: all entities) -- canonical version, one per group.
    # Guard len > 1: a single-entity exactly_n collapses into is_in in natural
    # language ("exactly 1 of [X]" == "X is in ..."), so never emit one.
    all_ents = list(entity_names)
    if len(all_ents) > 1:
        for g in range(1, num_groups + 1):
            n_in_g = sum(1 for e in entity_names if groups[e] == g)
            true_p.append({"type": "exactly_n", "entities": all_ents, "n": n_in_g, "group": g})

    # exactly_n (subsets of size 3-4, non-trivial: 0 < count_in_G < |subset|)
    # larger subsets add marginal information and flood the pool
    for size in (3, 4):
        if len(entity_names) < size:
            continue
        for subset in combinations(entity_names, size):
            subset_list = list(subset)
            for g in range(1, num_groups + 1):
                count = sum(1 for e in subset_list if groups[e] == g)
                if 0 < count < len(subset_list):
                    true_p.append({"type": "exactly_n", "entities": subset_list,
                                   "n": count, "group": g})

    return true_p, false_p


def _gen_not_wrappers(false_primitives: list, budget: int) -> list:
    """Wrap FALSE primitives with NOT -> produces TRUE clues.
    E.g. NOT(slot_fixed(Alice, 3)) when Alice is actually in slot 2."""
    candidates = [{"type": "not", "claim": p} for p in false_primitives]
    random.shuffle(candidates)
    return candidates[:budget]


def _gen_ifthen_wrappers(budget: int, true_by_domain: dict) -> list:
    """Generate materially-true same-domain if_then wrappers.

    Cross-domain pairs are handled separately by _gen_cross_domain_ifthen(),
    which has its own CROSS_DOMAIN_BIAS budget. KK speech acts are handled by
    _gen_kk_speech_acts() and are not generated here.
    """
    candidates = []
    for domain_prims in true_by_domain.values():
        for p, q in combinations(domain_prims, 2):
            candidates.append({"type": "if_then", "antecedent": p, "consequent": q})
            candidates.append({"type": "if_then", "antecedent": q, "consequent": p})
    random.shuffle(candidates)
    return candidates[:budget]


def _gen_cross_domain_ifthen(true_by_domain: dict, budget: int) -> list:
    """Generate materially-true cross-domain if_then wrappers.

    Called with a separate CROSS_DOMAIN_BIAS budget so dependencies like "if Bob is
    in slot 2 then Alice is a deceiver" appear with higher frequency than the general
    wrapper budget would allow. Only called when multiple domains are active.
    """
    candidates = []
    keys = list(true_by_domain.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for p in true_by_domain[keys[i]]:
                for q in true_by_domain[keys[j]]:
                    candidates.append({"type": "if_then", "antecedent": p, "consequent": q})
                    candidates.append({"type": "if_then", "antecedent": q, "consequent": p})
    random.shuffle(candidates)
    return candidates[:budget]


def _gen_kk_speech_acts(entity_names: list, truth: dict,
                         true_by_domain: dict, false_by_domain: dict) -> list:
    """Generate KK speech act pairs for all KK entities across all active domains.

    A speech act "X says [claim]" is encoded as a pair of if_then constraints:
      if_then(is_truth_teller(X), claim) + if_then(is_deceiver(X), not(claim))
    Both are TRUE in ground truth: knight X makes true claims, knave X makes false claims
    (so not(false_claim) holds). Claims are drawn from ALL active domains, so speech acts
    cover ordering and grouping facts in addition to KK status -- e.g. "Alice says Bob
    is in slot 3" or "Carol says Dave is in group 2".
    Self-referential KK claims (X says X is a knight/knave) are excluded.
    """
    kk = truth["kk"]
    pairs = []
    for x in entity_names:
        x_is_knight = kk[x] == "knight"
        claims_pool = (
            [c for prims in true_by_domain.values()  for c in prims]
            if x_is_knight else
            [c for prims in false_by_domain.values() for c in prims]
        )
        for claim in claims_pool:
            # skip "X says X is a knight/knave" -- self-referential and trivial
            if claim.get("entity") == x and claim.get("type") in ("is_truth_teller", "is_deceiver"):
                continue
            pairs.append([
                {"type": "if_then",
                 "antecedent": {"type": "is_truth_teller", "entity": x},
                 "consequent": claim},
                {"type": "if_then",
                 "antecedent": {"type": "is_deceiver",     "entity": x},
                 "consequent": {"type": "not", "claim": claim}},
            ])
    return pairs


def generate_candidate_pool(params: dict, gt: dict) -> tuple:
    """Build the full candidate clue pool.

    Returns (candidate_pool, false_primitives, speech_act_pairs).
    - candidate_pool: primitives + same-domain wrappers + cross-domain wrappers
    - false_primitives: used as question constraint candidates in Phase 5
    - speech_act_pairs: KK speech act pairs kept SEPARATE from candidate_pool;
      added post-pruning in generate_one_puzzle() to avoid overwhelming pruning
    """
    entity_names   = gt["entity_names"]
    active_domains = params["active_domains"]
    truth          = gt["truth"]
    num_groups     = gt["num_groups"]
    wrapper_pct    = params["wrapper_pct"]

    all_true, all_false = [], []
    true_by_domain  = {}
    false_by_domain = {}  # needed for KK speech acts (knaves make false claims)

    if "ordering" in active_domains:
        t, f = _gen_ordering_primitives(entity_names, truth)
        all_true.extend(t); all_false.extend(f)
        true_by_domain["ordering"]  = t
        false_by_domain["ordering"] = f

    if "knights_and_knaves" in active_domains:
        t, f = _gen_kk_primitives(entity_names, truth)
        all_true.extend(t); all_false.extend(f)
        true_by_domain["knights_and_knaves"]  = t
        false_by_domain["knights_and_knaves"] = f

    if "grouping" in active_domains:
        t, f = _gen_grouping_primitives(entity_names, truth, num_groups)
        all_true.extend(t); all_false.extend(f)
        true_by_domain["grouping"]  = t
        false_by_domain["grouping"] = f

    primitive_pool_size = len(all_true)
    wrapper_budget      = math.floor(wrapper_pct * primitive_pool_size)

    # same-domain if_then wrappers + not wrappers
    wrappers = []
    if wrapper_budget > 0:
        half     = wrapper_budget // 2
        not_w    = _gen_not_wrappers(all_false, half)
        ifthen_w = _gen_ifthen_wrappers(wrapper_budget - half, true_by_domain)
        wrappers = not_w + ifthen_w

    # cross-domain if_then wrappers (separate budget controlled by CROSS_DOMAIN_BIAS)
    cross_wrappers = []
    if CROSS_DOMAIN_BIAS > 0 and len(active_domains) > 1:
        cross_budget   = math.floor(CROSS_DOMAIN_BIAS * primitive_pool_size)
        cross_wrappers = _gen_cross_domain_ifthen(true_by_domain, cross_budget)

    # KK speech acts are generated here but returned SEPARATELY -- they go into the pool
    # post-pruning (in generate_one_puzzle) with a strict count cap, so they don't
    # overwhelm the greedy pruning step with hundreds of redundant if_then pairs
    speech_act_pairs = []
    if "knights_and_knaves" in active_domains:
        speech_act_pairs = _gen_kk_speech_acts(entity_names, truth, true_by_domain, false_by_domain)

    return all_true + wrappers + cross_wrappers, all_false, speech_act_pairs


# ==============================================================================
# PHASE 4 -- PRUNE TO TARGET WITH MULTI-SOLUTION GUARANTEE  (Z3)
# ==============================================================================

def _build_z3_vars_and_base(entity_names: list, active_domains: list,
                             num_slots, num_groups) -> tuple:
    """Build Z3 variable dict and domain-axiom base expressions.

    Refactored from z3_solve() in pipeline.py to work on raw parameters rather
    than a LogicProblem instance, so pruning (Phase 4) and question checking
    (Phase 5) can call it before the Pydantic object is assembled.

    Returns (vars_dict, base_exprs) where base_exprs encodes structural domain
    axioms (bounds, distinctness). Group sizes flow through exactly_n constraints
    in the clue list, not through base_exprs.
    """
    vars_dict  = {}
    base_exprs = []

    if "ordering" in active_domains:
        slot_vars = {f"slot_{e}": Int(f"slot_{e}") for e in entity_names}
        vars_dict.update(slot_vars)
        base_exprs.append(Distinct(list(slot_vars.values())))
        for e in entity_names:
            v = slot_vars[f"slot_{e}"]
            base_exprs.append(v >= 1)
            base_exprs.append(v <= num_slots)

    if "grouping" in active_domains:
        group_vars = {f"group_{e}": Int(f"group_{e}") for e in entity_names}
        vars_dict.update(group_vars)
        for e in entity_names:
            v = group_vars[f"group_{e}"]
            base_exprs.append(v >= 1)
            base_exprs.append(v <= num_groups)
        # group sizes flow through exactly_n constraints, not base_exprs

    if "knights_and_knaves" in active_domains:
        kk_vars = {f"kk_{e}": Bool(f"kk_{e}") for e in entity_names}
        vars_dict.update(kk_vars)

    return vars_dict, base_exprs


def count_solutions(exprs: list, vars_dict: dict, min_count: int = 2) -> int:
    """Count Z3 solutions up to min_count using blocking clauses.

    Each iteration finds a solution, then adds a clause that blocks it by
    requiring at least one variable to differ. Returns as soon as min_count
    solutions are found (short-circuit). Used in Phase 4 to verify >=2 solutions
    remain after each greedy clue removal.
    """
    s = Solver()
    s.add(exprs)
    count = 0
    while count < min_count:
        if s.check() != sat:
            break
        m = s.model()
        count += 1
        block = Or([v != m.eval(v, model_completion=True) for v in vars_dict.values()])
        s.add(block)
    return count


def prune_to_target(candidate_pool: list, params: dict, gt: dict,
                    unique_solution: bool = False) -> tuple:
    """Greedily remove clues while maintaining the required solution count.

    For unique_solution=False (default): guarantees >=2 solutions remain.
      Phase A breaks the initial 1-solution deadlock by freely removing
      constraints until ambiguity is unlocked; Phase B then prunes normally.
      Because solutions are monotone (removing constraints never decreases
      count), Phase B always reaches prune_target once count >= 2.
    For unique_solution=True: maintains exactly 1 solution throughout.
      Two passes handle cascading unlocks (constraints that become removable
      only after others are shed first).

    Returns (surviving_clues, pruned_away_clues).
    pruned_away_clues are candidates for question constraints in Phase 5 --
    they were redundant given the full pool but the surviving body no longer
    contains them, so adding one back as a question constraint narrows the
    solution space in a controlled way.
    """
    entity_names   = gt["entity_names"]
    active_domains = params["active_domains"]
    prune_target   = params["prune_target"]

    vars_dict, base_exprs = _build_z3_vars_and_base(
        entity_names, active_domains,
        gt["num_slots"], gt["num_groups"]
    )

    # encode all candidate clues; skip any that fail encoding
    encoded    = []
    valid_pool = []
    for c_dict in candidate_pool:
        try:
            expr = encode(dict_to_ns(c_dict), vars_dict)
            encoded.append(expr)
            valid_pool.append(c_dict)
        except Exception:
            pass

    if not valid_pool:
        return [], []

    # Phase 4b: full pool must be SAT (always true by construction)
    if count_solutions(base_exprs + encoded, vars_dict, min_count=1) < 1:
        print("  [!] Full clue pool is UNSAT -- something went wrong in Phase 3")
        return [], []

    indices       = list(range(len(valid_pool)))
    random.shuffle(indices)
    surviving_set = set(indices)
    pruned_list   = []

    if unique_solution:
        # Phase 4c (1-solution mode): only remove if count stays == 1.
        # Two passes handle cascading unlocks.
        for _ in range(2):
            pass_indices  = list(surviving_set)
            random.shuffle(pass_indices)
            made_progress = False
            for idx in pass_indices:
                if len(surviving_set) <= prune_target:
                    break
                test_set   = surviving_set - {idx}
                test_exprs = base_exprs + [encoded[i] for i in test_set]
                if count_solutions(test_exprs, vars_dict, min_count=2) == 1:
                    surviving_set.discard(idx)
                    pruned_list.append(idx)
                    made_progress = True
            if not made_progress:
                break

        # Phase 4d: verify uniqueness preserved
        surviving_exprs = base_exprs + [encoded[i] for i in surviving_set]
        final_count     = count_solutions(surviving_exprs, vars_dict, min_count=2)
        if final_count == 0:
            print("  [!] unique_solution mode: surviving set is UNSAT.")
        elif final_count >= 2:
            print("  [!] unique_solution mode: uniqueness lost -- puzzle has >=2 solutions.")

    else:
        # Phase A: the full primitive pool always starts at 1 solution (by construction --
        # all primitives are true statements about the ground truth, over-determining it).
        # Freely remove constraints one-at-a-time until ambiguity (>=2 solutions) is
        # unlocked, then hand off to Phase B.
        initial_count = count_solutions(
            base_exprs + [encoded[i] for i in surviving_set], vars_dict, min_count=2
        )
        if initial_count < 2:
            for idx in indices:
                if len(surviving_set) <= prune_target:
                    break
                surviving_set.discard(idx)
                pruned_list.append(idx)
                test_exprs = base_exprs + [encoded[i] for i in surviving_set]
                if count_solutions(test_exprs, vars_dict, min_count=2) >= 2:
                    break  # ambiguity unlocked; hand off to Phase B

        # Phase B: only remove if >=2 solutions maintained. Solutions are monotone
        # (removing constraints never decreases count), so this always reaches
        # prune_target -- no second pass needed.
        indices_b = list(surviving_set)
        random.shuffle(indices_b)
        for idx in indices_b:
            if len(surviving_set) <= prune_target:
                break
            test_set   = surviving_set - {idx}
            test_exprs = base_exprs + [encoded[i] for i in test_set]
            if count_solutions(test_exprs, vars_dict, min_count=2) >= 2:
                surviving_set.discard(idx)
                pruned_list.append(idx)

        # Phase 4d: post-loop safety check -- recover if count accidentally dropped
        surviving_exprs = base_exprs + [encoded[i] for i in surviving_set]
        if count_solutions(surviving_exprs, vars_dict, min_count=2) < 2:
            if pruned_list:
                last = pruned_list.pop()
                surviving_set.add(last)
                surviving_exprs = base_exprs + [encoded[i] for i in surviving_set]
                if count_solutions(surviving_exprs, vars_dict, min_count=2) < 2:
                    print("  [!] Could not achieve >=2 solutions after recovery.")

    if len(surviving_set) > prune_target:
        print(f"  [!] prune_target={prune_target} not met; "
              f"surviving={len(surviving_set)} (puzzle's minimal set is larger)")

    # Phase 4e: pruned_away clues are question constraint candidates in Phase 5
    surviving_clues = [valid_pool[i] for i in sorted(surviving_set)]
    pruned_away     = [valid_pool[i] for i in pruned_list]
    return surviving_clues, pruned_away


# ==============================================================================
# PHASE 5 -- GENERATE QUESTION, QUESTION CONSTRAINT, AND ANSWER CHOICES  (Z3)
#
# Each answer choice independently draws its own type from QUESTION_TYPES. A
# single question can therefore contain a must_be_true choice alongside a
# could_be_false choice. Exactly one choice must pass its Z3 check; all others
# must fail theirs. This differs from a single question_type design -- see the
# table in _z3_check_choice for per-choice check semantics.
#
# temporarily: exactly one question per puzzle.
# ==============================================================================

def _candidate_values(domain: str, gt: dict) -> list:
    """All possible domain values for an entity (the answer choice candidates)."""
    if domain == "ordering":
        return list(range(1, gt["num_slots"] + 1))
    elif domain == "knights_and_knaves":
        return ["knight", "knave"]
    elif domain == "grouping":
        return list(range(1, gt["num_groups"] + 1))
    return []


def _ground_truth_value(domain: str, entity: str, truth: dict):
    """Return the ground truth value for (entity, domain)."""
    if domain == "ordering":           return truth["slots"][entity]
    if domain == "knights_and_knaves": return truth["kk"][entity]
    if domain == "grouping":           return truth["groups"][entity]


def _value_to_constraint(entity: str, domain: str, value) -> dict:
    """Convert a (entity, domain, value) triple to a constraint dict."""
    if domain == "ordering":
        return {"type": "slot_fixed", "entity": entity, "slot": int(value)}
    elif domain == "knights_and_knaves":
        typ = "is_truth_teller" if value == "knight" else "is_deceiver"
        return {"type": typ, "entity": entity}
    elif domain == "grouping":
        return {"type": "is_in", "entity": entity, "group": int(value)}
    raise ValueError(f"Unknown domain: {domain}")


def _z3_check_choice(value_dict: dict, choice_type: str,
                     puzzle_exprs: list, vars_dict: dict) -> bool:
    """Return True if (value, type) pair is a CORRECT answer; False if a distractor.

    +----------------+---------------------------+---------------------------+
    | choice_type    | CORRECT when              | DISTRACTOR when           |
    +----------------+---------------------------+---------------------------+
    | must_be_true   | puzzle + NOT(V) = UNSAT   | puzzle + NOT(V) = SAT     |
    | must_be_false  | puzzle + V = UNSAT        | puzzle + V = SAT          |
    | could_be_true  | puzzle + V = SAT          | puzzle + V = UNSAT        |
    | could_be_false | puzzle + NOT(V) = SAT     | puzzle + NOT(V) = UNSAT   |
    +----------------+---------------------------+---------------------------+
    """
    try:
        value_expr = encode(dict_to_ns(value_dict), vars_dict)
    except Exception:
        return False

    s = Solver()
    s.add(puzzle_exprs)

    if choice_type == "must_be_true":
        s.add(Not(value_expr));  return s.check() == unsat
    elif choice_type == "must_be_false":
        s.add(value_expr);       return s.check() == unsat
    elif choice_type == "could_be_true":
        s.add(value_expr);       return s.check() == sat
    elif choice_type == "could_be_false":
        s.add(Not(value_expr));  return s.check() == sat
    return False


def generate_question(surviving_clues: list, pruned_clues: list,
                      params: dict, gt: dict,
                      unique_solution: bool = False):
    """Generate a question with question constraint and labeled answer choices.

    Each answer choice has its own independently drawn type from QUESTION_TYPES
    (not a single shared question_type). Exactly one choice must be correct per
    its type's Z3 check; all others must fail.

    For unique_solution=True: answer types are restricted to must_be_true /
    must_be_false (could_be_* are semantically identical to must_be_* when only
    one solution exists), and question_constraints are forced empty.

    Retries up to MAX_QUESTION_RETRIES by resampling the answer target.
    Returns a result dict or None -- most likely failure point for small entity
    counts where the domain has few candidate values to form distractors from.
    """
    entity_names       = gt["entity_names"]
    active_domains     = params["active_domains"]
    num_answer_choices = params["num_answer_choices"]
    truth              = gt["truth"]

    if unique_solution:
        available_types          = ["must_be_true", "must_be_false"]
        num_question_constraints = 0
    else:
        available_types          = QUESTION_TYPES
        num_question_constraints = params["num_question_constraints"]

    vars_dict, base_exprs = _build_z3_vars_and_base(
        entity_names, active_domains,
        gt["num_slots"], gt["num_groups"]
    )

    # encode surviving puzzle body clues into Z3
    body_exprs = list(base_exprs)
    for c_dict in surviving_clues:
        try:
            body_exprs.append(encode(dict_to_ns(c_dict), vars_dict))
        except Exception:
            pass

    shuffled_pruned = list(pruned_clues)
    random.shuffle(shuffled_pruned)

    for _ in range(MAX_QUESTION_RETRIES):
        # 5a: sample answer target -- one entity and one domain attribute
        domain = random.choice(active_domains)
        entity = random.choice(entity_names)
        values = _candidate_values(domain, gt)
        v_gt   = _ground_truth_value(domain, entity, truth)

        # cap num_answer_choices to the number of available values
        actual_num_choices = min(num_answer_choices, len(values))
        if actual_num_choices < 2:
            continue  # not enough values to form a meaningful question

        # 5b: optional question constraint from pruned_away pool
        # valid if it doesn't directly encode the answer value
        qc_exprs = []
        qc_dicts = []
        if num_question_constraints > 0 and shuffled_pruned:
            v_gt_dict = _value_to_constraint(entity, domain, v_gt)
            for qc_candidate in shuffled_pruned[:15]:
                if qc_candidate == v_gt_dict:
                    continue
                try:
                    qc_exprs = [encode(dict_to_ns(qc_candidate), vars_dict)]
                    qc_dicts = [qc_candidate]
                    break
                except Exception:
                    continue

        # puzzle_exprs = domain axioms + surviving body clues + optional QC
        puzzle_exprs = body_exprs + qc_exprs

        # 5c: designate v_gt as the correct answer; find a type T_c where its check passes
        correct_value_dict = _value_to_constraint(entity, domain, v_gt)
        types_shuffled     = random.sample(available_types, len(available_types))
        correct_type       = None
        for t in types_shuffled:
            if _z3_check_choice(correct_value_dict, t, puzzle_exprs, vars_dict):
                correct_type = t
                break
        if correct_type is None:
            continue  # no type works for v_gt -- retry with different target

        # build distractors: for each other value, find a type where the check FAILS
        distractor_values = [v for v in values if v != v_gt]
        random.shuffle(distractor_values)

        distractors = []
        for v_d in distractor_values:
            if len(distractors) >= actual_num_choices - 1:
                break
            v_d_dict   = _value_to_constraint(entity, domain, v_d)
            d_types    = random.sample(available_types, len(available_types))
            found_type = None
            for t_d in d_types:
                if not _z3_check_choice(v_d_dict, t_d, puzzle_exprs, vars_dict):
                    found_type = t_d
                    break
            if found_type is not None:
                distractors.append((v_d_dict, found_type))

        if len(distractors) < actual_num_choices - 1:
            continue  # not enough valid distractors -- retry

        # 5e: assemble choices, shuffle correct position, assign labels
        # correct_label is not persisted -- the extraction model is trained on
        # problem_text -> extracted_json, and Z3 re-derives the answer at eval time
        choices = (
            [(correct_value_dict, correct_type, True)]
            + [(v, t, False) for v, t in distractors[:actual_num_choices - 1]]
        )
        random.shuffle(choices)
        labels = ["A", "B", "C", "D"][:len(choices)]

        answer_choices = []
        correct_label  = None
        for label, (v_dict, c_type, is_correct) in zip(labels, choices):
            answer_choices.append({
                "label":       label,
                "type":        c_type,
                "constraints": [v_dict],
            })
            if is_correct:
                correct_label = label

        return {
            "question_constraints": qc_dicts,
            "answer_choices":       answer_choices,
            "correct_label":        correct_label,
            "answer_entity":        entity,
            "answer_domain":        domain,
        }

    return None  # retries exhausted


# ==============================================================================
# PHASE 6 -- ASSEMBLE EXTRACTED JSON  (Pydantic validation)
# ==============================================================================

def assemble_extracted_json(surviving_clues: list, question_info: dict,
                            params: dict, gt: dict):
    """Build and validate a LogicProblem Pydantic instance.

    domain axiom info (num_slots, num_groups) is included as top-level fields
    per DOMAIN_LP_FIELDS. Group sizes are encoded as exactly_n constraints.
    The natural-language question stem is NOT stored here -- it lives only in
    problem_text produced by Phase 7.

    Raises on schema validation failure; caller should catch and discard.
    """
    active_domains = params["active_domains"]

    extracted_dict = {
        "entities":    gt["entity_names"],
        "constraints": surviving_clues,
        "questions": [{
            "question_constraints": question_info["question_constraints"],
            "answer_choices":       question_info["answer_choices"],
        }],
    }

    # domain axiom info -- include only the fields required by active domains
    if "ordering" in active_domains:
        # temporarily: num_slots always equals entity_count
        extracted_dict["num_slots"] = gt["num_slots"]
    if "grouping" in active_domains:
        extracted_dict["num_groups"] = gt["num_groups"]

    LogicProblem = build_hybrid_schema(active_domains)
    return LogicProblem(**extracted_dict)


# ==============================================================================
# PHASE 7 -- LLM PARAPHRASE  (one call; only LLM in pipeline)
# ==============================================================================

# Evidence-first output contract (shared by both system prompts below). The model first
# commits to one self-contained sentence per constraint -- clues, the "Given that" question
# premise, and each answer choice -- then assembles the full puzzle ("problem_text") that
# contains each of those strings VERBATIM. The verbatim string is stored as evidence_text
# on the constraint (inject_evidence) -- the gold target the extraction model learns to
# emit. attribution.py converts that text to a [start, end] span later (substring match);
# the LLM never computes offsets. See _OUTPUT_CONTRACT for the rules.
_OUTPUT_CONTRACT = (
    "Output a single JSON object with exactly these keys:\n"
    '  "clue_evidence": an array of strings, one per clue, in the SAME order the clues are '
    "listed. Each is one self-contained sentence stating exactly that clue.\n"
    '  "question_evidence": a string giving the clause that states the "Given that" question '
    "premise, or null when no premise is listed.\n"
    '  "choice_evidence": an object mapping each answer-choice label ("A", "B", ...) to the '
    "clause in that choice that states its proposition (exclude the label and the [modality] "
    "tag).\n"
    '  "problem_text": the complete natural-language puzzle as one string.\n'
    "HARD REQUIREMENT: every string in clue_evidence, question_evidence, and choice_evidence "
    "must appear in problem_text as an EXACT, contiguous, verbatim substring. Do NOT split it "
    "across other text, reword it, change its punctuation or capitalization, or merge two of "
    "them into one sentence. Write each clue as its own sentence."
)

# Domain-language discipline: keep grouping and ordering vocabulary distinct, and keep
# counting clues (exactly_n) distinct from plain membership clues (is_in) so they never
# collapse into the same wording. Appended to both paraphrase system prompts.
_DOMAIN_LANGUAGE_RULES = (
    "Domain language rules:\n"
    "- GROUPING clues describe MEMBERSHIP. Use group/team/category language only "
    "('is in group N', 'are in the same group', 'are in different groups'). "
    "Never describe a group with order or position words.\n"
    "- ORDERING clues describe ORDER. Use position/slot/order language only "
    "('is in position N', 'is before', 'is immediately before', 'are adjacent'). "
    "Never describe a position as a 'group'.\n"
    "- COUNTING vs MEMBERSHIP: reserve the word 'exactly' for counting clues of the form "
    "'exactly N of [...] are in group M'. A plain membership clue ('X is in group M') must "
    "NEVER use 'exactly' or any counting / number-of phrasing. Conversely, a counting clue "
    "must always state its count explicitly with 'exactly N'."
)

PARAPHRASE_SYSTEM = (
    "You render structured logic puzzles as natural language. "
    "Write exactly the puzzle described -- no extra clues, no omissions. "
    "Vary surface phrasing naturally. Do not add or remove any information. "
    "If-then constraints are CONDITIONAL statements -- always phrase them as "
    "'If [antecedent], then [consequent]' or 'Whenever [A], [B]'. "
    "Never use causal or consequentialist phrasing such as 'Because A is true, B follows' "
    "or 'Since A, B must be the case' -- that incorrectly implies the antecedent is an "
    "established fact rather than a hypothetical condition.\n\n"
    + _DOMAIN_LANGUAGE_RULES + "\n\n"
    + _OUTPUT_CONTRACT
)

PARAPHRASE_SYSTEM_UNIQUE = (
    "You render structured logic puzzles as natural language. "
    "The puzzle has exactly one valid solution — write it as a deterministic deduction, "
    "not a process of elimination. "
    "Write exactly the puzzle described -- no extra clues, no omissions. "
    "Do not add or remove any information. "
    "If-then constraints are CONDITIONAL statements -- always phrase them as "
    "'If [antecedent], then [consequent]'. Never use causal phrasing that implies "
    "the antecedent is an established fact.\n\n"
    + _DOMAIN_LANGUAGE_RULES + "\n\n"
    + _OUTPUT_CONTRACT
)

# concrete example included in every paraphrase prompt so the LLM has an
# unambiguous rendering target: the question always asks "which is correct?" and
# each answer choice carries its verbatim modality tag in square brackets
# Every clue_evidence / question_evidence / choice_evidence string appears verbatim inside
# problem_text — that is the contract. Example 1 shows a "Given that" premise (non-null
# question_evidence); the unique example shows the null case.
_PARAPHRASE_EXAMPLE = (
    "Example of the expected output format:\n\n"
    "Input:\n"
    "  Entities: Alice, Bob, Carol\n"
    "  Clues:\n"
    "    - Alice is before Bob\n"
    "    - Bob is immediately before Carol\n"
    "  Given that: Alice is in slot 1\n"
    "  Question: Which of the following is correct?\n"
    "  Answer choices:\n"
    "    A) [must be true] Bob is in slot 2\n"
    "    B) [could be false] Carol is in slot 1\n\n"
    "Output:\n"
    + json.dumps({
        "clue_evidence": [
            "Alice stands somewhere before Bob.",
            "Bob is immediately in front of Carol.",
        ],
        "question_evidence": "Alice is in the first position",
        "choice_evidence": {
            "A": "Bob is in the second position",
            "B": "Carol is in the first position",
        },
        "problem_text": (
            "Alice, Bob, and Carol are standing in a line of three positions. "
            "Alice stands somewhere before Bob. Bob is immediately in front of Carol. "
            "Given that Alice is in the first position, which of the following is correct?\n"
            "A) [must be true] Bob is in the second position\n"
            "B) [could be false] Carol is in the first position"
        ),
    }, indent=2)
)

_PARAPHRASE_EXAMPLE_UNIQUE = (
    "Example of the expected output format:\n\n"
    "Input:\n"
    "  Entities: Alice, Bob, Carol\n"
    "  Clues:\n"
    "    - Alice is in slot 1\n"
    "    - Bob is immediately before Carol\n"
    "    - Carol is in slot 3\n"
    "  Question: Which of the following is correct?\n"
    "  Answer choices:\n"
    "    A) [must be true] Bob is in slot 2\n"
    "    B) [must be false] Alice is in slot 3\n\n"
    "Output:\n"
    + json.dumps({
        "clue_evidence": [
            "Alice occupies the first position.",
            "Bob stands immediately before Carol.",
            "Carol is in the third position.",
        ],
        "question_evidence": None,
        "choice_evidence": {
            "A": "Bob is in the second position",
            "B": "Alice is in the third position",
        },
        "problem_text": (
            "Alice, Bob, and Carol are placed in three positions. "
            "Alice occupies the first position. Bob stands immediately before Carol. "
            "Carol is in the third position. Which of the following is correct?\n"
            "A) [must be true] Bob is in the second position\n"
            "B) [must be false] Alice is in the third position"
        ),
    }, indent=2)
)


def _constraint_to_english(c: dict) -> str:
    """Convert a constraint dict to a brief English description for the prompt."""
    t = c.get("type", "")
    if t == "slot_fixed":         return f"{c['entity']} is in slot {c['slot']}"
    elif t == "before":           return f"{c['left']} is before {c['right']}"
    elif t == "immediately_before": return f"{c['left']} is immediately before {c['right']}"
    elif t == "adjacent":         return f"{c['left']} and {c['right']} are adjacent"
    elif t == "is_in":            return f"{c['entity']} is in group {c['group']}"
    elif t == "is_truth_teller":  return f"{c['entity']} is a truth-teller (knight)"
    elif t == "is_deceiver":      return f"{c['entity']} is a deceiver (knave)"
    elif t == "same_group":       return f"{' and '.join(c['entities'])} are in the same group"
    elif t == "different_group":  return f"{' and '.join(c['entities'])} are in different groups"
    elif t == "exactly_n":
        ents = ", ".join(c["entities"])
        return f"exactly {c['n']} of [{ents}] are in group {c.get('group', '?')}"
    elif t == "not":
        return f"it is not the case that: {_constraint_to_english(c['claim'])}"
    elif t == "if_then":
        return (f"if {_constraint_to_english(c['antecedent'])}, "
                f"then {_constraint_to_english(c['consequent'])}")
    elif t in ("and", "or"):
        sep   = " and " if t == "and" else " or "
        parts = sep.join(_constraint_to_english(cl) for cl in c["claims"])
        return f"({parts})"
    return str(c)


def _build_paraphrase_prompt(extracted, question_info: dict,
                              active_domains: list = None,
                              unique_solution: bool = False) -> str:
    """Serialise extracted_json into a structured intermediate format for the LLM.
    Lists entities, clues, question type, and answer choices clearly so the LLM
    has an unambiguous rendering target. Avoids passing raw JSON."""
    dump         = extracted.model_dump()
    entities_str = ", ".join(dump["entities"])
    clue_strs    = "\n".join(f"  - {_constraint_to_english(c)}"
                             for c in dump["constraints"])

    question = dump["questions"][0]
    qc_str   = ""
    if question["question_constraints"]:
        qc_str = f"  Given that: {_constraint_to_english(question['question_constraints'][0])}\n"

    # question always asks "which is correct?" -- per-choice modality tags carry the semantics
    stem = "Which of the following is correct?"

    # each choice has its verbatim modality tag in square brackets so the LLM preserves it
    choices_str = "\n".join(
        f"  {ch['label']}) [{ch['type'].replace('_', ' ')}] {_constraint_to_english(ch['constraints'][0])}"
        for ch in question["answer_choices"]
        if ch.get("constraints")
    )

    # domain axioms: explicit structural facts that must be stated at the start of the problem
    axiom_lines = []
    if active_domains:
        if "ordering" in active_domains:
            n = dump.get("num_slots", "?")
            axiom_lines.append(f"  - There are {n} positions (numbered 1 through {n})")
        if "knights_and_knaves" in active_domains:
            axiom_lines.append(
                "  - Every person is either a knight (always tells the truth) "
                "or a knave (always lies)"
            )
        if "grouping" in active_domains:
            n = dump.get("num_groups", "?")
            axiom_lines.append(f"  - There are {n} groups (group sizes are NOT given here; "
                               "they may or may not appear in the clues)")
    axiom_str = "\n".join(axiom_lines) if axiom_lines else ""

    axiom_block = (
        f"Domain axioms (MUST be stated explicitly at the start of the problem):\n{axiom_str}\n"
        if axiom_str else ""
    )

    example = _PARAPHRASE_EXAMPLE_UNIQUE if unique_solution else _PARAPHRASE_EXAMPLE
    return (
        f"{example}\n\n"
        f"Now render this puzzle as natural language:\n\n"
        f"{axiom_block}"
        f"Entities: {entities_str}\n"
        f"Clues:\n{clue_strs}\n"
        f"{qc_str}"
        f"Question: {stem}\n"
        f"Answer choices:\n{choices_str}\n\n"
        f"Output a single JSON object with keys \"clue_evidence\", \"question_evidence\", "
        f"\"choice_evidence\", and \"problem_text\" as shown in the example. clue_evidence has "
        f"exactly one sentence per clue listed above, in order. choice_evidence has one entry "
        f"per answer-choice label. Every clue_evidence, question_evidence, and choice_evidence "
        f"string must appear verbatim inside problem_text. "
        f"Open problem_text with a sentence or two that states the domain axioms above. "
        f"Phrase any if-then clues as conditionals ('If A, then B'), never as causal statements. "
        f"If a 'Given that:' line is present above, it is a question-specific premise, NOT a "
        f"general clue: introduce it in the question portion of problem_text, starting with "
        f"'Given that ... {{question constraint}}', do NOT list it among the clues, do NOT add "
        f"it to clue_evidence, and put its clause in question_evidence. If no 'Given that:' line "
        f"is present, set question_evidence to null."
    )


def paraphrase_gemini(prompt: str, model: str,
                      system: str = PARAPHRASE_SYSTEM) -> str:
    """Call Gemini for free-text paraphrase.
    Mirrors call() in gemini_auto_sft.py but without response_mime_type='application/json'
    -- paraphrase output is natural language, not structured JSON."""
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-genai not installed; set PARAPHRASE_BACKEND='ollama'")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=gtypes.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=2048,
        ),
    ).text


def paraphrase_ollama(prompt: str, model: str,
                      system: str = PARAPHRASE_SYSTEM) -> str:
    """Call Ollama for free-text paraphrase.
    Uses ask_llm() from pipeline.py without the fmt= argument -- no grammar
    constraints needed since the output is natural language, not structured JSON.
    Default model: gpt-oss:120b-cloud (the cloud model used in ollama_auto_sft.py)."""
    return ask_llm(prompt=prompt, system=system, model=model)


def paraphrase(prompt: str, system: str = PARAPHRASE_SYSTEM) -> str:
    if PARAPHRASE_BACKEND == "gemini":
        return paraphrase_gemini(prompt, PARAPHRASE_MODEL, system=system)
    return paraphrase_ollama(prompt, PARAPHRASE_MODEL, system=system)


_DOMAIN_KEYWORDS = {
    "ordering":           ["before", "after", "slot", "position", "first", "last",
                           "line", "order", "ahead", "behind"],
    "knights_and_knaves": ["knight", "knave", "truth", "lie", "deceiv", "says",
                           "claim", "statement"],
    "grouping":           ["group", "team", "categor", "together", "same", "different",
                           "partition"],
}


def verify_paraphrase(text: str, extracted, question_info: dict,
                      active_domains: list) -> tuple:
    """Five structural assertions that must all pass before accepting LLM output.
    Returns (ok, failure_reason). Failures trigger a retry in Phase 7.
    """
    entities = extracted.entities
    dump     = extracted.model_dump()

    # 1. every entity name must appear in the generated text
    missing = [e for e in entities if e.lower() not in text.lower()]
    if missing:
        return False, f"missing entities: {missing}"

    # 2. no entity names from NAMES pool appear that aren't in the schema
    intruders = [n for n in NAMES
                 if n not in entities and re.search(rf"\b{n}\b", text, re.IGNORECASE)]
    if intruders:
        return False, f"hallucinated entity names: {intruders[:3]}"

    # 3. each answer choice's verbatim modality tag must appear in the text
    # the LLM is instructed to keep "[must be true]" / "[could be false]" etc. as-is
    for ch in dump["questions"][0]["answer_choices"]:
        tag = f"[{ch['type'].replace('_', ' ')}]"
        if tag not in text.lower():
            return False, f"missing modality tag for choice {ch['label']}: {tag}"

    # 4. each answer choice label must appear in the text
    labels = [ch["label"] for ch in dump["questions"][0]["answer_choices"]]
    for lbl in labels:
        if not re.search(rf"\b{lbl}\b", text):
            return False, f"missing answer label: {lbl}"

    # 5. at least one domain-vocabulary keyword must be present
    for domain in active_domains:
        if any(kw in text.lower() for kw in _DOMAIN_KEYWORDS.get(domain, [])):
            break
    else:
        return False, "no domain vocabulary keywords found"

    return True, ""


def parse_paraphrase(raw: str):
    """Parse the structured paraphrase output into
    (clue_evidence, question_evidence, choice_evidence, problem_text).

    The paraphrase LLM returns a JSON object with keys clue_evidence (list[str]),
    question_evidence (str | null), choice_evidence (dict label->str), and problem_text (str).
    Cloud backends may wrap it in ``` fences or add prose, so strip fences before json.loads
    (same treatment as pipeline.extract_logic_problem). Returns None on any malformed output.
    """
    cleaned = re.sub(r"^```(json)?\s*|```$", "", raw.strip(), flags=re.M).strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    clue_evidence     = obj.get("clue_evidence")
    question_evidence = obj.get("question_evidence")          # str or None
    choice_evidence   = obj.get("choice_evidence") or {}      # dict label -> str
    problem_text      = obj.get("problem_text")

    if not isinstance(problem_text, str) or not isinstance(clue_evidence, list):
        return None
    if not all(isinstance(e, str) for e in clue_evidence):
        return None
    if question_evidence is not None and not isinstance(question_evidence, str):
        return None
    if not isinstance(choice_evidence, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in choice_evidence.items()):
        return None

    return clue_evidence, question_evidence, choice_evidence, problem_text.strip()


def verify_evidence(extracted, clue_evidence: list, question_evidence,
                    choice_evidence: dict, problem_text: str) -> tuple:
    """Confirm every evidence string aligns with the extraction and is a verbatim substring
    of problem_text — the same substring guarantee attribution.py's Option A relies on at
    extraction time. Covers clues, the question premise, and each answer choice. Returns
    (ok, failure_reason); failures trigger a Phase-7 retry."""
    def substr_ok(label, ev):
        if not ev or not ev.strip():
            return False, f"{label} is empty"
        if ev not in problem_text:
            return False, f"{label} not a verbatim substring of problem_text: {ev!r}"
        return True, ""

    # Clues: one per top-level global constraint, in order.
    if len(clue_evidence) != len(extracted.constraints):
        return False, (f"expected {len(extracted.constraints)} clue_evidence entries, "
                       f"got {len(clue_evidence)}")
    for i, ev in enumerate(clue_evidence):
        ok, reason = substr_ok(f"clue_evidence[{i}]", ev)
        if not ok:
            return False, reason

    question = extracted.questions[0]

    # Question premise: required iff the question has a question_constraint.
    if question.question_constraints:
        ok, reason = substr_ok("question_evidence", question_evidence)
        if not ok:
            return False, reason

    # Answer choices: one entry per labelled choice that carries a constraint.
    for ch in question.answer_choices:
        if not ch.constraints:
            continue
        if ch.label not in choice_evidence:
            return False, f"choice_evidence missing label {ch.label}"
        ok, reason = substr_ok(f"choice_evidence[{ch.label}]", choice_evidence[ch.label])
        if not ok:
            return False, reason

    return True, ""


def inject_evidence(extracted, clue_evidence: list, question_evidence,
                    choice_evidence: dict, active_domains: list):
    """Return a copy of `extracted` with evidence_text written onto every constraint the
    paraphrase rendered: the i-th top-level global constraint <- clue_evidence[i], the
    question premise <- question_evidence, and each labelled answer choice <- choice_evidence.
    The text is stored verbatim; attribution.py converts it to a [start, end] span later by
    substring-matching against problem_text. Wrappers carry the whole clue's sentence; nested
    children stay None."""
    dump = extracted.model_dump()
    for i, ev in enumerate(clue_evidence):
        dump["constraints"][i]["evidence_text"] = ev

    question = dump["questions"][0]
    if question_evidence and question["question_constraints"]:
        question["question_constraints"][0]["evidence_text"] = question_evidence
    for ch in question["answer_choices"]:
        if ch.get("constraints") and ch["label"] in choice_evidence:
            ch["constraints"][0]["evidence_text"] = choice_evidence[ch["label"]]

    return build_hybrid_schema(active_domains)(**dump)


def _strip_nested_evidence(constraint: dict) -> None:
    """Drop evidence_text from every constraint nested inside `constraint` (the children of
    composite wrappers). The top-level object keeps its evidence_text -- a wrapper already
    captures the whole clue's sentence, so child evidence_text=None just wastes tokens in the
    gold JSON. Mutates in place, recursively."""
    for key in ("antecedent", "consequent", "claim"):
        child = constraint.get(key)
        if isinstance(child, dict):
            child.pop("evidence_text", None)
            _strip_nested_evidence(child)
    for child in constraint.get("claims") or []:
        if isinstance(child, dict):
            child.pop("evidence_text", None)
            _strip_nested_evidence(child)


def dump_for_emit(extracted) -> dict:
    """LogicProblem.model_dump() with evidence_text stripped from all nested child claims,
    keeping it only on the top-level constraints (global clues, question premise, answer
    choices) where attribution reads it."""
    dump = extracted.model_dump()
    top_level = list(dump.get("constraints", []) or [])
    for q in dump.get("questions", []) or []:
        top_level += q.get("question_constraints", []) or []
        for ch in q.get("answer_choices", []) or []:
            top_level += ch.get("constraints", []) or []
    for c in top_level:
        _strip_nested_evidence(c)
    return dump


# ==============================================================================
# PHASE 8 -- EMIT
#
# Appends one row to sft_dataset.jsonl (the full dataset). The dataset is now
# decoupled from the SQLite DB: split_dataset.py carves this file into train/test,
# and sft_test.db is populated only later, by run.py, during model evaluation.
# The answer is NOT stored: the training target is problem_text -> extracted_json,
# and run.py re-derives ground truth by Z3-solving the gold extraction.
#
# TODO(evidence_text): rows emitted from now on carry evidence_text on every rendered
# constraint -- each top-level global clue, the question premise, and each answer choice
# (see inject_evidence). Existing sft_dataset.jsonl / sft_train.jsonl / sft_test.jsonl
# predate this and have none, so the current FT model never emits it and attribution falls
# back to BM25. Regenerate the dataset and retrain the extraction model to activate Option A
# (LLM-supplied evidence_text) at extraction time.
# ==============================================================================

def log_and_emit(problem_text: str, extracted, question_info: dict,
                 active_domains: list, run_id: str) -> None:
    """Append one gold row to sft_dataset.jsonl. (question_info is unused — the
    answer is derived from the extraction by Z3 at evaluation time, not stored.)"""
    row = {
        "run_id":         run_id,
        "problem_text":   problem_text,
        "active_domains": json.dumps(active_domains),
        "extracted_json": json.dumps(dump_for_emit(extracted)),
        "model_name":     f"algorithmic_{PARAPHRASE_MODEL}",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }
    with open(SFT_OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


# ==============================================================================
# MAIN -- ORCHESTRATE ALL PHASES
# ==============================================================================

def generate_one_puzzle():
    """Run all eight phases for a single puzzle.
    Returns (problem_text, extracted, question_info, active_domains, run_id)
    or None if any phase fails or cannot produce a valid result.
    """
    # Phase 1
    params          = draw_puzzle_params()
    active_domains  = params["active_domains"]
    unique_solution = params["unique_solution"]

    # Phase 2
    gt = draw_domain_axioms_and_ground_truth(params)
    if not gt["entity_names"]:
        return None

    # Phase 3
    candidate_pool, false_primitives, speech_act_pairs = generate_candidate_pool(params, gt)
    if not candidate_pool:
        return None

    # Phase 4
    surviving_clues, pruned_clues = prune_to_target(
        candidate_pool, params, gt, unique_solution=unique_solution
    )
    if not surviving_clues:
        return None

    # post-pruning KK speech act enrichment: add at most KK_SPEECH_ACT_MAX pairs
    # (each pair = 2 if_then constraints encoding "X says [claim]").
    # Kept separate from the pruning pool to avoid overwhelming it with 60+ pairs.
    if speech_act_pairs and KK_SPEECH_ACT_MAX > 0:
        random.shuffle(speech_act_pairs)
        for pair in speech_act_pairs[:KK_SPEECH_ACT_MAX]:
            surviving_clues = surviving_clues + pair

    # Phase 5
    question_info = generate_question(
        surviving_clues, pruned_clues, params, gt, unique_solution=unique_solution
    )
    if question_info is None:
        return None

    # Phase 6
    try:
        extracted = assemble_extracted_json(surviving_clues, question_info, params, gt)
    except Exception as e:
        print(f"  [!] Pydantic validation failed: {e}")
        return None

    # Phase 7 -- the only LLM call in the entire pipeline.
    # Evidence-first: the LLM emits one sentence per clue (clue_evidence) plus a problem_text
    # that contains each verbatim, so every clue is attributable to an exact source span.
    system_prompt = PARAPHRASE_SYSTEM_UNIQUE if unique_solution else PARAPHRASE_SYSTEM
    prompt        = _build_paraphrase_prompt(extracted, question_info,
                                             active_domains=active_domains,
                                             unique_solution=unique_solution)
    problem_text  = None
    for attempt in range(MAX_PARAPHRASE_RETRIES):
        try:
            raw = paraphrase(prompt, system=system_prompt)
        except Exception as e:
            print(f"  [!] Paraphrase call failed: {e}")
            break

        parsed = parse_paraphrase(raw)
        if parsed is None:
            print(f"  [Paraphrase parse failed "
                  f"({attempt + 1}/{MAX_PARAPHRASE_RETRIES}): not valid JSON with "
                  f"clue_evidence/question_evidence/choice_evidence/problem_text]")
            continue
        clue_evidence, question_evidence, choice_evidence, candidate_text = parsed

        ok, reason = verify_evidence(extracted, clue_evidence, question_evidence,
                                     choice_evidence, candidate_text)
        if not ok:
            print(f"  [Evidence check failed "
                  f"({attempt + 1}/{MAX_PARAPHRASE_RETRIES}): {reason}]")
            continue

        ok, reason = verify_paraphrase(candidate_text, extracted, question_info, active_domains)
        if ok:
            extracted    = inject_evidence(extracted, clue_evidence, question_evidence,
                                           choice_evidence, active_domains)
            problem_text = candidate_text
            break
        print(f"  [Paraphrase assertion failed "
              f"({attempt + 1}/{MAX_PARAPHRASE_RETRIES}): {reason}]")

    if problem_text is None:
        return None

    return problem_text, extracted, question_info, active_domains, uuid.uuid4().hex


def main(target: int) -> None:
    # seen reflects sft_dataset.jsonl — blocks duplicate problems across runs.
    # `target` is the desired TOTAL number of rows in the dataset: rows already
    # present count toward it, so we only generate enough to reach `target`.
    seen     = load_seen_hashes(SFT_OUT)
    existing = len(seen)
    added    = 0
    print(f"Starting -- target total: {target} "
          f"| already have: {existing} | need: {max(0, target - existing)} "
          f"| out: {SFT_OUT} "
          f"| backend: {PARAPHRASE_BACKEND}/{PARAPHRASE_MODEL}")

    while existing + added < target:
        result = generate_one_puzzle()
        if result is None:
            continue
        problem_text, extracted, question_info, active_domains, run_id = result
        h = fp(problem_text)
        if h in seen:
            continue  # duplicate problem text — already in the dataset
        log_and_emit(problem_text, extracted, question_info, active_domains, run_id)
        seen.add(h)
        added += 1
        print(f"added {added}  |  total {existing + added}/{target}  |  domains: {active_domains}"
              f"  |  correct: {question_info['correct_label']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100)
    main(ap.parse_args().target)
