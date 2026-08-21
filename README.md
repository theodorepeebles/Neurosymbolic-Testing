# Neurosymbolic Testing

Language models are unreliable at exact reasoning. So nothing
here asks a model to solve anything — it asks the model to **translate** a problem written in
English into structured data, hands that to [Z3] solver, and comes back 
to the model to phrase the result.


| Folder | Problem | Status |
|---|---|---|
| [math/](math/) | Arithmetic word problems | The original prototype 
| [logic/](logic/) | Logic puzzles (ordering, knights & knaves, grouping) | The successor: the same shape, plus a fine-tuned model and an explanation engine |

Each folder also has its own README with similar material, if you'd rather read just one.

---

# Math

Arithmetic word problems. The model translates the problem into variables and equations; Z3
does the calculation. To show this actually helps, each run also solves the problem with the
model alone and prints the two accuracies side by side.

## How it works

`run_ns_pipeline()` in [math/pipeline/pipeline.py](math/pipeline/pipeline.py):

| Step | Where | What happens |
|---|---|---|
| Translate | `extract_problem()` | The model returns JSON, its shape forced to match the schema. Up to `MAX_ATTEMPTS = 3` tries. |
| Check | `validate_math_logic()` | Five rules from the `VALIDATORS` list in [validators.py](math/pipeline/validators.py) |
| Solve | `z3_solve()` | Each variable becomes a Z3 number, known values are pinned, each constraint is added as an equation |
| Retry on contradiction | `_handle_unsat_retry()` | If the equations contradict, the contradicting ones are shown to the model for one more try |
| Phrase | end of `run_ns_pipeline()` | One model call turns Z3's number into a sentence with units |

## The two repair loops

**Growing hints.** Each validation rule has its own exception class carrying a plain-English
`hint`. When the model breaks a rule, that hint joins a set that **accumulates across
attempts** rather than resetting — so by the third try the model sees every rule it has
broken so far, alongside its own bad JSON and the exact error. Failures with no matching hint
are collected and printed at the end, telling you which hint to write next.

**Contradiction feedback.** When Z3 finds the equations impossible, it reports the smallest
conflicting subset, rendered back into readable lines (`total_pay = daily_pay * days`) and
fed to the model for one re-translation.

The rule doing the most work is `SelfReferentialConstraint`: Z3 solves every equation at once,
not top to bottom, so `total = total * days` isn't a running total — it's a contradiction, and
the model has to introduce a new variable instead.

## Running it

```
cd math/pipeline
python run.py
```

---

# Logic

Logic puzzles. Same translate-then-solve shape, but with a **fine-tuned** model doing the
translating and a separate engine that works out *why* each answer choice is right or wrong.

Two parts:

- **The solver** — what you run.
- **The training program** — what produced the model the solver runs.

## The solver

`run_ns_pipeline()` in [logic/pipeline/pipeline.py](logic/pipeline/pipeline.py):

| Step | What happens |
|---|---|
| Identify the puzzle type | Which of the three types are in play. Can be skipped by reading the types from the dataset (`USE_GROUND_TRUTH_DOMAINS`), so extraction can be tested without a misclassification skewing the result. |
| Translate | A schema is built for **just those types**, and the model fills it in under constrained decoding — it can't emit a shape the schema disallows. |
| Solve | Constraints go to Z3. Each answer choice is tested **on its own**. |
| Explain | For a wrong choice, Z3 finds the smallest set of clues ruling it out; templates turn that into prose. |

### The schema is built per puzzle, on purpose

The main design decision in the solver, and it's about **narrowing what the model is allowed
to get wrong**. Two things narrow it:

1. `build_hybrid_schema()` in [validators.py](logic/pipeline/validators.py) includes only the
   constraint types belonging to the puzzle types actually present. A knights-and-knaves
   puzzle never sees grouping constraints as an option.
2. Constraint classes are grouped **by field shape**. One class per type would produce
   near-duplicates (`before` / `immediately_before` / `adjacent` take identical fields). One
   class per puzzle type would need a catch-all listing every field its types might use — and
   once every field is declared, nothing can reject a nonsensical combination like a
   fixed-position constraint carrying left/right operands. By shape, each class declares
   exactly its own fields, so invalid combinations can't be expressed at all.

Fewer available choices means fewer wrong ones — a large part of how a model this small
manages the task at all. (The recursive `if_then` / `not` / `and` / `or` wrappers are rebuilt
fresh on every call; otherwise the first puzzle's classes would be reused for every puzzle
after it.)

### Judging answers, and explaining them

Each choice is labelled `must_be_true`, `could_be_true`, `must_be_false` or `could_be_false`.
The solver asserts either the choice or its opposite, then asks Z3 whether the puzzle is
still solvable — "must be true" means asserting the *opposite* makes it impossible. Every
choice is checked independently.

[explanation.py](logic/pipeline/explanation.py) does the reasoning and
[verbalization.py](logic/pipeline/verbalization.py) the wording, in that order, with no model
in either. Given a ruled-out choice, it finds the *minimal* sets of clues that conflict with
it — the smallest group you could point at and say "these are why not". It rebuilds its own
constraint set to do this, so implicit rules ("two people can't share a seat") can appear in
an explanation even though the solver never states them out loud. Constraints also carry the
phrase they came from, so an explanation can quote the actual sentence;
[attribution.py](logic/pipeline/attribution.py) maps that text back to a position in the
problem, falling back to a similarity search when the quote isn't exact.

Looks like a bug, isn't: the retry-on-contradiction path exists but is **deliberately switched
off** — the fine-tuned model was never trained on correction prompts, so re-asking made things
worse.

## The training program

```
generate → split → train → test → analyse
```

- **[algorithmic_sft_generator.py](logic/pipeline/algorithmic_sft_generator.py)** builds
  puzzles *backwards*: pick the answer first, derive every statement true about it, trim the
  clues while checking the puzzle stays valid, then let Z3 decide each answer choice. The
  model only paraphrases the finished structure into English. Because it never touches the
  logic, it cannot introduce a logical error — correctness is guaranteed by construction
  rather than checked afterwards.
- **[split_dataset.py](logic/pipeline/split_dataset.py)** carves the dataset 90/10 into
  `sft_train.jsonl` and `sft_test.jsonl` (fixed seed, so the split is reproducible).
- **NS_training.ipynb** fine-tunes Qwen3-0.6B in Colab on the training half only, splitting
  again internally for validation so training, validation and test stay disjoint.
- **[run.py](logic/pipeline/run.py)** scores the trained model against the held-out half,
  comparing its extraction to the known-correct one field by field, and writes a detailed row
  per puzzle into `sft_test.db`.
- **[analyze.py](logic/pipeline/analyze.py)** fits a decision tree over those rows to find
  where the model fails most, and **open_debug_viewer.py** renders it as a browsable page.

Two rules the pipeline depends on: the dataset files and the results database are **kept
separate and never synced** (one is input, the other a record of one evaluation run), and the
held-out test split is **never uploaded** to the training environment.

## Running it

All from `logic/pipeline/`. A few docstrings still say `python logic/repl.py` from before the
files moved.

| Command | What it does |
|---|---|
| `python run.py` | Scores the model over the test set. Settings are constants at the top. |
| `python repl.py <run_id>` | Ask about one puzzle: `whynot`, `can`, `forces`. `--freetext` for plain English. |
| `python chatbox.py` | The same in a browser, at `localhost:8765`. |
| `python analyze.py` | Finds failure patterns in `sft_test.db`. |
| `python open_debug_viewer.py` | Builds and opens the report page. Run `analyze.py` first. |
| `python explanation_debug.py [run_id]` | Prints the full explanation for one puzzle. |
| `python algorithmic_sft_generator.py --target 100` | Generates more training data. Needs `GEMINI_API_KEY`. |

## Extra setup for logic

Beyond the shared setup above:

- Extraction uses a **fine-tuned model**, `SFT_Extraction_Qwen3_0.6b-v4` — not the base one.
  `qwen3:8b` still handles puzzle-type classification, free-text REPL questions and the
  baseline.
- **The trained weights are not in this repo.** They're ~610 MB each and excluded from git,
  so a fresh clone can't run the solver until you obtain them or retrain.
  `logic/models/Modelfile` holds the Ollama configuration for them.

> ⚠️ The extraction system prompt in [prompts.py](logic/pipeline/prompts.py) must stay
> **byte-identical** to the one in the training script. If they drift, the fine-tuned model
> sees a prompt it was never trained on and accuracy drops. Change both together.

`gemini_auto_sft.py`, `ollama_auto_sft.py`, `groq_auto_sft.py` and `verify_and_add.py` are
kept for reference and marked superseded in their own headers — see below for why.

---

# How it all evolved

**Arithmetic first.** It started with plain JSON parsing that let bad output through and just
recorded it. Then Pydantic plus `instructor`, which retries automatically **and tells the
model what it got wrong** instead of just asking again. `instructor` was dropped days later
(`168f1f1` — *"too complicated and wasnt working"*) for a retry loop written by hand.

But the idea outlived the library, and became the backbone of everything after. The hint
system is that same tell-it-what-it-did-wrong retry, rebuilt by hand and made specific to each
rule — first as surgical reprompting (`0fa7b36`), then split into small validators with one
custom error each (`111c262`), then with each hint attached to its error class (`6c4c820`).
Validation was deliberately moved **out** of Pydantic's own validators into plain functions
(`83e1f8f`), because Pydantic buried the custom errors in its own noise and the model retried
worse for it. A retry for contradictory equations followed (`d4e2c0b`), later learning to
include the actual conflict rather than just reporting failure (`4d2f579`). The model-only
baseline came last (`21e6665`), so the improvement could be measured rather than assumed.

**Then logic puzzles.** The math pipeline was adapted to a new domain (`f3a98f4`) — same
shape, but with a constraint vocabulary and the per-puzzle schema building described above.

*Dead end — `outlines`.* The `outlines` library plus HuggingFace was tried as another way to
force output into shape, compared head to head against Ollama, then removed (`2f5d71d`). What
survived was the pluggable extractor design it forced, which is why extraction is swappable
today.

**Getting training data took four attempts before the one that worked.** Two things shaped
this: everything had to fit inside **free API tiers**, and trying three different providers
was itself the point — to find which was actually best and most efficient at the job.

1. **By hand, with a flagship model.** Problems and extractions generated in a Gemini Pro
   chat window (`52dc641`), then `verify_and_add.py` (`63d26b1`) to check them before
   appending. Write the data with a strong model, verify, add.
2. **`gemini_auto_sft.py`** automated that, but hard problems had to be dropped — the free
   Flash tier couldn't do them reliably.
3. **`ollama_auto_sft.py`** used a much larger cloud model, which handled hard problems again.
4. **`groq_auto_sft.py`** tried the same on a third provider, comparing speed and rate limits.
5. **`algorithmic_sft_generator.py` won** (`a49b934`), and the other four were marked
   superseded.

The reason it won: in every earlier version the model had to both *invent* a valid puzzle and
be *checked* afterwards, and anything subtly wrong had to be caught. The algorithmic generator
moves the model's job down to just writing the prose. Correctness stops depending on the model
at all — making the data far more reliable, and cheap enough to stay inside a free tier no
matter the provider.

**After that.** Logging was restructured so everything lands in one dataset file and is split
from there (`67ebc4e`), earlier bad data moved aside rather than deleted. The failure-analysis
decision tree (`c782ce2`) and debug viewer (`3048469`) made evaluation results readable rather
than merely stored. Attribution — tying each constraint back to the sentence it came from —
meant regenerating the dataset so problem text was built *from* the constraints instead of
matched to them afterwards (`9d1c13d`). Then the explanation engine (`60a43fa`), template
wording (`131766d`), and the REPL and browser chat on top (`29cfe07`, `740b206`).

Work wound down once the features it was built to prove were working and integrated end to
end, with the same approach then carried into a further domain. Neither subsystem is
half-finished — both reached what they were for.
