# Neurosymbolic Testing

| Folder | Problem | Status |
|---|---|---|
| [math/](math/) | Arithmetic word problems | The original prototype 
| [logic/](logic/) | Logic puzzles (ordering, knights & knaves, grouping) | The successor: the same shape, plus a fine-tuned model and an explanation engine |

## Setup

Requires Python 3.12. Install dependencies with `pip install -r requirements.txt`.

---

# Math

Arithmetic word problems. The model translates the problem into variables and equations; Z3
does the calculation. To show the difference this makes, each run also solves the problem with the
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
`hint`. When the model breaks a rule, that hint joins a set that accumulates across
attempts  — so by the third try the model sees every rule it has
broken so far, alongside its own bad JSON and the exact error. Failures with no matching hint
are collected and printed at the end.

**Contradiction feedback.** When Z3 finds the equations impossible, it reports the smallest
conflicting subset, rendered back into readable lines (`total_pay = daily_pay * days`) and
fed to the model for one re-translation.

## Running it

```
cd math/pipeline
python run.py
```

---

# Logic

Logic puzzles. Same translate-then-solve shape, but with a **fine-tuned** model doing the
translating and a separate engine that works out *why* each answer choice is right or wrong.
Built for logic problems of grouping, ordering, and knights and knaves (truth-tellers and deceivers.

Two parts:

- **The solver** — the pipeline that solves problems
- **The training program** — what produced the model the solver runs

## The solver

`run_ns_pipeline()` in [logic/pipeline/pipeline.py](logic/pipeline/pipeline.py):

| Step | What happens |
|---|---|
| Identify the puzzle type | Which of the three types are in play. Can be skipped by reading the types from the dataset (`USE_GROUND_TRUTH_DOMAINS`), so extraction can be tested without a misclassification skewing the result. |
| Translate | A schema is built for **just those types**, and the model fills it in under constrained decoding — it can't emit a shape the schema disallows. |
| Solve | Constraints go to Z3. Each answer choice is tested **on its own**. |
| Explain | For a wrong choice, Z3 finds the smallest set of clues ruling it out; templates turn that into prose. |

### Schema

The JSON schema used in constrained decoding is designed to narrowing what the model is allowed
to get wrong. Two things narrow it:

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
manages the task at all.

### Judging answers, and explaining them

Each answer choice is labelled `must_be_true`, `could_be_true`, `must_be_false` or `could_be_false`.
The solver asserts either the choice or its opposite, then asks Z3 whether the puzzle is
still solvable — "must be true" means asserting the *opposite* makes it impossible.

[explanation.py](logic/pipeline/explanation.py) does the reasoning and
[verbalization.py](logic/pipeline/verbalization.py) the wording, in that order.
Given a ruled-out choice, it finds the *minimal* sets of clues that conflict with
it. It rebuilds its own constraint set to do this, so implicit rules ("two people can't share a seat") 
can appear in an explanation even though the solver never states them out loud. Constraints also carry the
phrase they came from, so an explanation can quote the actual sentence;
[attribution.py](logic/pipeline/attribution.py) maps that text back to a position in the
problem, falling back to a similarity search when the quote isn't exact.

Note: the retry-on-contradiction path exists but is deliberately switched
off — the fine-tuned model was never trained on correction prompts.

## The training program

```
generate → split → train → test → analyse
```

- **[algorithmic_sft_generator.py](logic/pipeline/algorithmic_sft_generator.py)** builds
  puzzles backwards: the complete satisfying answer is generated first, then every true statement is derived about it,
  then the clues are trimmed while checking to ensure the puzzle stays valid, then the answer choices are chosen. The
  model paraphrases the finished json structure into English.
- **[split_dataset.py](logic/pipeline/split_dataset.py)** carves the dataset 90/10 into
  `sft_train.jsonl` and `sft_test.jsonl`
- **NS_training.ipynb** fine-tunes Qwen3-0.6B in Colab on the training half only, splitting
  again internally for validation so training, validation and test stay disjoint.
- **[run.py](logic/pipeline/run.py)** scores the trained model against the held-out half,
  comparing its extraction to the known-correct one field by field, and writes a detailed row
  per puzzle into `sft_test.db`.
- **[analyze.py](logic/pipeline/analyze.py)** fits a decision tree over those rows to find
  where the model fails most, and **open_debug_viewer.py** renders it as a browsable page.

The dataset files and the results database are kept separate and never 
synced (one is input, the other a record of one evaluation run), and the
held-out test split is never uploaded to the training environment.

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

- Extraction uses a fine-tuned Qwen3_0.6b model
  `qwen3:8b` still handles puzzle-type classification, free-text REPL questions and the
  baseline.
- **The trained weights are not in this repo.** They're ~610 MB each and excluded from git,
  so a fresh clone can't run the solver until you obtain them or retrain.

> The extraction system prompt in [prompts.py](logic/pipeline/prompts.py) must stay
> **byte-identical** to the one in the training script.

`gemini_auto_sft.py`, `ollama_auto_sft.py`, `groq_auto_sft.py` and `verify_and_add.py` are
kept for reference and marked superseded in their own headers.