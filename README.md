# Logic — neurosymbolic logic puzzle solver

Same idea as [math/](../math/), scaled up. A small language model reads a logic puzzle and
**translates** it into structured constraints. [Z3](https://github.com/Z3Prover/z3) does all
the actual reasoning. A separate engine then works out *why* each answer choice is right or
wrong, and the wording of that explanation comes from templates rather than a model — so an
explanation can never claim something the solver didn't prove.

The model is never asked for an answer. It only ever describes the puzzle.

Three puzzle types are supported: **ordering** (who sits where in a line),
**knights and knaves** (who is lying), and **grouping** (who goes in which group).

## Two halves

This folder is really two projects that meet in the middle:

- **The solver** — what you run. Takes a puzzle, produces an answer and an explanation.
- **The training program** — what produced the model the solver runs. Generates its own
  training data, fine-tunes a 0.6B model on it, then scores and diagnoses the result.

They connect at exactly one point: the training program produces a model, and the solver
loads it.

---

## Half A — the solver

`run_ns_pipeline()` in [pipeline.py](pipeline/pipeline.py):

| Step | What happens |
|---|---|
| Identify the puzzle type | Which of the three types are in play. Can be skipped by reading the types straight from the dataset (`USE_GROUND_TRUTH_DOMAINS`), so extraction can be tested on its own without a misclassification skewing the result. |
| Translate | A schema is built for **just those types**, and the model fills it in under constrained decoding — it literally cannot emit a shape the schema doesn't allow. |
| Solve | Constraints go to Z3. Each answer choice is then tested **on its own**. |
| Explain | For a wrong choice, Z3 finds the smallest set of clues that rules it out; templates turn that into prose. |

### The schema is built per puzzle, on purpose

This is the main design decision in the solver, and it's about **narrowing what the model is
allowed to get wrong**. Two things narrow it:

1. `build_hybrid_schema()` in [validators.py](pipeline/validators.py) only includes the
   constraint types belonging to the puzzle types actually present. A knights-and-knaves
   puzzle never sees grouping constraints as an option.
2. Constraint classes are grouped **by field shape**. One class per type would produce
   near-duplicates (`before` / `immediately_before` / `adjacent` take identical fields).
   One class per puzzle type would need a catch-all listing every field its types might
   use — and once every field is declared, nothing can reject a nonsensical combination
   like a fixed-position constraint carrying left/right operands. By shape, each class
   declares exactly its own fields, so invalid combinations can't be expressed at all.

Fewer available choices means fewer wrong ones — which is a large part of how a model this
small manages the task at all. (The recursive `if_then` / `not` / `and` / `or` wrappers are
rebuilt fresh on every call; otherwise the first puzzle's classes would get baked in and
reused for every puzzle after it.)

### How answer choices are judged

Each choice is labelled `must_be_true`, `could_be_true`, `must_be_false` or
`could_be_false`. The solver either asserts the choice or its opposite, then asks Z3 whether
the puzzle is still solvable. "Must be true" means asserting the *opposite* makes the puzzle
impossible. Every choice is checked independently, so one bad choice can't affect another.

### Explaining a wrong answer

[explanation.py](pipeline/explanation.py) does the reasoning and
[verbalization.py](pipeline/verbalization.py) does the wording, in that order, with no model
in either. Given a choice Z3 ruled out, it searches for the *minimal* sets of clues that
conflict with it — the smallest group of statements you could point at and say "these are
why not". It rebuilds its own constraint set to do this, so implicit rules (like "two people
can't share a seat") can appear in an explanation even though the solver never states them
out loud.

Constraints also carry an `evidence_text`: the phrase in the original puzzle they came from,
so an explanation can quote the actual sentence.
[attribution.py](pipeline/attribution.py) maps that text back to a position in the problem,
falling back to a similarity search when the model's quote isn't exact.

One thing that looks like a bug and isn't: the retry-on-contradiction path exists but is
**deliberately switched off**. The fine-tuned model was never trained on correction prompts,
so re-asking it made things worse rather than better.

---

## Half B — the training program

```
generate → split → train → test → analyse
```

- **[algorithmic_sft_generator.py](pipeline/algorithmic_sft_generator.py)** builds puzzles
  *backwards*: pick the answer first, derive every statement that's true about it, trim the
  clues down while checking the puzzle stays valid, then let Z3 decide each answer choice.
  The model is only asked to paraphrase the finished structure into English. Because it
  never touches the logic, it cannot introduce a logical error — correctness is guaranteed
  by construction rather than checked afterwards.
- **[split_dataset.py](pipeline/split_dataset.py)** carves the dataset 90/10 into
  `sft_train.jsonl` and `sft_test.jsonl` (fixed seed, so the split is reproducible).
- **NS_training.ipynb** fine-tunes Qwen3-0.6B in Colab, on the training half only. The
  notebook splits again internally for validation, so training, validation and test sets
  stay disjoint.
- **[run.py](pipeline/run.py)** scores the trained model against the held-out half —
  comparing its extraction to the known-correct one field by field — and writes a detailed
  row per puzzle into `sft_test.db`.
- **[analyze.py](pipeline/analyze.py)** fits a decision tree over those rows to find the
  conditions where the model fails most often, and
  **[open_debug_viewer.py](pipeline/open_debug_viewer.py)** renders the results as a
  browsable page.

Two rules the pipeline depends on:

- The dataset files and the results database are **kept separate and never synced**. The
  dataset is the input; the database is a record of one evaluation run.
- The held-out test split is **never uploaded** to the training environment.

---

## Entry points

Everything runs from `logic/pipeline/` — imports are flat and data paths are relative, so
running from elsewhere breaks. (A few docstrings still say `python logic/repl.py` from
before the files moved; the folder is `logic/pipeline/`.)

| Command | What it does |
|---|---|
| `python run.py` | Scores the model over the test set. No flags — settings are constants at the top of the file. |
| `python repl.py <run_id>` | Ask questions about one puzzle: `whynot`, `can`, `forces`. Structured commands by default; `--freetext` lets you type plain English instead. |
| `python chatbox.py` | The same thing in a browser, at `localhost:8765`. |
| `python analyze.py` | Finds failure patterns in `sft_test.db`. |
| `python open_debug_viewer.py` | Builds and opens the report page. Run `analyze.py` first. |
| `python explanation_debug.py [run_id]` | Prints the full explanation for one puzzle. |
| `python algorithmic_sft_generator.py --target 100` | Generates more training data. Needs `GEMINI_API_KEY`. |

## Setup

- **[Ollama](https://ollama.com) running locally.** `qwen3:8b` handles puzzle-type
  classification, free-text REPL questions and the comparison baseline.
- **Extraction uses a fine-tuned model**, `SFT_Extraction_Qwen3_0.6b-v4` — not the base
  model.
- **The trained weights are not in this repo.** They're ~610 MB each and excluded from git,
  so a fresh clone can't run the solver until you obtain them or retrain.
  `models/Modelfile` holds the Ollama configuration for them.
- From the shared `requirements.txt` at the repo root: `z3-solver`, `pydantic`, `requests`
  and `rank-bm25` for the solver; `pandas` and `scikit-learn` for `analyze.py`;
  `google-genai` only to generate data; torch and transformers only for training.

> ⚠️ The extraction system prompt in [prompts.py](pipeline/prompts.py) must stay
> **byte-identical** to the one in the training script. If they drift, the fine-tuned model
> sees a prompt it was never trained on and its accuracy drops. Change both together.

## File map

| | |
|---|---|
| Solver | `pipeline.py`, `validators.py`, `prompts.py` |
| Explanation | `explanation.py`, `verbalization.py`, `attribution.py` |
| Training | `algorithmic_sft_generator.py`, `split_dataset.py`, `NS_training.ipynb`, `ns_training.py` |
| Evaluation | `run.py`, `logger.py`, `eval_metrics.py`, `analyze.py`, `open_debug_viewer.py` |
| Interactive | `repl.py`, `chatbox.py`, `explanation_debug.py` |
| Data | `data/sft_dataset.jsonl` (full), `sft_train.jsonl`, `sft_test.jsonl`, `sft_test.db` |

`gemini_auto_sft.py`, `ollama_auto_sft.py`, `groq_auto_sft.py` and `verify_and_add.py` are
kept for reference and marked superseded in their own headers — see below for why.

---

## How it got here

- It started as the [math/](../math/) pipeline adapted to logic puzzles (`f3a98f4`) — same
  translate-then-solve shape, but with a constraint vocabulary and the per-puzzle schema
  building described above.
- **Dead end — `outlines`.** The `outlines` library plus HuggingFace was tried as an
  alternative way to force the model's output into shape, compared head to head against
  Ollama, and then removed (`2f5d71d`). What survived was the pluggable extractor design it
  forced, which is why extraction is swappable today.

### Getting training data — four attempts before the one that worked

Two things shaped this: everything had to fit inside **free API tiers**, and trying three
different providers was itself the point — to find out which was actually the best and most
efficient at the job.

1. **By hand, with a flagship model.** Problems and extractions generated in a Gemini Pro
   chat window (`52dc641`), then `verify_and_add.py` (`63d26b1`) to check them before
   appending. Write the data with a strong model, verify, add.
2. **`gemini_auto_sft.py`** automated that, but hard problems had to be dropped — the free
   Flash tier couldn't do them reliably.
3. **`ollama_auto_sft.py`** used a much larger cloud model, which handled hard problems
   again.
4. **`groq_auto_sft.py`** tried the same idea on a third provider, to compare speed and
   rate limits.
5. **`algorithmic_sft_generator.py` won** (`a49b934`), and the other four were marked
   superseded.

The reason it won: in every earlier version the model had to both *invent* a valid puzzle
and be *checked* afterwards, and anything it got subtly wrong had to be caught. The
algorithmic generator moves the model's job down to just writing the prose. Correctness
stops depending on the model at all — which made the data far more reliable, and cheap
enough to stay inside a free tier no matter which provider was used.

### After that

- Logging was restructured so everything lands in one dataset file and is split from there
  (`67ebc4e`). Earlier bad data was moved aside rather than deleted.
- The failure-analysis decision tree (`c782ce2`) and the debug viewer (`3048469`) were added
  to make evaluation results readable instead of just stored.
- Attribution — tying each constraint back to the sentence it came from — meant regenerating
  the dataset so that the problem text was built *from* the constraints rather than matched
  to them afterwards (`9d1c13d`).
- The explanation engine came next (`60a43fa`), then template wording (`131766d`), then the
  REPL and the browser chat on top (`29cfe07`, `740b206`).

Work here wound down once the features it was built to prove were working and integrated
end to end, with the same approach then carried into the next domain. It isn't
half-finished — it reached what it was for.
