# Logic

Benchmark results against direct-LLM baselines are in the [root README](../README.md).

Logic puzzles. Same translate-then-solve shape as [math/](../math/README.md), but with a **fine-tuned** model doing the
translating and a separate engine that works out *why* each answer choice is right or wrong.
Built for logic problems of grouping, ordering, and knights and knaves (truth-tellers and deceivers).

Two parts:

- **The solver** — the pipeline that solves problems
- **The training program** — what produced the model the solver runs

## The solver

`run_ns_pipeline()` in [pipeline/pipeline.py](pipeline/pipeline.py):

| Step | What happens |
|---|---|
| Identify the puzzle type | Which of the three types are in play. Can be skipped by reading the types from the dataset (`USE_GROUND_TRUTH_DOMAINS`), so extraction can be tested without a misclassification skewing the result. |
| Translate | A schema is built for **just those types**, and the model fills it in under constrained decoding — it can't emit a shape the schema disallows. |
| Solve | Constraints go to Z3. Each answer choice is tested **on its own**. |
| Explain | For a wrong choice, Z3 finds the smallest set of clues ruling it out; templates turn that into prose. |

### Schema

The JSON schema used in constrained decoding is designed to narrow what the model is allowed
to get wrong. Two things narrow it:

1. `build_hybrid_schema()` in [validators.py](pipeline/validators.py) includes only the
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

[explanation.py](pipeline/explanation.py) does the reasoning and
[verbalization.py](pipeline/verbalization.py) the wording, in that order.
Given a ruled-out choice, it finds the *minimal* sets of clues that conflict with
it. It rebuilds its own constraint set to do this, so implicit rules ("two people can't share a seat")
can appear in an explanation even though the solver never states them out loud. Constraints also carry the
phrase they came from, so an explanation can quote the actual sentence;
[attribution.py](pipeline/attribution.py) maps that text back to a position in the
problem, falling back to a similarity search when the quote isn't exact.

Note: the retry-on-contradiction path exists but is deliberately switched
off — the fine-tuned model was never trained on correction prompts.

## The training program

```
generate → split → train → test → analyse
```

- **[algorithmic_sft_generator.py](pipeline/algorithmic_sft_generator.py)** builds
  puzzles backwards: the complete satisfying answer is generated first, then every true statement is derived about it,
  then the clues are trimmed while checking to ensure the puzzle stays valid, then the answer choices are chosen. An
  LLM paraphrases the finished json structure into English. I used Gemini 3.1 Flash Lite to
  paraphrase; other models work too (set `PARAPHRASE_BACKEND` / `PARAPHRASE_MODEL`).
- **[split_dataset.py](pipeline/split_dataset.py)** carves the dataset 90/10 into
  `sft_train.jsonl` and `sft_test.jsonl`
- **NS_training.ipynb** fine-tunes Qwen3-0.6B in Colab on the training half only, splitting
  again internally for validation so training, validation and test stay disjoint.
- **[run.py](pipeline/run.py)** scores the trained model against the held-out half,
  writing one result line per puzzle to `data/demo_runs/`. With `LOG_TO_DB = True` (off by
  default), it also compares the model's extraction to the known-correct one field by field
  and writes a detailed row per puzzle into `sft_test.db`.
- **[analyze.py](pipeline/analyze.py)** fits a decision tree over those rows to find
  where the model fails most, and **open_debug_viewer.py** renders it as a browsable page.

The dataset files and the results database are kept separate and never
synced (one is input, the other a record of one evaluation run), and the
held-out test split is never uploaded to the training environment.

## Running it

All from `logic/pipeline/`. A few docstrings still say `python logic/repl.py` from before the
files moved.

`sft_test.db` is not in the repo. `repl.py`, `chatbox.py`, `analyze.py`, `open_debug_viewer.py`
and `explanation_debug.py` all read from it, so first run `run.py` with `LOG_TO_DB = True`.

| Command | What it does |
|---|---|
| `python run.py` | Scores the model over the test set. Settings are constants at the top. |
| `python plot_results.py` | Builds the benchmark graphs and tables in `data/plots/` from the files in `data/demo_runs/`. |
| `python repl.py <run_id>` | Ask about one puzzle: `whynot`, `can`, `forces`. `--freetext` for plain English. |
| `python chatbox.py` | The same in a browser, at `localhost:8765`. |
| `python analyze.py` | Finds failure patterns in `sft_test.db`. |
| `python open_debug_viewer.py` | Builds and opens the report page. Run `analyze.py` first. |
| `python explanation_debug.py [run_id]` | Prints the full explanation for one puzzle. |
| `python algorithmic_sft_generator.py --target 100` | Generates more training data. Needs `GEMINI_API_KEY`. |

## Replay demo

An animated page that replays the pipeline on 10 test puzzles from the
`ns__SFT-0.6b-v4__cls-qwen3-4b-q4_K_M` run (8 answered correctly, 2 answered wrong): the
classifier's puzzle types, each extracted rule linked to the phrase it came from, Z3's solution,
and for every answer choice the proof that confirms or eliminates it. It lives in
[replay/](replay/) and is separate from the chatbox.

On the page: the numbered tabs switch examples, the stage names in the bottom bar replay one
stage on its own, and hovering a rule or a phrase shows what it is linked to. The last tab,
"Full results", is a static copy of the benchmark graphs, table and analysis from the
[root README](../README.md); it is written into
[replay.template.html](replay/replay.template.html), so update it there when the README's
results change. Its graphs are served from [data/plots/](data/plots/).

The page never calls a model. The models' outputs for the 10 puzzles are saved in
[data/demo_model_outputs.json](data/demo_model_outputs.json); everything after them is rebuilt
by `build_trace()` in [replay_trace.py](replay/replay_trace.py) on each request. The 10 puzzle
ids are in [demo_examples.py](replay/demo_examples.py).

All commands are run from the repo root.

**1. Save the model outputs (once).** Only needed again if the puzzle list changes. Needs Ollama
running with `qwen3:4b-q4_K_M` and `SFT_Extraction_Qwen3_0.6b-v4`.

```
python logic/replay/save_model_outputs.py
```

It runs the classifier and the extractor on each puzzle that is not in the file yet (`--all`
reruns all 10) and checks it against the source run: same predicted puzzle types, Z3 status and
answer. The file is written only if all 10 match. The classifier is not run at temperature 0,
so it is retried up to 10 times until it gives the puzzle types recorded in the source run; the
number of tries is saved in the file. If a puzzle still does not match, run the script again.

**2. Run locally.** No Ollama needed.

```
python logic/replay/replay_server.py
```

Opens `http://localhost:8770`. Add `--no-browser` to skip opening a browser. The server binds
to `127.0.0.1` unless `HOST` is set, and to `PORT` if that is set.

**3. Deploy to Cloud Run.**

```
gcloud meta list-files-for-upload logic   # show the files that would be uploaded
bash logic/deploy.sh
```

[deploy.sh](deploy.sh) builds [Dockerfile](Dockerfile) and deploys it as the `ns-logic-demo`
service. The image runs the replay server; the chatbox is no longer deployed and runs locally
only. [.gcloudignore](.gcloudignore) limits the upload to `pipeline/`, `replay/`, the two data
files the server reads and the graphs in `data/plots/`.

Live extraction is not built. `MODEL_OUTPUTS` in
[replay_server.py](replay/replay_server.py) is the one switch for where model outputs come
from (`saved` now, `live` later); a live mode would pass fresh model output to the same
`build_trace()`.

## Extra setup for logic

Beyond the shared setup in the [root README](../README.md):

- Extraction uses a fine-tuned Qwen3-0.6B model. In `run.py`, `qwen3:4b-q4_K_M` classifies the
  puzzle type (see [classifier_compare_results.md](pipeline/classifier_compare_results.md));
  the REPL and chatbox still use `qwen3:8b` for classification and free-text questions.
- **The trained weights are not in this repo.** They're ~610 MB and excluded from git,
  so a fresh clone can't run the solver until you retrain.

> The extraction system prompt in [prompts.py](pipeline/prompts.py) must stay
> **byte-identical** to the one in the training script.

`gemini_auto_sft.py`, `ollama_auto_sft.py`, `groq_auto_sft.py` and `verify_and_add.py` are
kept for reference and marked superseded in their own headers.
