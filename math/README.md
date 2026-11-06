# Math

Arithmetic word problems, the original prototype. The [logic/](../logic/README.md) pipeline is its successor.

The model translates the problem into variables and equations; Z3
does the calculation. To show the difference this makes, each run also solves the problem with the
model alone and prints the two accuracies side by side.

## How it works

`run_ns_pipeline()` in [pipeline/pipeline.py](pipeline/pipeline.py):

| Step | Where | What happens |
|---|---|---|
| Translate | `extract_problem()` | The model returns JSON, its shape forced to match the schema. Up to `MAX_ATTEMPTS = 3` tries. |
| Check | `validate_math_logic()` | Five rules from the `VALIDATORS` list in [validators.py](pipeline/validators.py) |
| Solve | `z3_solve()` | Each variable becomes a Z3 number, known values are pinned, each constraint is added as an equation |
| Retry on contradiction | `_handle_unsat_retry()` | If the equations contradict, the contradicting ones are shown to the model for one more try |
| Phrase | end of `run_ns_pipeline()` | One model call turns Z3's number into a sentence with units |

## The two repair loops

**Growing hints.** Each validation rule has its own exception class carrying a plain-English
`hint`. When the model breaks a rule, that hint joins a set that accumulates across
attempts — so by the third try the model sees every rule it has
broken so far, alongside its own bad JSON and the exact error. Failures with no matching hint
are collected and printed at the end.

**Contradiction feedback.** When Z3 finds the equations impossible, it reports the smallest
conflicting subset, rendered back into readable lines (`total_pay = daily_pay * days`) and
fed to the model for one re-translation.

## Running it

Setup is in the [root README](../README.md).

```
cd math/pipeline
python run.py
```
