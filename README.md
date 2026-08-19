# Math — neurosymbolic word problem solver

Language models are unreliable at arithmetic. They're good at reading. So this doesn't ask
one to solve anything — it asks it to **translate** a word problem into structured data,
hands that to [Z3](https://github.com/Z3Prover/z3) (a solver that does exact maths), and
then asks the model one last time to phrase the answer in a sentence.

To show that this actually helps, every run also solves each problem the ordinary way —
model alone, no solver — and prints the two accuracies side by side.

This is where the project started, and it's what the repo is named after. The same approach
was later carried into logic puzzles in [logic/](../logic/), which is where the work
continued.

## How it works

`run_ns_pipeline()` in [pipeline.py](pipeline/pipeline.py) runs five steps per problem:

| Step | Where | What happens |
|---|---|---|
| Translate | `extract_problem()` | The model returns JSON. Its shape is forced to match the schema, so it can't invent fields. Up to `MAX_ATTEMPTS = 3` tries. |
| Check | `validate_math_logic()` | Five rules from the `VALIDATORS` list in [validators.py](pipeline/validators.py) |
| Solve | `z3_solve()` | Each variable becomes a Z3 number, known values are pinned, and each constraint is added as an equation |
| Retry on contradiction | `_handle_unsat_retry()` | If the equations contradict each other, the contradicting ones are shown to the model for one more try |
| Phrase | end of `run_ns_pipeline()` | One model call turns Z3's number into a sentence with units |

The model never does arithmetic at any point. It produces a description of the problem, and
Z3 computes from that description.

## The two repair loops

This is the interesting part, and most of the design history below is about getting here.

**Growing hints.** Each validation rule has its own exception class, and each class carries
a plain-English `hint` explaining the rule. When the model breaks a rule, that hint is added
to a set that **accumulates across attempts** rather than resetting — so on the third try
the model is looking at every rule it has broken so far, along with its own bad JSON and the
exact error. Anything that fails without a matching hint gets collected and printed at the
end, so you know which hint to write next.

**Contradiction feedback.** If Z3 finds the equations impossible to satisfy, it can report
the smallest conflicting subset. Those get rendered back into readable lines
(`total_pay = daily_pay * days`) and fed to the model for one re-translation.

The rule doing the most work is `SelfReferentialConstraint`. Z3 solves every equation
simultaneously, not top to bottom, so `total = total * days` isn't a running total — it's a
contradiction. The model has to introduce a new variable instead. This is spelled out in
both the hint and the system prompt.

## Setup

- **[Ollama](https://ollama.com) running locally** with `qwen3:8b` pulled. The pipeline
  talks to it over plain HTTP at `localhost:11434` and gives up after 150 seconds.
- From the shared `requirements.txt` at the repo root, this part only needs `z3-solver`,
  `pydantic` and `requests`.

## Running it

Run it from inside the folder. Imports are flat (`from pipeline import ...`), and the folder
is called `math`, which would otherwise shadow Python's own `math` module:

```
cd math/pipeline
python run.py
```

There are no command-line flags — settings are constants you edit in place. The problems
live in [test_suite.py](pipeline/test_suite.py), where different ones are commented out for
debugging; uncomment whichever you want to run.

## Reading the output

Each problem prints its extraction, Z3's answer, and both verdicts. The summary at the end
gives `Baseline accuracy` (model alone) against `Neurosymbolic acc` (model + Z3) and the
`Delta` between them. An answer counts as correct within 0.01.

Below that, `UNMATCHED ERRORS — write hints for these` lists failures that no rule had a
hint for. That list is the point: it tells you which validator to write next.

## How it got here

- It started with plain JSON parsing that let bad output through and just recorded it.
- Then Pydantic plus `instructor`, which retries automatically **and tells the model what it
  got wrong** instead of just asking again. `instructor` was dropped a few days later
  (`168f1f1` — *"too complicated and wasnt working"*) for a retry loop written by hand.
- But the idea outlived the library. The hint system above is that same
  tell-it-what-it-did-wrong retry, rebuilt from scratch and made specific to each rule —
  first as surgical reprompting (`0fa7b36`), then split into small validators with one
  custom error each (`111c262`), then with each hint attached directly to its error class
  (`6c4c820`).
- Validation was deliberately moved **out** of Pydantic's own validators into plain
  functions (`83e1f8f`), because Pydantic wrapped the custom errors in its own noise and the
  model retried worse for it.
- A retry for contradictory equations came next (`d4e2c0b`), and later learned to include
  the actual conflict rather than just saying it failed (`4d2f579`).
- The model-only baseline was added last (`21e6665`), so the improvement could be measured
  rather than assumed.

Work here wound down once the features it was built to prove were working and integrated —
the approach then moved on to [logic/](../logic/). It isn't half-finished; everything in the
history after June 2026 is file reorganisation.
