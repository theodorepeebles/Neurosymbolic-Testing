"""
Smoke test: does each model report token counts, and does thinking on/off work?

For each model, sends one short puzzle once per thinking setting, and prints tokens
and how much thinking text came back. Run from pipeline/:  python smoke_test_tokens.py
"""

from pipeline import ask_llm, new_usage
from prompts import BASELINE_LOGIC_SYSTEM

# model -> thinking settings to test. Most models take True/False; gpt-oss can't turn
# thinking off and instead takes a level: "low" | "medium" | "high".
MODELS = {
    "gpt-oss:120b-cloud": ("low", "high"),
    "gemma4:31b-cloud":   (False, True),
}

PROMPT = (
    "Alice, Bob and Carol stand in a line. Alice is not first. Bob is directly after Alice. "
    "Who is first?\nA: Alice\nB: Bob\nC: Carol"
)

for model, think_settings in MODELS.items():
    print(f"\n=== {model} ===")
    for think in think_settings:
        usage = new_usage()
        try:
            answer = ask_llm(prompt=PROMPT, system=BASELINE_LOGIC_SYSTEM, model=model,
                             think=think, timeout=300, usage=usage)
        except RuntimeError as e:
            print(f"  think={think!s:<5}  ERROR: {e}")
            continue

        flags = []
        if usage["in_tok"] == 0 or usage["out_tok"] == 0:
            flags.append("TOKEN COUNTS MISSING")
        if think and usage["thinking_chars"] == 0:
            flags.append("thinking on but no thinking returned")
        if not think and usage["thinking_chars"] > 0:
            flags.append("thinking off but thinking still returned")

        last_line = answer.strip().splitlines()[-1] if answer.strip() else "(empty)"
        print(f"  think={think!s:<5}  in: {usage['in_tok']:<5} out: {usage['out_tok']:<6} "
              f"thinking chars: {usage['thinking_chars']:<6} answer: {last_line}")
        for f in flags:
            print(f"    !! {f}")
