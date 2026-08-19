import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parent / "sft_dataset.jsonl"


def extract_types(c):
    """Recursively yield all constraint type strings, including nested ones."""
    yield c["type"]
    if c["type"] == "if_then":
        yield from extract_types(c["antecedent"])
        yield from extract_types(c["consequent"])
    elif c["type"] == "not":
        yield from extract_types(c["claim"])


def run():
    counts = Counter()

    with open(DATASET) as f:
        for line in f:
            row = json.loads(line)
            data = json.loads(row["extracted_json"])

            for c in data.get("constraints", []):
                counts.update(extract_types(c))

            for q in data.get("questions", []):
                for c in q.get("question_constraints", []):
                    counts.update(extract_types(c))
                for choice in q.get("answer_choices", []):
                    for c in choice.get("constraints", []):
                        counts.update(extract_types(c))

    print(f"=== Constraint Type Counts ({len(counts)} unique) ===")
    for ctype, count in counts.most_common():
        print(f"  {ctype}: {count}")


if __name__ == "__main__":
    run()
