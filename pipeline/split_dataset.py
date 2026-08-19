"""
Split the full dataset into train/test (= validation) sets.

Reads ../data/sft_dataset.jsonl, shuffles deterministically (seed=42), and writes:
  ../data/sft_train.jsonl   (train, used in the cloud)
  ../data/sft_test.jsonl    (held-out testing/validation, run locally by run.py)

Stdlib only — no `datasets` dependency. Run from the logic/ directory:
    python split_dataset.py
"""

import os
import json
import random

SEED      = 42
TEST_SIZE = 0.10

_DATA   = os.path.join(os.path.dirname(__file__), "..", "data")
SRC     = os.path.join(_DATA, "sft_dataset.jsonl")
TRAIN   = os.path.join(_DATA, "sft_train.jsonl")
TEST    = os.path.join(_DATA, "sft_test.jsonl")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln for ln in (line.rstrip("\n") for line in f) if ln.strip()]


def _write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def main() -> None:
    lines = _read_lines(SRC)
    if not lines:
        raise SystemExit(f"No rows in {SRC} — generate the dataset first.")

    # Validate each line parses; fail loud on a corrupt dataset.
    for ln in lines:
        json.loads(ln)

    random.seed(SEED)
    random.shuffle(lines)

    n_test = int(len(lines) * TEST_SIZE)
    test_lines  = lines[:n_test]
    train_lines = lines[n_test:]

    _write_lines(TRAIN, train_lines)
    _write_lines(TEST, test_lines)

    print(f"Total : {len(lines)}")
    print(f"Train : {len(train_lines)}  -> {os.path.relpath(TRAIN)}")
    print(f"Test  : {len(test_lines)}  -> {os.path.relpath(TEST)}")


if __name__ == "__main__":
    main()
