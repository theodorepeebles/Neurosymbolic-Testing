"""
Retrieval of few-shot extraction examples (RAG) for non-finetuned extraction models.

Builds an in-memory Chroma collection from the training set (sft_train.jsonl) on first
use. Chroma embeds each problem_text with its default embedding model (all-MiniLM-L6-v2),
and retrieve() returns the k nearest training puzzles with their gold extraction JSON.

Only the training set is indexed. The test set is never retrieved from (asserted below).
"""

import json
import os

import chromadb

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(_DATA, "sft_train.jsonl")
TEST_PATH = os.path.join(_DATA, "sft_test.jsonl")

_collection = None


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_collection():
    global _collection
    if _collection is None:
        train = _load(TRAIN_PATH)

        # Leakage guard: no test puzzle may be retrievable.
        test = _load(TEST_PATH)
        test_ids = {r["run_id"] for r in test}
        test_texts = {r["problem_text"] for r in test}
        assert not any(r["run_id"] in test_ids or r["problem_text"] in test_texts for r in train), \
            "sft_train.jsonl overlaps sft_test.jsonl"

        _collection = chromadb.EphemeralClient().get_or_create_collection("sft_train")
        _collection.add(
            ids=[r["run_id"] for r in train],
            documents=[r["problem_text"] for r in train],
            metadatas=[{"json": r["extracted_json"]} for r in train],
        )
    return _collection


def retrieve(problem_text: str, k: int = 3) -> list[tuple[str, str]]:
    """Return the k most similar training puzzles as (problem_text, extracted_json) pairs."""
    res = _get_collection().query(query_texts=[problem_text], n_results=k)
    return [(doc, meta["json"]) for doc, meta in zip(res["documents"][0], res["metadatas"][0])]
