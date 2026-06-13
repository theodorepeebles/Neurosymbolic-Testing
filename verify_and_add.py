"""
Usage:
    python verify_and_add.py input.jsonl
Reads rows with {problem_text, answer, active_domains, extracted_json, model_name},
Z3-verifies each, appends passing rows to sft_positives.jsonl.
"""

import json
import sys
import uuid
from datetime import datetime, timezone

from validators import build_hybrid_schema
from pipeline import z3_solve

SFT_OUT = "data/sft_positives.jsonl"

def verify_file(input_path: str):
    with open(input_path) as f:
        content = f.read().strip()
    if content.startswith("["):
        rows = json.loads(content)
    else:
        rows = [json.loads(l) for l in content.splitlines() if l.strip()]

    kept, dropped = 0, 0

    for row in rows:
        label = row.get("problem_text", "")[:60]
        try:
            domains   = row["active_domains"] if isinstance(row["active_domains"], list) else json.loads(row["active_domains"])
            LP        = build_hybrid_schema(domains)
            ext_obj   = row["extracted_json"] if isinstance(row["extracted_json"], dict) else json.loads(row["extracted_json"])
            extracted = LP(**ext_obj)
            res          = z3_solve(extracted)

            if res["status"] != "sat":
                print(f"  UNSAT     : {label}"); dropped += 1; continue

            verified = [l for l, v in res["question_results"][0].items() if v]

            if len(verified) != 1:
                print(f"  {len(verified)} VERIFIED : {label}"); dropped += 1; continue

            if verified[0].upper() != row["answer"].upper():
                print(f"  WRONG ANS (Z3={verified[0]}, expected={row['answer']}): {label}")
                dropped += 1; continue

            out = {
                "run_id":        uuid.uuid4().hex,
                "problem_text":  row["problem_text"],
                "active_domains": json.dumps(domains),
                "extracted_json": json.dumps(ext_obj),
                "model_name":    row.get("model_name", "gemini"),
                "timestamp":     datetime.now(timezone.utc).isoformat(),
            }
            with open(SFT_OUT, "a") as f:
                f.write(json.dumps(out) + "\n")
            print(f"  OK        : {label}")
            kept += 1

        except Exception as e:
            print(f"  ERROR ({type(e).__name__}: {e}): {label}")
            dropped += 1

    print(f"\nKept: {kept}  Dropped/errored: {dropped}  →  {SFT_OUT}")

if __name__ == "__main__":
    verify_file(sys.argv[1])