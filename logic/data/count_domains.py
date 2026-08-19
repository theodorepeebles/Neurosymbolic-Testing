import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parent / "sft_dataset.jsonl"


def run():
    domain_counts = Counter()
    combo_counts = Counter()

    with open(DATASET) as f:
        for line in f:
            row = json.loads(line)
            domains = json.loads(row["active_domains"])
            for d in domains:
                domain_counts[d] += 1
            combo_counts[tuple(sorted(domains))] += 1

    print("=== Individual Domain Counts ===")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count}")

    print(f"\n=== Domain Combination Counts ({len(combo_counts)} unique) ===")
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
        label = " + ".join(combo) if combo else "(none)"
        print(f"  {label}: {count}")


if __name__ == "__main__":
    run()
