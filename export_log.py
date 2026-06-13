from logger import export_sft_positives, export_all

n = export_all()
print(f"Exported {n} total rows to all_attempts.jsonl")

n = export_sft_positives()
print(f"Exported {n} SFT positives to sft_positives.jsonl")