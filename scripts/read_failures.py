"""
read_failures.py: print judge failures so their reasoning can be checked by eye
"""

import json
from pathlib import Path

import pandas as pd

with Path("data/outputs/judge_dry_run.jsonl").open(encoding="utf-8") as f:
    df = pd.DataFrame([json.loads(line) for line in f])

failed = df[~df["passed"]]
selected = failed[(failed["judge"] == "image") | (failed["criterion"] == "Correctness")]
selected = selected.sort_values(["judge", "criterion", "condition"])

print(len(selected), "failures")
for row in selected.itertuples():
    print(f"\n[{row.judge} | {row.criterion} | {row.condition} | {row.item_id}]")
    print(row.reasoning)
