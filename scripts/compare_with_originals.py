"""
compare_with_originals.py: does a generated MCQ fail more often when its original failed the same criterion?
"""

import json
from pathlib import Path

import pandas as pd

VISUAL = r"\b(?:image|picture|diagram|figure|photo|illustration|visual|map|table)s?\b"


def load_verdicts(path: str) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f])


generated = load_verdicts("data/outputs/judge_verdicts.jsonl")
original = load_verdicts("data/outputs/original_judge_verdicts.jsonl")

baseline = original[["item_id", "judge", "criterion", "passed"]].rename(
    columns={"passed": "original_passed"}
)
merged = generated.merge(
    baseline, on=["item_id", "judge", "criterion"], validate="many_to_one"
)
merged["failed"] = ~merged["passed"]
merged["original_group"] = merged["original_passed"].map(
    {True: "orig_pass", False: "orig_fail"}
)

print(len(merged), "rows merged, expected", len(generated))

print("\nFail rate of generated MCQs, by whether the original failed")
table = merged.pivot_table(
    index=["judge", "criterion"],
    columns=["condition", "original_group"],
    values="failed",
    aggfunc="mean",
)
print(table[["zero", "rag"]].round(2).to_string())

failed_original = original[(original["judge"] == "v1") & ~original["passed"]].copy()
failed_original["mentions_visual"] = failed_original["reasoning"].str.contains(
    VISUAL, case=False, regex=True
)
print("\nOriginals: share of V1 failures whose reasoning mentions a visual")
print(
    failed_original.groupby("criterion")["mentions_visual"]
    .agg(["mean", "count"])
    .round(2)
    .to_string()
)
