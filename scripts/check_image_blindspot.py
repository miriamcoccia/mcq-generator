"""
check_image_blindspot.py: split V1 failures into those whose reasoning mentions a visual and the rest
"""

import json
from pathlib import Path

import pandas as pd

VISUAL = r"\b(?:image|picture|diagram|figure|photo|illustration|visual|map|table)s?\b"
TRAITS = ["Accessibility", "Clarity", "Correctness", "Reliability", "Validity"]

with Path("data/outputs/judge_verdicts.jsonl").open(encoding="utf-8") as f:
    df = pd.DataFrame([json.loads(line) for line in f])

df = df[(df["judge"] == "v1") & df["criterion"].isin(TRAITS)].copy()
mentions_visual = df["reasoning"].str.contains(VISUAL, case=False, regex=True)

df["fail_visual"] = ~df["passed"] & mentions_visual
df["fail_other"] = ~df["passed"] & ~mentions_visual

for column in ["fail_visual", "fail_other"]:
    print(f"\n{column}: share of MCQs")
    print(
        df.pivot_table(
            index="criterion",
            columns="condition",
            values=column,
            aggfunc="mean",
        )
        .round(3)
        .to_string()
    )
