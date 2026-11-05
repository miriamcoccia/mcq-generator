"""
paired_bootstrap.py: paired differences between conditions, with 95% bootstrap intervals over items
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mcq_generator.generate import load_records

CONTRASTS = [("random", "zero"), ("metadata", "random"), ("rag", "metadata")]
METRICS = [
    "f1",
    "bleu_4",
    "bertscore",
    "ppl",
    "divs",
    "n_words",
    "answer_match",
    "any_distractor_match",
]
N_BOOT = 10_000
LEAK = re.compile(r"\b(choice|option)\s+\d\b", re.IGNORECASE)

parser = argparse.ArgumentParser()
parser.add_argument("--exclude-leaks", action="store_true")
args = parser.parse_args()

suffix = "_no_leaks" if args.exclude_leaks else ""
OUTPUT_PATH = Path(f"data/outputs/paired_bootstrap{suffix}.csv")

rng = np.random.default_rng(42)

metrics = pd.read_csv("data/outputs/metrics.csv")
metrics = metrics.melt(
    id_vars=["item_id", "condition"],
    value_vars=METRICS,
    var_name="measure",
    value_name="value",
)

with Path("data/outputs/judge_verdicts.jsonl").open(encoding="utf-8") as f:
    verdicts = pd.DataFrame([json.loads(line) for line in f])
verdicts = verdicts.assign(
    measure=verdicts["judge"] + ":" + verdicts["criterion"], value=verdicts["passed"]
)
verdicts = verdicts[["item_id", "condition", "measure", "value"]]

long = pd.concat([metrics, verdicts])
long["item_id"] = long["item_id"].astype(str)
long["value"] = long["value"].astype(float)

if args.exclude_leaks:
    records = load_records(Path("data/outputs/generations.jsonl"))
    leaked_ids = {
        r.item_id for r in records if LEAK.search(f"{r.mcq.lecture} {r.mcq.solution}")
    }
    long = long[~long["item_id"].isin(leaked_ids)]
    print(len(leaked_ids), "items excluded")

rows = []
for measure, group in long.groupby("measure", sort=False):
    wide = group.pivot(index="item_id", columns="condition", values="value").dropna()
    for a, b in CONTRASTS:
        differences = (wide[a] - wide[b]).to_numpy()
        samples = rng.integers(0, len(differences), size=(N_BOOT, len(differences)))
        means = differences[samples].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        rows.append(
            {
                "measure": measure,
                "contrast": f"{a} - {b}",
                "items": len(differences),
                "difference": differences.mean(),
                "ci_low": low,
                "ci_high": high,
                "excludes_zero": low > 0 or high < 0,
            }
        )

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_PATH, index=False)

results["shown"] = results["difference"].map("{:+.3f}".format) + results[
    "excludes_zero"
].map({True: " *", False: ""})
print(results.pivot(index="measure", columns="contrast", values="shown").to_string())
print("\n* 95% interval excludes 0, with no correction for multiple comparisons")
print("Items per measure:", results["items"].unique())
