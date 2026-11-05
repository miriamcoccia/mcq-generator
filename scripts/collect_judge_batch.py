"""
collect_judge_batch.py: download judge results, keep the raw copy, save one row per verdict
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from mcq_generator.batch import fetch_results
from mcq_generator.judge_batch import parse_judge_result

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="judge")
args = parser.parse_args()

IDS_PATH = Path(f"data/batches/{args.prefix}_batch_ids.txt")
RAW_PATH = Path(f"data/batches/{args.prefix}_results_raw.jsonl")
OUTPUT_PATH = Path(f"data/outputs/{args.prefix}_verdicts.jsonl")

load_dotenv()
client = OpenAI()

batch_ids = IDS_PATH.read_text(encoding="utf-8").split()
results = [
    result for batch_id in batch_ids for result in fetch_results(client, batch_id)
]

with RAW_PATH.open("w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

rows, unparsed = [], []
for result in results:
    try:
        rows.extend(parse_judge_result(result))
    except Exception as error:
        unparsed.append((result["custom_id"], str(error)))

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

df = pd.DataFrame(rows)

print(len(results), "results")
print(len(unparsed), "unparsed", unparsed[:3])
print("\nVerdicts per MCQ (expect v1 = 14, image = 3):")
print(
    df.groupby(["item_id", "condition", "judge"])
    .size()
    .groupby("judge")
    .agg(["min", "max", "count"])
)
print("\nERROR verdicts:", (df["criterion"] == "ERROR").sum())
print("\nPass rate by condition:")
print(
    df.pivot_table(
        index=["judge", "criterion"],
        columns="condition",
        values="passed",
        aggfunc="mean",
    )
    .round(3)
    .to_string()
)
