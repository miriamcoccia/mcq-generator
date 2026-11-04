"""
dry_run_judges.py: run both judges on the dry-run MCQs and save one row per verdict
"""

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.generate import load_records
from mcq_generator.judge import (
    format_conditions,
    format_image_conditions,
    format_question,
    judge,
    judge_image,
    load_image_properties,
    load_traits,
    render_image_judge_prompt,
    render_judge_prompt,
)
from mcq_generator.prompting import format_characteristics
from mcq_generator.split import split_items

OUTPUT_PATH = Path("data/outputs/judge_dry_run.jsonl")


def load_done() -> set[tuple[str, str, str]]:
    if not OUTPUT_PATH.exists():
        return set()
    with OUTPUT_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {(r["item_id"], r["condition"], r["judge"]) for r in rows}


def save_rows(rows: list[dict]) -> None:
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_row(
    record, judge_name: str, criterion: str, passed: bool, reasoning: str
) -> dict:
    return {
        "item_id": record.item_id,
        "condition": record.condition,
        "judge": judge_name,
        "criterion": criterion,
        "passed": passed,
        "reasoning": reasoning,
    }


load_dotenv()
client = OpenAI()

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
by_id = {i.id: i for i in items}
_, eval_set = split_items(items)
sample_ids = {i.id for i in eval_set[::32]}

records = [
    r
    for r in load_records(Path("data/outputs/generations.jsonl"))
    if r.item_id in sample_ids
]
print(len(records), "MCQs to judge")

traits = format_conditions(load_traits())
properties = format_image_conditions(load_image_properties())
done = load_done()

for n, record in enumerate(records, start=1):
    item = by_id[record.item_id]
    characteristics = format_characteristics(item)
    question = format_question(record.mcq)
    print(f"{n}/{len(records)} {record.item_id} {record.condition}")

    if (record.item_id, record.condition, "v1") not in done:
        result = judge(client, render_judge_prompt(characteristics, question, traits))
        save_rows(
            [
                make_row(record, "v1", t.trait, t.valid, t.reasoning)
                for t in result.traits_output
            ]
        )

    if (record.item_id, record.condition, "image") not in done:
        prompt = render_image_judge_prompt(characteristics, question, properties)
        result = judge_image(client, prompt, item.image)
        save_rows(
            [
                make_row(record, "image", v.property, v.passed, v.reasoning)
                for v in result.verdicts
            ]
        )

with OUTPUT_PATH.open(encoding="utf-8") as f:
    df = pd.DataFrame([json.loads(line) for line in f])

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
    ).round(2)
)
