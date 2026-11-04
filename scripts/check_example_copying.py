"""
check_example_copying.py: do generated questions match the original, their examples, or both?
"""

from pathlib import Path

import pandas as pd

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.generate import load_records


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
by_id = {i.id: i for i in items}
records = load_records(Path("data/outputs/generations.jsonl"))

rows = []
for r in records:
    generated = normalize(r.mcq.question)
    original = normalize(by_id[r.item_id].question)
    example_questions = {normalize(by_id[i].question) for i in r.example_ids}

    rows.append(
        {
            "condition": r.condition,
            "matches_original": generated == original,
            "matches_example": generated in example_questions,
            "generated": generated,
            "examples": sorted(example_questions),
        }
    )

df = pd.DataFrame(rows)
print(pd.crosstab(df["condition"], [df["matches_original"], df["matches_example"]]))

unexplained = df[df["matches_original"] & ~df["matches_example"]]

for row in unexplained.head(10).itertuples():
    print(f"\n[{row.condition}] {row.generated}")
    for example in row.examples:
        print(f"   example: {example}")
