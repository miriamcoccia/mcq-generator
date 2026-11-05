"""
build_original_judge_requests.py: writes V1 and image judge requests for the original eval items
"""

import json
from collections import Counter
from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import GenerationRecord
from mcq_generator.judge_batch import build_judge_lines
from mcq_generator.prompting import item_to_mcq
from mcq_generator.split import split_items

OUTPUT_PATH = Path("data/batches/original_judge_requests.jsonl")

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
_, eval_set = split_items(items)

custom_ids = []
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for item in eval_set:
        record = GenerationRecord(
            item_id=item.id,
            condition="original",
            model="scienceqa",
            example_ids=[],
            ladder_level=None,
            mcq=item_to_mcq(item),
        )
        for line in build_judge_lines(record, item):
            f.write(json.dumps(line) + "\n")
            custom_ids.append(line["custom_id"])

print(len(custom_ids), "requests")
print(len(set(custom_ids)), "unique IDs")
print(Counter(c.split("__")[-1] for c in custom_ids))
print(round(OUTPUT_PATH.stat().st_size / 1024**2), "MB")
