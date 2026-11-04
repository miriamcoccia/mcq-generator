"""
build_judge_requests.py: writes one V1 and one image judge request for every generated MCQ
"""

import json
from collections import Counter
from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.generate import load_records
from mcq_generator.judge_batch import build_judge_lines

OUTPUT_PATH = Path("data/batches/judge_requests.jsonl")

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
by_id = {i.id: i for i in items}
records = load_records(Path("data/outputs/generations.jsonl"))

custom_ids = []
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for record in records:
        for line in build_judge_lines(record, by_id[record.item_id]):
            f.write(json.dumps(line) + "\n")
            custom_ids.append(line["custom_id"])

print(len(custom_ids), "requests")
print(len(set(custom_ids)), "unique IDs")
print(Counter(c.split("__")[-1] for c in custom_ids))
print(round(OUTPUT_PATH.stat().st_size / 1024**2), "MB")
