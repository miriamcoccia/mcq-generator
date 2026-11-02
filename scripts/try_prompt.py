"""
try_prompt.py: builds the zero-shot and few-shot content for one eval item and prints a summary
"""

from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import Query
from mcq_generator.prompting import build_content
from mcq_generator.split import split_items

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)

item = eval_set[0]
query = Query(
    item.image, item.grade, item.subject, item.topic, item.category, item.skill
)

for name, examples in [("zero-shot", []), ("few-shot", corpus[:3])]:
    parts = build_content(query, examples)
    print(name, [p["type"] for p in parts])

print(build_content(query, [])[0]["text"])
