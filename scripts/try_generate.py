"""
try_generate.py: one zero-shot generation for one eval item, end to end
"""

from pathlib import Path

from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import Query
from mcq_generator.generate import generate
from mcq_generator.split import split_items

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)

item = eval_set[0]
query = Query(
    item.image, item.grade, item.subject, item.topic, item.category, item.skill
)

client = OpenAI()
mcq = generate(client, query, examples=[])  # zero-shot first: cheapest, simplest
print(mcq.model_dump_json(indent=2))
