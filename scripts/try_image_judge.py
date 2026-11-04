"""
try_image_judge.py: run the image judge on the same generated RAG MCQ as try_judge.py
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.generate import load_records
from mcq_generator.judge import (
    format_image_conditions,
    format_question,
    judge_image,
    load_image_properties,
    render_image_judge_prompt,
)
from mcq_generator.prompting import format_characteristics

load_dotenv()

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
by_id = {i.id: i for i in items}

records = load_records(Path("data/outputs/generations.jsonl"))
record = next(r for r in records if r.condition == "rag")
item = by_id[record.item_id]

prompt = render_image_judge_prompt(
    format_characteristics(item),
    format_question(record.mcq),
    format_image_conditions(load_image_properties()),
)
result = judge_image(OpenAI(), prompt, item.image)

print(record.mcq.question)
print(len(result.verdicts), "verdicts")
for v in result.verdicts:
    print(f"{v.property}: {v.passed} {v.reasoning}")
