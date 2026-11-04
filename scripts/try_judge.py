"""
try_judge.py: run the V1 judge on one generated RAG MCQ
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.generate import load_records
from mcq_generator.judge import (
    format_conditions,
    format_question,
    judge,
    load_traits,
    render_judge_prompt,
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

prompt = render_judge_prompt(
    format_characteristics(item),
    format_question(record.mcq),
    format_conditions(load_traits()),
)

result = judge(OpenAI(), prompt)

print(record.mcq.question)
print(len(result.traits_output), "traits")
for t in result.traits_output:
    print(f"{t.trait}: {t.valid} {t.reasoning}")
