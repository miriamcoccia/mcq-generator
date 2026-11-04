"""
try_judge_prompt.py: render the V1 judge prompt for one real eval item
"""

from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.judge import (
    format_characteristics,
    format_conditions,
    format_question,
    load_traits,
    render_judge_prompt,
)
from mcq_generator.split import split_items
from mcq_generator.prompting import item_to_mcq

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
_, eval_set = split_items(items)
item = eval_set[0]

prompt = render_judge_prompt(
    format_characteristics(item),
    format_question(item_to_mcq(item)),
    format_conditions(load_traits()),
)
print(prompt)
