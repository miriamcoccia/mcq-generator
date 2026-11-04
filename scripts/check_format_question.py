"""
check_format_question.py: confirm the refactored format_question matches the saved snapshot
"""

import json
from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.judge import format_question
from mcq_generator.prompting import item_to_mcq

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
snapshot = json.loads(
    Path("data/outputs/format_question_snapshot.json").read_text(encoding="utf-8")
)

different = [i.id for i in items if format_question(item_to_mcq(i)) != snapshot[i.id]]

print(len(snapshot), "in snapshot")
print(len(different), "different")
print(different[:5])
