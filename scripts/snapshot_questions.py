"""
snapshot_questions.py: save format_question output for every item, before refactoring it
"""

import json
from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.judge import format_question

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)

snapshot = {item.id: format_question(item) for item in items}

path = Path("data/outputs/format_question_snapshot.json")
path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
print(len(snapshot), "saved to", path)
