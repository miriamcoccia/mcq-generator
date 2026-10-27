# src/mcq_generator/data.py
'''
data.py: only file that knows the raw structure of ScienceQA

'''

row = {
    "question_id": "5", "set": "test",
    "choices": "['if the spacecraft...', 'how steady...', 'whether...']",
    "answer": "1", "grade": "grade8", "image": "image.png",
    "hint": "", 
}

import ast
from pathlib import Path
import pandas as pd

from mcq_generator.domain import ScienceQAItem

def row_to_item(row: dict[str, str], images_dir: Path) -> ScienceQAItem:
    image_path = images_dir / row["set"] / row["question_id"] / "image.png"
    item = ScienceQAItem(
        id=row["question_id"], 
        question=row["question"],
        choices=tuple(ast.literal_eval(row["choices"])),
        answer=int(row["answer"]),
        hint=row["hint"],
        image=image_path,
        task=row["task"],
        grade=int(row["grade"].removeprefix("grade")),
        subject=row["subject"],
        topic=row["topic"],
        category=row["category"],
        skill=row["skill"],
        lecture=row["lecture"],
        solution=row["solution"],
        split=row["set"]
    )
    return item

def load_items(csv_path: Path, images_dir: Path) -> list[ScienceQAItem]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df =  df[df["image"] != ""]
    return [row_to_item(row, images_dir) for row in df.to_dict("records")]

def drop_image_choices(items: list[ScienceQAItem]) -> list[ScienceQAItem]:
    return [i for i in items if not (i.image.parent / "choice_0.png").exists()]