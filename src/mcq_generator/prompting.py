"""
prompting.py: builds the message parts sent to the LLM (examples, images, task)
"""

from mcq_generator.domain import GeneratedMCQ, ScienceQAItem, Query
import base64
from pathlib import Path

PROMPTS_DIR = Path("prompts/generation")


def format_characteristics(item: ScienceQAItem | Query) -> str:
    return (
        f"Student Grade Level: {item.grade}\n"
        f"Subject: {item.subject}\n"
        f"Topic: {item.topic}\n"
        f"Category: {item.category}\n"
        f"Skill: {item.skill}\n"
    )


def item_to_mcq(item: ScienceQAItem) -> GeneratedMCQ:
    return GeneratedMCQ(
        question=item.question,
        context=item.hint,
        choices=item.choices,
        answer=item.answer,
        lecture=item.lecture,
        solution=item.solution,
    )


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def build_content(query: Query, examples: list[ScienceQAItem]) -> list[dict]:
    task = (PROMPTS_DIR / "simple_mcq.txt").read_text(encoding="utf-8")
    parts = []
    if examples:
        intro = (PROMPTS_DIR / "examples_block.txt").read_text(encoding="utf-8")
        parts.append(text_part(intro))
        for n, ex in enumerate(examples, start=1):
            mcq_json = item_to_mcq(ex).model_dump_json(indent=2)
            parts.append(
                text_part(f"Example {n}:\n{format_characteristics(ex)}{mcq_json}")
            )
            parts.append(image_part(ex.image))
    parts.append(text_part(task.format(characteristics=format_characteristics(query))))
    parts.append(image_part(query.image))
    return parts
