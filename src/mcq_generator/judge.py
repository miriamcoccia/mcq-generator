"judge_prompts.py: reuses the prompt from Robust LLM-as-a-Judge Validators to validate the generated MCQs in its various declinations (zero-shot, RAG few-shot, random few-shot) at the end of the pipeline."

from mcq_generator.domain import ScienceQAItem, Query, GeneratedMCQ
import string
import json
from pathlib import Path

JUDGE_PROMPT_PATH = Path("prompts/judges/general_v1.txt")
TRAITS_PATH = Path("prompts/judges/general_v1_traits.json")


def format_characteristics(item: ScienceQAItem | Query) -> str:
    return (
        f"Student Grade Level: {item.grade}\n"
        f"Subject: {item.subject}\n"
        f"Topic: {item.topic}\n"
        f"Category: {item.category}\n"
        f"Skill: {item.skill}\n"
    )


def format_choices(choices: tuple[str, ...]) -> str:
    options = ""
    uppercase = string.ascii_uppercase
    for i, choice in enumerate(choices):
        options += f"({uppercase[i]}) {choice} "
    return options.strip()


def format_context(hint: str) -> str:
    return hint.strip() or "N/A"


def format_question(mcq: GeneratedMCQ) -> str:
    input_part = (
        f"Question: {mcq.question}\n"
        f"Context: {format_context(mcq.context)}\n"
        f"Options: {format_choices(mcq.choices)}\n"
    )

    letter = string.ascii_uppercase[mcq.answer]
    lecture = mcq.lecture.replace("\n", "\\n")
    solution = mcq.solution.replace("\n", "\\n")
    output_part = f"Answer: The answer is {letter}. BECAUSE: {lecture} {solution}"

    text = input_part + output_part
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def load_traits() -> dict[str, str]:
    return json.loads(TRAITS_PATH.read_text(encoding="utf-8"))


def format_conditions(traits: dict[str, str]) -> str:
    return str(traits)  # V1 inserts the traits as a Python dict


def render_judge_prompt(characteristics: str, question: str, conditions: str) -> str:
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{prompt}", characteristics)
        .replace("{questionnaire}", question)
        .replace("{conditions}", conditions)
    )
