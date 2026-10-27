"judge_prompts.py: reuses the prompt from Robust LLM-as-a-Judge Validators to validate the generated MCQs in its various declinations (zero-shot, RAG few-shot, random few-shot) at the end of the pipeline."
from mcq_generator.domain import ScienceQAItem
import string

def format_characteristics(item: ScienceQAItem) -> str:
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

def format_question(item: ScienceQAItem) -> str:
    input_part = (
        f"Question: {item.question}\n"
        f"Context: {format_context(item.hint)}\n"
        f"Options: {format_choices(item.choices)}\n"
    )

    letter = string.ascii_uppercase[item.answer]
    lecture = item.lecture.replace("\n", "\\n")
    solution = item.solution.replace("\n", "\\n")
    output_part = f"Answer: The answer is {letter}. BECAUSE: {lecture} {solution}"

    text = input_part + output_part
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text