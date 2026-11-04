"""
judge.py: render judge prompts and run the V1 text judge and the image judge separately
"""

import json
import string
from pathlib import Path
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from mcq_generator.domain import GeneratedMCQ
from mcq_generator.prompting import image_part, text_part

JUDGE_MODEL = "gpt-5.6-luna"

JUDGE_PROMPT_PATH = Path("prompts/judges/general_v1.txt")
TRAITS_PATH = Path("prompts/judges/general_v1_traits.json")
IMAGE_PROMPT_PATH = Path("prompts/judges/image_judge.txt")
IMAGE_PROPERTIES_PATH = Path("prompts/judges/image_properties.json")


# ---------- Shared formatting ----------


def format_choices(choices: tuple[str, ...]) -> str:
    options = ""
    uppercase = string.ascii_uppercase
    for i, choice in enumerate(choices):
        options += f"({uppercase[i]}) {choice} "
    return options.strip()


def format_context(hint: str) -> str:
    return "N/A" if not hint.strip() else hint.strip()


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


# ---------- V1 text judge ----------


def load_traits() -> dict[str, str]:
    return json.loads(TRAITS_PATH.read_text(encoding="utf-8"))


traits_list = load_traits()


def format_conditions(traits: dict[str, str]) -> str:
    return str(traits)  # V1 inserts the traits as a Python dict


def render_judge_prompt(characteristics: str, question: str, conditions: str) -> str:
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    for slot in ("{prompt}", "{questionnaire}", "{conditions}"):
        if slot not in template:
            raise ValueError(f"Slot {slot} not found in {JUDGE_PROMPT_PATH}")

    return (
        template.replace("{prompt}", characteristics)
        .replace("{questionnaire}", question)
        .replace("{conditions}", conditions)
    )


class Trait(BaseModel):
    trait: str = Field(
        description='A key "trait" with the trait of the corresponding LIST_OF_CONDITIONS item'
    )
    valid: bool = Field(
        description=' A key "valid" only with "True" if the corresponding condition of the LIST_OF_CONDITIONS item is fulfilled by the QUESTION; or "False" otherwise.'
    )
    reasoning: str = Field(
        description='A key "reasoning", empty if "valid" is "True"; or, only if "valid" is "False", with an explanation of no more than 40 words of why the value in key "valid" is "False", pointing out which concrete questions or answers raise this problem.'
    )

    @model_validator(mode="after")
    def check_dependencies(self) -> "Trait":
        error = False
        msg = ""
        if self.trait == "":
            msg = "Invalid trait (1): empty."
            error = True
        elif self.trait not in traits_list.keys():
            msg = "Invalid trait (2): name not supported: {}".format(self.trait)
            error = True
        elif self.valid:
            if self.reasoning != "":
                msg = "Invalid trait (3): reasoning should be empty when valid is True."
                error = True
        elif self.reasoning == "":
            msg = "Invalid trait (4):reasoning should not be empty when valid is False."
            error = True
        if error:
            print(msg)
            self.trait = "ERROR"
            self.valid = False
            self.reasoning = msg
        return self


class TraitList(BaseModel):
    traits_output: List[Trait] = Field(description="A list of Trait's objects.")


def judge(client: OpenAI, prompt: str, model: str = JUDGE_MODEL) -> TraitList:
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=TraitList,
    )
    return completion.choices[0].message.parsed


# ---------- Image judge ----------


def load_image_properties() -> list[dict[str, str]]:
    return json.loads(IMAGE_PROPERTIES_PATH.read_text(encoding="utf-8"))


def format_image_conditions(properties: list[dict[str, str]]) -> str:
    return "\n".join(f"- {p['name']}: {p['definition']}" for p in properties)


def render_image_judge_prompt(
    characteristics: str, question: str, conditions: str
) -> str:
    template = IMAGE_PROMPT_PATH.read_text(encoding="utf-8")

    for slot in ("{characteristics}", "{question}", "{conditions}"):
        if slot not in template:
            raise ValueError(f"Slot {slot} not found in {IMAGE_PROMPT_PATH}")

    return (
        template.replace("{characteristics}", characteristics)
        .replace("{question}", question)
        .replace("{conditions}", conditions)
    )


class PropertyVerdict(BaseModel):
    property: str
    passed: bool
    reasoning: str


class ImageJudgeResult(BaseModel):
    verdicts: list[PropertyVerdict]


def judge_image(
    client: OpenAI, prompt: str, image: Path, model: str = JUDGE_MODEL
) -> ImageJudgeResult:
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": [text_part(prompt), image_part(image)]}],
        response_format=ImageJudgeResult,
    )
    return completion.choices[0].message.parsed
