"judge_prompts.py: reuses the prompt from Robust LLM-as-a-Judge Validators to validate the generated MCQs in its various declinations (zero-shot, RAG few-shot, random few-shot) at the end of the pipeline."

from mcq_generator.domain import ScienceQAItem, Query, GeneratedMCQ
import string
import json
from pathlib import Path
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

JUDGE_PROMPT_PATH = Path("prompts/judges/general_v1.txt")
TRAITS_PATH = Path("prompts/judges/general_v1_traits.json")
JUDGE_MODEL = "gpt-5.6-luna"


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


traits_list = load_traits()


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
