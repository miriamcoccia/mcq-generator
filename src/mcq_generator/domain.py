from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, model_validator


@dataclass(frozen=True)
class ScienceQAItem:
    id: str
    question: str
    choices: tuple[str, ...]
    answer: int
    hint: str
    image: Path
    task: str
    grade: int
    subject: str
    topic: str
    category: str
    skill: str
    lecture: str
    solution: str
    split: str

    @property
    def answer_text(self) -> str:
        return self.choices[self.answer]


@dataclass(frozen=True)
class Query:
    image: Path
    grade: int
    subject: str
    topic: str
    category: str
    skill: str


class GeneratedMCQ(BaseModel):
    question: str
    context: str
    choices: tuple[str, ...]
    answer: int
    lecture: str
    solution: str

    @model_validator(mode="after")
    def answer_in_range(self) -> "GeneratedMCQ":
        if not self.answer in range(len(self.choices)):
            raise ValueError(
                f"answer {self.answer} is not a valid index for {len(self.choices)} choices"
            )
        return self
