from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, model_validator, ConfigDict


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
    model_config = ConfigDict(extra="forbid")
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


class GenerationRecord(BaseModel):  # one line of the result file should look like this
    item_id: str
    condition: str
    model: str
    example_ids: list[str]  # which examples were included in the prompt
    ladder_level: (
        int | None
    )  # which fallback level? (none for zero and random condition)
    mcq: GeneratedMCQ
