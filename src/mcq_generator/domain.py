from dataclasses import dataclass
from pathlib import Path

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