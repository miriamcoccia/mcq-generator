"""
generate.py: sends one generation request and parses the reply into a GeneratedMCQ
"""

from openai import OpenAI

from pathlib import Path
from mcq_generator.domain import GenerationRecord
from mcq_generator.domain import GeneratedMCQ, Query, ScienceQAItem
from mcq_generator.prompting import build_content

MODEL = "gpt-5.6-luna"


def generate(
    client: OpenAI, query: Query, examples: list[ScienceQAItem], model: str = MODEL
) -> GeneratedMCQ:
    response = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": build_content(query, examples)}],
        response_format=GeneratedMCQ,
    )
    return response.choices[0].message.parsed


def append_record(record: GenerationRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def load_records(path: Path) -> list[GenerationRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GenerationRecord.model_validate_json(line) for line in lines]
