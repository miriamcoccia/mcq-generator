"""
batch.py: builds, submits and collects OpenAI batch jobs for MCQ generation
"""

from openai.lib._parsing._completions import type_to_response_format_param
from pathlib import Path

from mcq_generator.domain import GeneratedMCQ, Query, ScienceQAItem
from mcq_generator.generate import MODEL
from mcq_generator.prompting import build_content

RESPONSE_FORMAT = type_to_response_format_param(GeneratedMCQ)
MAX_BATCH_BYTES = 190 * 1024 * 1024


def build_request_line(
    custom_id: str, query: Query, examples: list[ScienceQAItem], model: str = MODEL
) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": build_content(query, examples)}],
            "response_format": RESPONSE_FORMAT,
        },
    }


def split_jsonl(path: Path, max_bytes: int = MAX_BATCH_BYTES) -> list[Path]:
    parts, current, size = [], [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for the "\n"
        if current and size + line_bytes > max_bytes:
            parts.append(current)
            current, size = [], 0
        current.append(line)
        size += line_bytes
    if current:
        parts.append(current)  # the last, partly filled part

    paths = []
    for n, lines in enumerate(parts, start=1):
        part_path = path.with_name(f"{path.stem}_part{n}.jsonl")
        part_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(part_path)
    return paths
