"""
generate.py: sends one generation request and parses the reply into a GeneratedMCQ
"""

from openai import OpenAI

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
