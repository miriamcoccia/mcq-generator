"""
judge_batch.py: builds batch request lines for the V1 and image judges
"""

from openai.lib._parsing._completions import type_to_response_format_param

from mcq_generator.domain import GenerationRecord, ScienceQAItem
from mcq_generator.judge import (
    JUDGE_MODEL,
    ImageJudgeResult,
    TraitList,
    format_conditions,
    format_image_conditions,
    format_question,
    load_image_properties,
    load_traits,
    render_image_judge_prompt,
    render_judge_prompt,
)
from mcq_generator.prompting import format_characteristics, image_part, text_part

V1_RESPONSE_FORMAT = type_to_response_format_param(TraitList)
IMAGE_RESPONSE_FORMAT = type_to_response_format_param(ImageJudgeResult)
V1_CONDITIONS = format_conditions(load_traits())
IMAGE_CONDITIONS = format_image_conditions(load_image_properties())


def build_custom_id(record: GenerationRecord, judge_name: str) -> str:
    return f"{record.item_id}__{record.condition}__{judge_name}"


def build_request_line(
    custom_id: str, content: str | list[dict], response_format: dict
) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": content}],
            "response_format": response_format,
        },
    }


def build_judge_lines(record: GenerationRecord, item: ScienceQAItem) -> list[dict]:
    characteristics = format_characteristics(item)
    question = format_question(record.mcq)

    v1_prompt = render_judge_prompt(characteristics, question, V1_CONDITIONS)
    image_prompt = render_image_judge_prompt(
        characteristics, question, IMAGE_CONDITIONS
    )

    return [
        build_request_line(
            build_custom_id(record, "v1"), v1_prompt, V1_RESPONSE_FORMAT
        ),
        build_request_line(
            build_custom_id(record, "image"),
            [text_part(image_prompt), image_part(item.image)],
            IMAGE_RESPONSE_FORMAT,
        ),
    ]


def parse_judge_result(result: dict) -> list[dict]:
    item_id, condition, judge_name = result["custom_id"].split("__")
    content = result["response"]["body"]["choices"][0]["message"]["content"]

    if judge_name == "v1":
        verdicts = TraitList.model_validate_json(content).traits_output
        triples = [(v.trait, v.valid, v.reasoning) for v in verdicts]
    else:
        verdicts = ImageJudgeResult.model_validate_json(content).verdicts
        triples = [(v.property, v.passed, v.reasoning) for v in verdicts]

    return [
        {
            "item_id": item_id,
            "condition": condition,
            "judge": judge_name,
            "criterion": criterion,
            "passed": passed,
            "reasoning": reasoning,
        }
        for criterion, passed, reasoning in triples
    ]
