"""
collect_batch.py: downloads batch results, validates them, joins with the manifest, writes generations.jsonl
"""

from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

from openai import OpenAI

from mcq_generator.batch import fetch_results, parse_result
from mcq_generator.domain import GenerationRecord
from mcq_generator.generate import append_record

BATCH_DIR = Path("data/batches")
OUT = Path("data/outputs/generations.jsonl")
FAILED = Path("data/outputs/failed.jsonl")

manifest = {
    m["custom_id"]: m
    for m in map(
        json.loads,
        (BATCH_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines(),
    )
}
batch_ids = (BATCH_DIR / "batch_ids.txt").read_text(encoding="utf-8").split()
client = OpenAI()

OUT.unlink(missing_ok=True)  # collecting = fresh start: never mix two collections
FAILED.unlink(missing_ok=True)
saved, failed = 0, 0

for batch_id in batch_ids:
    for result in fetch_results(client, batch_id):
        custom_id = result["custom_id"]
        try:
            mcq = parse_result(result)
        except Exception as e:  # wrong status, refusal, invalid JSON, validator failure
            with FAILED.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"custom_id": custom_id, "error": str(e)}) + "\n")
            failed += 1
            continue
        m = manifest[custom_id]
        append_record(
            GenerationRecord(
                item_id=m["item_id"],
                condition=m["condition"],
                model=m["model"],
                example_ids=m["example_ids"],
                ladder_level=m["ladder_level"],
                mcq=mcq,
            ),
            OUT,
        )
        saved += 1

print(f"saved {saved}, failed {failed}, total {saved + failed} of {len(manifest)}")
