"""
submit_original_judge_batch.py: submits the original-item judge requests and saves the batch IDs
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from mcq_generator.batch import split_jsonl, submit_batch

load_dotenv()
REQUESTS = Path("data/batches/original_judge_requests.jsonl")
IDS_FILE = Path("data/batches/original_judge_batch_ids.txt")

client = OpenAI()
batch_ids = [submit_batch(client, part) for part in split_jsonl(REQUESTS)]
IDS_FILE.write_text("\n".join(batch_ids) + "\n", encoding="utf-8")
print("Submitted:", batch_ids)
