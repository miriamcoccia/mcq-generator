"""
submit_batch.py: splits requests.jsonl into parts and submits each as a batch; saves the batch IDs
"""

from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from mcq_generator.batch import split_jsonl, submit_batch

load_dotenv()
REQUESTS = Path("data/batches/requests.jsonl")
IDS_FILE = Path("data/batches/batch_ids.txt")

client = OpenAI()
batch_ids = [submit_batch(client, part) for part in split_jsonl(REQUESTS)]
IDS_FILE.write_text("\n".join(batch_ids) + "\n", encoding="utf-8")
print("Submitted:", batch_ids)
