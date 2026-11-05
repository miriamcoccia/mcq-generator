"""
check_judge_batch.py: print the status and request counts of a judge batch
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="judge")
args = parser.parse_args()

load_dotenv()
client = OpenAI()

ids_path = Path(f"data/batches/{args.prefix}_batch_ids.txt")
batch_ids = ids_path.read_text(encoding="utf-8").split()

for batch_id in batch_ids:
    batch = client.batches.retrieve(batch_id)
    print(batch_id, batch.status, batch.request_counts)
