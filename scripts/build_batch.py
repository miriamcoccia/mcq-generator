"""
build_batch.py: writes one request line + one manifest line per (eval item, condition). Does NOT submit.
"""

import json
from pathlib import Path

import chromadb

from mcq_generator.batch import build_request_line
from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import Query
from mcq_generator.generate import MODEL
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME, embed_images
from mcq_generator.retrieve import Condition, select_examples
from mcq_generator.split import split_items

OUT_DIR = Path("data/batches")
REQUESTS = OUT_DIR / "requests.jsonl"
MANIFEST = OUT_DIR / "manifest.jsonl"

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)
by_id = {i.id: i for i in corpus}
corpus_ids = [i.id for i in corpus]
collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
    COLLECTION_NAME
)

vectors = embed_images(eval_set)
OUT_DIR.mkdir(parents=True, exist_ok=True)

total_bytes = 0
with (
    REQUESTS.open("w", encoding="utf-8") as req_f,
    MANIFEST.open("w", encoding="utf-8") as man_f,
):
    for item, vec in zip(eval_set, vectors):
        query = Query(
            item.image, item.grade, item.subject, item.topic, item.category, item.skill
        )
        for condition in Condition:
            custom_id = f"{item.id}__{condition.value}"
            ids, level = select_examples(
                condition, collection, corpus_ids, query, item.id, vec
            )

            line = json.dumps(
                build_request_line(custom_id, query, [by_id[i] for i in ids])
            )
            req_f.write(line + "\n")
            total_bytes += len(line.encode("utf-8"))

            man_f.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "item_id": item.id,
                        "condition": condition.value,
                        "model": MODEL,
                        "example_ids": ids,
                        "ladder_level": level,
                    }
                )
                + "\n"
            )

print(f"{len(eval_set) * len(Condition)} requests, {total_bytes / 1024**2:.1f} MB")
