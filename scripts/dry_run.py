"""
dry_run.py: every 32nd eval item x 4 conditions, saved to JSONL; rerun skips what's already done
"""

from pathlib import Path

import chromadb
from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import GenerationRecord, Query
from mcq_generator.generate import MODEL, append_record, generate, load_records
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME, embed_images
from mcq_generator.retrieve import Condition, select_examples
from mcq_generator.split import split_items
from dotenv import load_dotenv

load_dotenv()

OUT = Path("data/outputs/dry_run.jsonl")

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)
by_id = {i.id: i for i in corpus}
corpus_ids = [i.id for i in corpus]
collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
    COLLECTION_NAME
)
client = OpenAI()

sample = eval_set[::32]  # spread across skills, not just the first one
vectors = embed_images(sample)

done = {(r.item_id, r.condition) for r in load_records(OUT)} if OUT.exists() else set()

for item, vec in zip(sample, vectors):
    query = Query(
        item.image, item.grade, item.subject, item.topic, item.category, item.skill
    )
    for condition in Condition:
        if (item.id, condition.value) in done:
            continue
        ids, level = select_examples(
            condition, collection, corpus_ids, query, item.id, vec
        )
        mcq = generate(client, query, [by_id[i] for i in ids])
        append_record(
            GenerationRecord(
                item_id=item.id,
                condition=condition.value,
                model=MODEL,
                example_ids=ids,
                ladder_level=level,
                mcq=mcq,
            ),
            OUT,
        )
        print(item.id, condition.value, "saved")
