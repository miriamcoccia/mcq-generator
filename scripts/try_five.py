"""
try_five.py: zero-shot vs full RAG on the first 5 eval items, printed compactly for reading
"""

from pathlib import Path

import chromadb
from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import GeneratedMCQ, Query
from mcq_generator.generate import generate
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME, embed_images
from mcq_generator.retrieve import retrieve
from mcq_generator.split import split_items


def show(label: str, mcq: GeneratedMCQ) -> None:
    print(f"  [{label}] {mcq.question}")
    for n, choice in enumerate(mcq.choices):
        mark = "*" if n == mcq.answer else " "
        print(f"     {mark} {choice}")


items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)
by_id = {i.id: i for i in corpus}
collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
    COLLECTION_NAME
)
client = OpenAI()

batch = eval_set[:5]
vectors = embed_images(batch)

for item, vec in zip(batch, vectors):
    query = Query(
        item.image, item.grade, item.subject, item.topic, item.category, item.skill
    )
    ids, level = retrieve(collection, vec, query)
    examples = [by_id[i] for i in ids]

    print(f"\n=== {item.id} | grade {item.grade} | {item.skill} | ladder level {level}")
    print(f"  [ORIGINAL] {item.question}")
    show("ZERO", generate(client, query, examples=[]))
    show("RAG ", generate(client, query, examples=examples))
