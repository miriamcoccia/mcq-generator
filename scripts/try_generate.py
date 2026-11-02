"""
try_generate.py: zero-shot vs full-RAG few-shot for one eval item, end to end
"""

from pathlib import Path

import chromadb
from openai import OpenAI

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import Query
from mcq_generator.generate import generate
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME, embed_images
from mcq_generator.retrieve import retrieve
from mcq_generator.split import split_items

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)
by_id = {i.id: i for i in corpus}
collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
    COLLECTION_NAME
)
client = OpenAI()

item = eval_set[0]
query = Query(
    item.image, item.grade, item.subject, item.topic, item.category, item.skill
)

# 1. Zero-shot
mcq_zero = generate(client, query, examples=[])
print("=== ZERO-SHOT ===")
print(mcq_zero.model_dump_json(indent=2))

# 2. Few-shot with retrieved examples (full RAG)
query_vector = embed_images([item])[0]
ids, level = retrieve(collection, query_vector, query)
examples = [by_id[i] for i in ids]

mcq_rag = generate(client, query, examples=examples)
print("=== FULL RAG (ladder level", level, ") ===")
print(mcq_rag.model_dump_json(indent=2))
