"""
try_retrieve.py: runs the fallback ladder for every eval item and counts which level was used
"""

from collections import Counter
from pathlib import Path

import chromadb

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import Query
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME, embed_images
from mcq_generator.retrieve import retrieve
from mcq_generator.split import split_items

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
corpus, eval_set = split_items(items)
collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
    COLLECTION_NAME
)

vectors = embed_images(eval_set)  # all eval images in one batch

levels = Counter()
short = []  # items that got fewer than 3 examples
for item, vec in zip(eval_set, vectors):
    query = Query(
        item.image, item.grade, item.subject, item.topic, item.category, item.skill
    )
    ids, level = retrieve(collection, vec, query)
    levels[level] += 1
    if len(ids) < 3:
        short.append(item.id)

print("Level used:", sorted(levels.items()))
print("Fewer than 3 examples:", len(short))
