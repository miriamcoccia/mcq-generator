"""
try_retrieve.py: runs retrieval for one eval item and prints what comes back
"""
from pathlib import Path
from mcq_generator.retrieve import grade_range_filter, search

import chromadb
from collections import Counter 

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.domain import Query
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME, embed_images
from mcq_generator.retrieve import level1_filter, search
from mcq_generator.split import split_items

items = drop_image_choices(load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images")))
corpus, eval_set = split_items(items)
by_id = {i.id: i for i in corpus}
collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)

item = eval_set[0]
query = Query(item.image, item.grade, item.subject, item.topic, item.category, item.skill)
query_vector = embed_images([item])[0]        # same model as the index, or distances mean nothing

ids = search(collection, query_vector, grade_range_filter(query, window=1))
print("QUERY:", query.skill, "| grade", query.grade)
for i in ids:
    print("  ", by_id[i].skill, "| grade", by_id[i].grade, "|", by_id[i].image)

matches = collection.get(where=grade_range_filter(query, window=1))
print(Counter(m["grade"] for m in matches["metadatas"]))