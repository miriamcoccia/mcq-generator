"""
check_neighbours.py: eyeball check that CLIP neighbours look related
"""
from pathlib import Path

import chromadb

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.index import CHROMA_PATH, COLLECTION_NAME
from mcq_generator.split import split_items

items = drop_image_choices(load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images")))
corpus, _ = split_items(items)
by_id = {i.id: i for i in corpus}

collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)

query_ids = [corpus[0].id, corpus[500].id, corpus[1000].id]
stored = collection.get(ids=query_ids, include=["embeddings"])

results = collection.query(
    query_embeddings=stored["embeddings"],
    n_results=4,
)

for q_id, ids, dists in zip(stored["ids"], results["ids"], results["distances"]):
    print("QUERY:", by_id[q_id].image, "|", by_id[q_id].skill)
    for n_id, d in zip(ids[1:], dists[1:]):   # [1:] skips the query itself
        print(f"  {d:.3f}", by_id[n_id].image, "|", by_id[n_id].skill)