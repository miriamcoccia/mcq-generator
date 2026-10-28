"""
build_index.py: embeds the retrieval corpus and stores it in ChromaDB. Rerun when data or model changes.
"""
from pathlib import Path

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.index import build_collection, embed_images
from mcq_generator.split import split_items

items = drop_image_choices(load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images")))
corpus, _ = split_items(items)          # index the corpus ONLY: eval items must never be retrievable

vectors = embed_images(corpus)
collection = build_collection(corpus, vectors)
print(collection.count(), "records in", collection.name)