"""
index.py: embeds corpus images and stores them, with their metadata, in ChromaDB
"""
from pathlib import Path

import chromadb
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from mcq_generator.domain import ScienceQAItem

# Baseline model; swap here (e.g. SigLIP 2) to compare
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "clip-vit-b-32-corpus"


def embed_images(items: list[ScienceQAItem], model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    images = [Image.open(i.image).convert("RGB") for i in items]
    return model.encode(
        images,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def item_metadata(item: ScienceQAItem) -> dict[str, str | int]:
    return {
        "grade": item.grade,
        "subject": item.subject,
        "topic": item.topic,
        "category": item.category,
        "skill": item.skill,
    }


def build_collection(
    items: list[ScienceQAItem],
    vectors: np.ndarray,
    path: Path = CHROMA_PATH,
    name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(name)
    collection.upsert(
        ids=[i.id for i in items],
        embeddings=vectors,
        metadatas=[item_metadata(i) for i in items],
    )
    return collection