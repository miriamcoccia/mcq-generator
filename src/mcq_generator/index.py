"""
index.py: generates the corpus where the images are embedded
"""
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from pathlib import Path


from mcq_generator.domain import ScienceQAItem

# we can use another model here if we want better performance, but this is a baseline, so I'll use the most famous one
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

# we embed once and we run them from disk 
def embed_images(items: list[ScienceQAItem], model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    images = [Image.open(i.image).convert("RGB") for i in items]
    return model.encode(
        images, 
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

def save_index(vectors: np.ndarray, items: list[ScienceQAItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, ids=np.array([i.id for i in items]))

def load_index(path: Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(path)
    return data["vectors"], data["ids"].tolist()