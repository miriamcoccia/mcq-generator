"""
retrieve.py: picks few-shot examples for a Query (metadata filter + image ranking)
"""
import chromadb
import numpy as np

from mcq_generator.domain import Query


def level1_filter(query: Query) -> dict:
    return {"$and": [{"skill": query.skill}, {"grade": query.grade}]}


def grade_range_filter(query: Query, window: int) -> dict:
    return {
        "$and": [
            {"skill": query.skill},
            {"grade": {"$gte": query.grade - window}},
            {"grade": {"$lte": query.grade + window}},
        ]
    }
def search(collection: chromadb.Collection, query_vector: np.ndarray, where: dict, k: int = 3) -> list[str]:
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
    )
    return results["ids"][0]