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


def skill_filter(query: Query) -> dict:
    return {"skill": query.skill}


def topic_filter(query: Query) -> dict:
    return {"topic": query.topic}


def search(
    collection: chromadb.Collection, query_vector: np.ndarray, where: dict, k: int = 3
) -> list[str]:
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
    )
    return results["ids"][0]


def ladder(query: Query) -> list[dict]:
    return [
        grade_range_filter(query, window=0),
        grade_range_filter(query, window=1),
        grade_range_filter(query, window=2),
        skill_filter(query),
        topic_filter(query),
    ]


def retrieve(
    collection: chromadb.Collection, query_vector: np.ndarray, query: Query, k: int = 3
) -> tuple[list[str], int]:
    for level, where in enumerate(ladder(query), start=1):
        ids = search(collection, query_vector, where, k)
        if len(ids) == k:
            return ids, level
    return ids, level
