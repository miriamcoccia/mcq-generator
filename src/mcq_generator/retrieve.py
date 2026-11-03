"""
retrieve.py: picks few-shot examples for a Query (metadata filter + image ranking)
"""

import chromadb
import numpy as np
import random

from mcq_generator.domain import Query
from enum import Enum


class Condition(str, Enum):
    ZERO = "zero"
    RANDOM = "random"
    METADATA = "metadata"  # pool of examples is chosen using metadata only, no image
    RAG = "rag"  # pool of examples is chosen also using the image similarity


def select_examples(
    condition: Condition,
    collection: chromadb.Collection,
    corpus_ids: list[str],
    query: Query,
    item_id: str,
    query_vector: np.ndarray,
    k: int = 3,
) -> tuple[list[str], int | None]:
    if condition == Condition.ZERO:
        return [], None
    if condition == Condition.RANDOM:
        return random_examples(corpus_ids, item_id, k), None
    if condition == Condition.METADATA:
        return metadata_examples(collection, query, item_id, k)
    if condition == Condition.RAG:
        return retrieve(collection, query_vector, k)
    raise ValueError(f"Unknown condition: {condition}")


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


# complete RAG using the filtering ladder
def retrieve(
    collection: chromadb.Collection, query_vector: np.ndarray, query: Query, k: int = 3
) -> tuple[list[str], int]:
    for level, where in enumerate(ladder(query), start=1):
        ids = search(collection, query_vector, where, k)
        if len(ids) == k:
            return ids, level
    return ids, level


# testing whether adding any example would help, regardless of their similarity
def random_examples(
    corpus_ids: list[str], item_id: str, k: int = 3, seed: int = 42
) -> list[str]:
    rng = random.Random(f"{seed}-{item_id}")
    return rng.sample(corpus_ids, k)


# testing whether similarity by picture is needed or not. Does adding photos help?
def metadata_examples(
    collection: chromadb.Collection,
    query: Query,
    item_id: str,
    k: int = 3,
    seed: int = 42,
) -> tuple[list[str], int]:
    rng = random.Random(f"{seed}-{item_id}")
    for level, where in enumerate(ladder(query), start=1):
        pool = sorted(collection.get(where=where, include=[])["ids"])
        if len(pool) >= k:
            return rng.sample(pool, k), level
    return pool, level
