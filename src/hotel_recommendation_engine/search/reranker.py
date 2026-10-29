"""Rerank retrieved review passages with a query-passage cross-encoder."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sentence_transformers import CrossEncoder

from hotel_recommendation_engine.search.retriever import SearchHit

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, model: Any = None) -> None:
        self.model = model if model is not None else CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.model.predict([(query, hit.text) for hit in hits], batch_size=32)
        if len(scores) != len(hits):
            raise ValueError("reranker returned a different number of scores than review hits")
        ranked = [replace(hit, rerank_score=float(score)) for hit, score in zip(hits, scores)]
        return sorted(ranked, key=lambda hit: hit.rerank_score, reverse=True)
