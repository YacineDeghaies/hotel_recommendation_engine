"""Embedding model wrapper shared by ingestion and query time.

Vectors are L2-normalized so that a plain inner product equals cosine similarity,
which is what both vector-store backends assume.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from hotel_recommendation_engine.search.similarity import l2_normalize

DEFAULT_MODEL = "all-MiniLM-L6-v2"

__all__ = ["Embedder", "l2_normalize", "DEFAULT_MODEL"]


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("the embedding model did not report its output dimension")
        return dimension

    def encode_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return l2_normalize(embeddings)

    def encode_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return l2_normalize(embedding)[0]
