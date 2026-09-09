"""Qdrant-backed vector store.

One backend, three ways to point it at storage (checked in this order):

* ``client`` — an already-configured :class:`~qdrant_client.QdrantClient`
* ``url``    — a Qdrant server, e.g. ``http://localhost:6333``
* ``path``   — embedded Qdrant persisting to a local directory
* nothing    — embedded, in-memory, ephemeral (used by the CLI demo and tests)

Vectors are stored with cosine distance. Point ids are derived deterministically
from the document id, so re-ingesting the same corpus updates rather than
duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

DEFAULT_COLLECTION = "hotel_reviews"


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class QdrantStore:
    def __init__(
        self,
        dimension: int,
        collection: str = DEFAULT_COLLECTION,
        *,
        client: QdrantClient | None = None,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        recreate: bool = False,
    ) -> None:
        if client is not None:
            self.client = client
        elif url is not None:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path is not None:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(location=":memory:")

        self.collection = collection
        self.dimension = dimension
        self._ensure_collection(recreate=recreate)

    def _ensure_collection(self, *, recreate: bool) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
            )

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[Document],
        batch_size: int = 256,
    ) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"expected embeddings of shape (n, {self.dimension}), got {embeddings.shape}"
            )
        if len(documents) != embeddings.shape[0]:
            raise ValueError("number of documents must match number of embeddings")

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, doc.id)),
                vector=vector.tolist(),
                payload={"doc_id": doc.id, "text": doc.text, **doc.metadata},
            )
            for vector, doc in zip(embeddings, documents)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
            )

    def search(self, embedding: np.ndarray, k: int) -> list[ScoredDocument]:
        vector = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
        )
        results: list[ScoredDocument] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            doc_id = payload.pop("doc_id", str(point.id))
            results.append(
                ScoredDocument(
                    document=Document(id=doc_id, text=text, metadata=payload),
                    score=float(point.score),
                )
            )
        return results

    def __len__(self) -> int:
        return self.client.count(collection_name=self.collection).count

    def close(self) -> None:
        self.client.close()
