"""Query-time retrieval: embed a query and search the Qdrant collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bonnuit.search.embeddings import Embedder
from bonnuit.search.vectorstore import DEFAULT_COLLECTION, Document, QdrantStore


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class Retriever:
    def __init__(self, store: QdrantStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or Embedder()

    @classmethod
    def connect(
        cls,
        *,
        collection: str = DEFAULT_COLLECTION,
        url: str | None = None,
        path: str | None = "data/qdrant",
        embedder: Embedder | None = None,
    ) -> "Retriever":
        """Attach to an existing Qdrant collection built by :mod:`bonnuit.ingest`."""
        embedder = embedder or Embedder()
        store = QdrantStore(
            dimension=embedder.dimension,
            collection=collection,
            url=url,
            path=None if url else path,
        )
        return cls(store, embedder)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedder: Embedder | None = None,
        collection: str = DEFAULT_COLLECTION,
    ) -> "Retriever":
        """Build an ephemeral in-memory collection (used by the CLI demo and tests)."""
        embedder = embedder or Embedder()
        store = QdrantStore(dimension=embedder.dimension, collection=collection)
        store.add(embedder.encode_documents([doc.text for doc in documents]), documents)
        return cls(store, embedder)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query_embedding = self.embedder.encode_query(query)
        return [
            SearchHit(
                id=result.document.id,
                score=result.score,
                text=result.document.text,
                metadata=result.document.metadata,
            )
            for result in self.store.search(query_embedding, k)
        ]
