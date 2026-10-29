"""Ingestion pipeline: load reviews -> clean -> chunk -> embed -> upsert to Qdrant.

Run once, offline, before serving queries:

    # embedded Qdrant, persisted to ./data/qdrant
    python -m hotel_recommendation_engine.ingest --locality Paris

    # against a running Qdrant server
    python -m hotel_recommendation_engine.ingest --locality Paris --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse

from hotel_recommendation_engine.data import load_reviews
from hotel_recommendation_engine.search.chunking import chunk_text
from hotel_recommendation_engine.search.embeddings import Embedder
from hotel_recommendation_engine.search.vectorstore import DEFAULT_COLLECTION, Document, QdrantStore

DEFAULT_QDRANT_PATH = "data/qdrant"


def build_index(
    locality: str | None = "Paris",
    *,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str | None = None,
    qdrant_path: str | None = DEFAULT_QDRANT_PATH,
    chunk_size: int = 512,
    overlap: int = 64,
    recreate: bool = True,
    embedder: Embedder | None = None,
) -> QdrantStore:
    df = load_reviews(locality=locality)
    embedder = embedder or Embedder()

    documents: list[Document] = []
    for review_id, row in df.iterrows():
        for chunk_index, chunk in enumerate(
            chunk_text(row["review_text"], chunk_size=chunk_size, overlap_size=overlap)
        ):
            documents.append(
                Document(
                    id=f"{review_id}-{chunk_index}",
                    text=chunk,
                    metadata={
                        "review_id": int(review_id),
                        "chunk_index": chunk_index,
                        "hotel_name": row.get("hotel_name"),
                        "locality": row.get("locality"),
                        "hotel_description": row.get("hotel_description"),
                    },
                )
            )
    if not documents:
        raise ValueError(f"no reviews found for locality={locality!r}")

    embeddings = embedder.encode_documents([doc.text for doc in documents])
    store = QdrantStore(
        dimension=embedder.dimension,
        collection=collection,
        url=qdrant_url,
        path=None if qdrant_url else qdrant_path,
        recreate=recreate,
    )
    store.add(embeddings, documents)
    target = qdrant_url or qdrant_path or ":memory:"
    print(f"indexed {len(store)} chunks from {len(df)} reviews -> {target} [{collection}]")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locality", default="Paris")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--qdrant-url", default=None, help="Qdrant server URL; overrides --qdrant-path"
    )
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help="local embedded-Qdrant directory",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--no-recreate",
        action="store_false",
        dest="recreate",
        help="add to an existing collection instead of dropping it first",
    )
    args = parser.parse_args()
    build_index(
        locality=args.locality,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
