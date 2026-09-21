"""CLI demo: semantic search over hotel reviews.

Connects to the Qdrant collection built by ``bonnuit.ingest`` if it exists,
otherwise builds an in-memory one on the fly:

    python -m bonnuit.ingest              # optional: populate ./data/qdrant first
    python -m bonnuit.demo "a quiet hotel near the Louvre" -k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bonnuit.data import load_reviews
from bonnuit.search.retriever import Retriever
from bonnuit.search.vectorstore import DEFAULT_COLLECTION, Document

DEFAULT_QUERY = "An affordable hotel with a view of the Eiffel Tower"
DEFAULT_QDRANT_PATH = "data/qdrant"


def _load_retriever(
    qdrant_path: str, qdrant_url: str | None, collection: str, locality: str
) -> Retriever:
    if qdrant_url or Path(qdrant_path).exists():
        target = qdrant_url or qdrant_path
        print(f"connecting to Qdrant at {target} [{collection}]")
        return Retriever.connect(collection=collection, url=qdrant_url, path=qdrant_path)

    print(f"no Qdrant store at {qdrant_path}; building an in-memory one for locality={locality!r}")
    df = load_reviews(locality=locality)
    documents = [
        Document(
            id=str(review_id),
            text=row["review_text"],
            metadata={"hotel_name": row.get("hotel_name"), "locality": row.get("locality")},
        )
        for review_id, row in df.iterrows()
    ]
    return Retriever.from_documents(documents, collection=collection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--locality", default="Paris")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-path", default=DEFAULT_QDRANT_PATH)
    args = parser.parse_args()

    retriever = _load_retriever(args.qdrant_path, args.qdrant_url, args.collection, args.locality)
    hits = retriever.search(args.query, k=args.k)

    print(f"\nQuery: {args.query}\n")
    for rank, hit in enumerate(hits, start=1):
        hotel = hit.metadata.get("hotel_name", "?")
        print(f"{rank}. {hotel}  (similarity {hit.score:.3f})")
        print(f"   {hit.text}\n")


if __name__ == "__main__":
    main()
