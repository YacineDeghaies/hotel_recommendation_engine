"""Convert review-level retrieval hits into unique hotel results."""

from __future__ import annotations

from hotel_search_engine.search.retriever import SearchHit


def aggregate_by_hotel(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """Return the strongest review hit for each hotel, ordered by score."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    best_by_hotel: dict[str, SearchHit] = {}
    for hit in hits:
        hotel_name = hit.metadata.get("hotel_name")
        if not hotel_name:
            continue

        hotel_key = " ".join(str(hotel_name).split()).casefold()
        current = best_by_hotel.get(hotel_key)
        if current is None or hit.score > current.score:
            best_by_hotel[hotel_key] = hit

    return sorted(best_by_hotel.values(), key=lambda hit: hit.score, reverse=True)[:limit]
