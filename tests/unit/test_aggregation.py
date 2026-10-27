import pytest

from hotel_search_engine.search.aggregation import aggregate_by_hotel
from hotel_search_engine.search.retriever import SearchHit


def _hit(id_: str, hotel: str | None, score: float) -> SearchHit:
    metadata = {"hotel_name": hotel} if hotel is not None else {}
    return SearchHit(id=id_, score=score, text=f"review {id_}", metadata=metadata)


def test_keeps_highest_scoring_hit_for_each_hotel() -> None:
    hits = [
        _hit("a-low", " Hotel A ", 0.70),
        _hit("b", "Hotel B", 0.80),
        _hit("a-high", "hotel a", 0.90),
    ]

    results = aggregate_by_hotel(hits, limit=5)

    assert [hit.id for hit in results] == ["a-high", "b"]


def test_limits_number_of_unique_hotels() -> None:
    hits = [_hit("a", "Hotel A", 0.90), _hit("b", "Hotel B", 0.80)]

    assert [hit.id for hit in aggregate_by_hotel(hits, limit=1)] == ["a"]


def test_skips_hits_without_a_hotel_name() -> None:
    assert aggregate_by_hotel([_hit("unknown", None, 0.99)], limit=5) == []


def test_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        aggregate_by_hotel([], limit=0)
