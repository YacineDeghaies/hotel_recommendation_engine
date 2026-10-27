from fastapi.testclient import TestClient

from hotel_search_engine.api.app import app, get_retriever
from hotel_search_engine.search.retriever import SearchHit


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append((query, k))
        return [
            SearchHit(
                id="review-1",
                score=0.91,
                text="Peaceful room within walking distance of the Louvre.",
                metadata={"hotel_name": "Hotel Test", "locality": "Paris"},
            )
        ]


def test_search_returns_real_retriever_results() -> None:
    retriever = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: retriever

    try:
        response = TestClient(app).post(
            "/api/search",
            json={"query": "quiet hotel near the Louvre", "k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert retriever.calls == [("quiet hotel near the Louvre", 3)]
    assert response.json() == {
        "query": "quiet hotel near the Louvre",
        "hits": [
            {
                "hotel_name": "Hotel Test",
                "review": "Peaceful room within walking distance of the Louvre.",
                "score": 0.91,
                "locality": "Paris",
            }
        ],
    }


def test_search_rejects_invalid_k_before_calling_retriever() -> None:
    retriever = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: retriever

    try:
        response = TestClient(app).post(
            "/api/search",
            json={"query": "quiet hotel", "k": 0},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert retriever.calls == []
