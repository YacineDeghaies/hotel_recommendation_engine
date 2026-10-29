import pytest
from fastapi.testclient import TestClient

from hotel_recommendation_engine.api.app import app, get_reranker, get_retriever
from hotel_recommendation_engine.search.reranker import Reranker
from hotel_recommendation_engine.search.retriever import SearchHit


class FakeRetriever:
    def __init__(self, results: list[SearchHit] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.results = results if results is not None else [
            SearchHit(
                id="review-1",
                score=0.91,
                text="Peaceful room within walking distance of the Louvre.",
                metadata={"hotel_name": "Hotel Test", "locality": "Paris"},
            )
        ]

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append((query, k))
        return self.results[:k]


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        self.pairs = pairs
        return self.scores


@pytest.fixture(autouse=True)
def fake_reranker() -> None:
    class IdentityReranker:
        def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
            return hits

    app.dependency_overrides[get_reranker] = lambda: IdentityReranker()
    yield
    app.dependency_overrides.clear()


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
    assert retriever.calls == [("quiet hotel near the Louvre", 15)]
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


def test_search_returns_each_hotel_only_once() -> None:
    retriever = FakeRetriever(
        [
            SearchHit("a-1", 0.95, "Best Hotel A review", {"hotel_name": "Hotel A"}),
            SearchHit("a-2", 0.90, "Another Hotel A review", {"hotel_name": " hotel a "}),
            SearchHit("b-1", 0.85, "Best Hotel B review", {"hotel_name": "Hotel B"}),
        ]
    )
    app.dependency_overrides[get_retriever] = lambda: retriever

    try:
        response = TestClient(app).post(
            "/api/search",
            json={"query": "quiet hotel", "k": 2},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert retriever.calls == [("quiet hotel", 10)]
    assert [hit["hotel_name"] for hit in response.json()["hits"]] == ["Hotel A", "Hotel B"]
    assert response.json()["hits"][0]["review"] == "Best Hotel A review"


def test_search_reranks_passages_before_grouping_hotels() -> None:
    retriever = FakeRetriever(
        [
            SearchHit("a-1", 0.95, "Weak Hotel A review", {"hotel_name": "Hotel A"}),
            SearchHit("a-2", 0.90, "Strong Hotel A review", {"hotel_name": " hotel a "}),
            SearchHit("b-1", 0.85, "Hotel B review", {"hotel_name": "Hotel B"}),
        ]
    )
    model = FakeCrossEncoder([0.1, 0.9, 0.7])
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_reranker] = lambda: Reranker(model=model)

    response = TestClient(app).post("/api/search", json={"query": "quiet hotel", "k": 2})

    assert response.status_code == 200
    assert model.pairs == [
        ("quiet hotel", "Weak Hotel A review"),
        ("quiet hotel", "Strong Hotel A review"),
        ("quiet hotel", "Hotel B review"),
    ]
    assert [(hit["hotel_name"].strip(), hit["review"], hit["score"])
            for hit in response.json()["hits"]] == [
        ("hotel a", "Strong Hotel A review", 0.90),
        ("Hotel B", "Hotel B review", 0.85),
    ]
