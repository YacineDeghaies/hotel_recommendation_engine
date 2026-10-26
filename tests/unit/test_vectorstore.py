import numpy as np
import pytest

from hotel_search_engine.search.similarity import l2_normalize
from hotel_search_engine.search.vectorstore import Document, QdrantStore


def _corpus():
    embeddings = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    documents = [
        Document(id="a", text="near the tower", metadata={"hotel": "A"}),
        Document(id="b", text="tower view room", metadata={"hotel": "B"}),
        Document(id="c", text="airport shuttle", metadata={"hotel": "C"}),
    ]
    return embeddings, documents


def _populated_store(**kwargs):
    embeddings, documents = _corpus()
    store = QdrantStore(dimension=3, **kwargs)
    store.add(embeddings, documents)
    return store


def test_search_ranks_by_cosine():
    store = _populated_store()
    query = l2_normalize(np.array([[1.0, 0.05, 0.0]]))[0]
    hits = store.search(query, k=3)

    assert [h.document.id for h in hits] == ["a", "b", "c"]
    assert hits[0].score > 0.99  # query is almost collinear with doc "a"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_metadata_round_trips_through_search():
    store = _populated_store()
    hit = store.search(l2_normalize(np.array([[0.0, 0.0, 1.0]]))[0], k=1)[0]
    assert hit.document.id == "c"
    assert hit.document.metadata == {"hotel": "C"}


def test_k_larger_than_collection_is_clamped():
    store = _populated_store()
    assert len(store.search(np.array([1.0, 0.0, 0.0]), k=99)) == 3


def test_len_reports_point_count():
    assert len(_populated_store()) == 3


def test_reingesting_same_ids_updates_not_duplicates():
    store = _populated_store()
    embeddings, documents = _corpus()
    store.add(embeddings, documents)
    assert len(store) == 3


def test_dimension_mismatch_raises():
    store = QdrantStore(dimension=3)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 4), dtype=np.float32), [Document("x", "x"), Document("y", "y")])


def test_doc_count_mismatch_raises():
    store = QdrantStore(dimension=3)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 3), dtype=np.float32), [Document("x", "x")])


def test_persists_to_disk_and_reopens(tmp_path):
    path = str(tmp_path / "qdrant")
    store = _populated_store(path=path)
    before = store.search(l2_normalize(np.array([[0.0, 0.0, 1.0]]))[0], k=2)
    store.close()

    reopened = QdrantStore(dimension=3, path=path)
    assert len(reopened) == 3
    after = reopened.search(l2_normalize(np.array([[0.0, 0.0, 1.0]]))[0], k=2)
    # The second result is a zero-score tie, whose order Qdrant does not define.
    assert before[0].document.id == after[0].document.id == "c"
    assert before[0].score == pytest.approx(after[0].score)
    reopened.close()
