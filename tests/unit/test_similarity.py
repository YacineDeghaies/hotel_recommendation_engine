import numpy as np
import pytest

from hotel_search_engine.search.similarity import top_k_cosine


def test_ranks_by_direction_not_magnitude():
    query = np.array([1.0, 0.0])
    corpus = np.array(
        [
            [10.0, 0.0],  # same direction, large magnitude -> most similar
            [1.0, 1.0],  # 45 degrees
            [0.0, 5.0],  # orthogonal -> least similar
        ]
    )
    indices, scores = top_k_cosine(query, corpus, k=3)
    assert indices.tolist() == [0, 1, 2]
    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)


def test_k_one_returns_single_best():
    query = np.array([0.0, 1.0])
    corpus = np.array([[1.0, 0.0], [0.1, 1.0]])
    indices, _ = top_k_cosine(query, corpus, k=1)
    assert indices.tolist() == [1]


def test_zero_vector_row_scores_zero_not_nan():
    query = np.array([1.0, 2.0])
    corpus = np.array([[0.0, 0.0], [1.0, 2.0]])
    _, scores = top_k_cosine(query, corpus, k=2)
    assert not np.isnan(scores).any()
    assert scores.tolist() == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize("bad_k", [0, -1, 99, True, 1.5, "2"])
def test_invalid_k_raises(bad_k):
    query = np.array([1.0, 0.0])
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        top_k_cosine(query, corpus, k=bad_k)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        top_k_cosine(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0]]), k=1)


def test_query_must_be_1d():
    with pytest.raises(ValueError):
        top_k_cosine(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), k=1)
