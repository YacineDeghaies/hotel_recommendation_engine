"""From-scratch cosine similarity helpers — just NumPy, no ML framework.

``l2_normalize`` is used by :mod:`hotel_search_engine.search.embeddings` to unit-length vectors
before they go to Qdrant. ``top_k_cosine`` is a self-contained reference
implementation of nearest-neighbour search: the live retrieval path uses Qdrant,
but this shows the underlying math and is exercised directly by the tests.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row of a 2-D array to unit length; all-zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, where=norms != 0, out=np.zeros_like(matrix))


def top_k_cosine(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, scores)`` of the ``k`` corpus rows most similar to the query.

    Parameters
    ----------
    query_embedding:
        1-D array of shape ``(d,)``.
    corpus_embeddings:
        2-D array of shape ``(n, d)``.
    k:
        Number of results to return, an integer in ``[1, n]``.

    The returned indices are ordered from most to least similar. Rows whose norm
    is zero score ``0.0`` rather than producing ``nan``.
    """
    query = np.asarray(query_embedding, dtype=np.float32)
    corpus = np.asarray(corpus_embeddings, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError(f"query_embedding must be 1-D, got shape {query.shape}")
    if corpus.ndim != 2:
        raise ValueError(f"corpus_embeddings must be 2-D, got shape {corpus.shape}")
    if corpus.shape[1] != query.size:
        raise ValueError(
            f"dimension mismatch: query has {query.size} dims, "
            f"corpus rows have {corpus.shape[1]}"
        )
    if (
        isinstance(k, (bool, np.bool_))  # guard against k=True / k=False
        or not isinstance(k, (int, np.integer))  # guard against k="5" / k=5.4
        or k < 1
        or k > corpus.shape[0]
    ):
        raise ValueError(
            f"k must be an int between 1 and {corpus.shape[0]}, got {k!r}"
        )

    dot_products = corpus @ query
    denominator = np.linalg.norm(query) * np.linalg.norm(corpus, axis=1)
    scores = np.divide(
        dot_products,
        denominator,
        where=denominator != 0,
        out=np.zeros_like(dot_products, dtype=np.float32),
    )

    # Partition to get the top-k unordered (O(n)), then sort only those k.
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top, scores[top]
