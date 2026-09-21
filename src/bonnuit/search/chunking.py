"""Split review text into overlapping character windows for embedding."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Break ``text`` into windows of ``chunk_size`` chars overlapping by ``overlap``.

    Whitespace is collapsed first. Text shorter than ``chunk_size`` is returned as a
    single chunk; empty text yields an empty list. The trailing window is skipped
    when its content is already fully contained in the previous chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        if start > 0 and len(text) - start <= overlap:
            break  # remainder already covered by the previous window
        chunks.append(text[start : start + chunk_size])
    return chunks
