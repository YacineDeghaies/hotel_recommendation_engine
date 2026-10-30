"""Split review text into overlapping character windows for embedding."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap_size: int = 64,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap_size < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap_size
    chunks: list[str] = []
    for start in range(0, len(text), step):
        if start > 0 and len(text) - start <= overlap_size:
            break  # remaining text already included the previous chunk
        chunks.append(text[start : start + chunk_size])
    return chunks
