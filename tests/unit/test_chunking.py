import pytest

from hotel_search_engine.search.chunking import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("a lovely quiet room", chunk_size=512) == ["a lovely quiet room"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", chunk_size=512) == []


def test_whitespace_is_collapsed():
    assert chunk_text("a\n\n  b\tc", chunk_size=512) == ["a b c"]


def test_windows_cover_text_with_overlap():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap_size=100)
    assert [len(c) for c in chunks] == [400, 400, 400]  # starts 0, 300, 600


def test_trailing_window_fully_inside_previous_is_dropped():
    text = "y" * 620
    # starts at 0, 300; a window at 600 would only add 20 chars (< overlap) -> dropped
    chunks = chunk_text(text, chunk_size=400, overlap_size=100)
    assert len(chunks) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -5},
        {"chunk_size": 100, "overlap_size": 100},
        {"chunk_size": 100, "overlap_size": 150},
        {"chunk_size": 100, "overlap_size": -1},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        chunk_text("some text", **kwargs)
