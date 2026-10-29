"""Load and clean the hotel-review dataset."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

DATASET = "traversaal-ai-hackathon/hotel_datasets"


def load_reviews(locality: str | None = "Paris") -> pd.DataFrame:
    """Return a cleaned DataFrame of hotel reviews, optionally filtered by ``locality``.

    Rows with a missing or blank ``review_text`` are dropped. The index is reset so
    positional lookups line up with embedding row numbers.
    """
    df = load_dataset(DATASET)["train"].to_pandas()
    df = df[df["review_text"].notna() & (df["review_text"].str.strip() != "")]
    if locality is not None:
        df = df[df["locality"] == locality]
    return df.reset_index(drop=True)
