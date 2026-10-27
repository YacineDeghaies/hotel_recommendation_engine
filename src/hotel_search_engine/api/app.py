"""Hotel Search Engine HTTP API (FastAPI).

Run from the repo root:

    uvicorn hotel_search_engine.api.app:app --reload

- Interactive docs:      http://localhost:8000/docs
- Machine-readable spec:  http://localhost:8000/openapi.json  (frontend generates
  its TypeScript types from this)

The search route uses the persisted Qdrant index built by
``hotel-search-engine-ingest``. The answer-generation route remains a stub.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hotel_search_engine.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    ReviewHit,
    SearchRequest,
    SearchResponse,
)
from hotel_search_engine.search.retriever import Retriever
from hotel_search_engine.search.vectorstore import DEFAULT_COLLECTION

API_VERSION = "0.1.0"

# Comma-separated list of allowed browser origins. Defaults cover a Next.js dev
# server (3000) and a Vite dev server (5173).
CORS_ORIGINS = os.getenv(
    "HOTEL_SEARCH_ENGINE_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")

app = FastAPI(
    title="Hotel Search Engine API",
    version=API_VERSION,
    summary="Semantic search and RAG over hotel reviews.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Connect once to the configured hotel-review index."""
    return Retriever.connect(
        collection=os.getenv("HOTEL_SEARCH_ENGINE_COLLECTION", DEFAULT_COLLECTION),
        url=os.getenv("HOTEL_SEARCH_ENGINE_QDRANT_URL") or None,
        path=os.getenv("HOTEL_SEARCH_ENGINE_QDRANT_PATH", "data/qdrant"),
    )


RetrieverDependency = Annotated[Retriever, Depends(get_retriever)]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION)


@app.post("/api/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, retriever: RetrieverDependency) -> SearchResponse:
    """Return the indexed reviews most similar to the query."""
    hits = [
        ReviewHit(
            hotel_name=str(hit.metadata.get("hotel_name") or "Unknown hotel"),
            review=hit.text,
            score=hit.score,
            locality=(
                str(hit.metadata["locality"])
                if hit.metadata.get("locality") is not None
                else None
            ),
        )
        for hit in retriever.search(req.query, k=req.k)
    ]
    return SearchResponse(query=req.query, hits=hits)


@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Retrieve context, then return a generated answer grounded in it.

    STUB: retrieve top-k reviews, then call the (not yet written)
    `hotel_search_engine.generator` which prompts Claude and returns an answer + citations.
    """
    return AskResponse(
        query=req.query,
        answer=(
            "(stubbed) A grounded, few-sentence answer will go here, naming "
            "specific hotels and quoting the reviews it relied on."
        ),
        citations=[
            Citation(
                hotel_name="Hôtel Exemple",
                review="Quiet room at the back, ten minutes on foot from the museum.",
            )
        ],
    )
