"""BonNuit HTTP API (FastAPI).

Run from the repo root:

    PYTHONPATH=src uvicorn backend.app:app --reload

- Interactive docs:      http://localhost:8000/docs
- Machine-readable spec:  http://localhost:8000/openapi.json  (frontend generates
  its TypeScript types from this)

The route handlers currently return STUB data so the frontend has a stable
contract to build against. Swap the bodies for the real `bonnuit` retriever /
generator once the pipeline runs end to end -- the request/response models in
`backend.schemas` (the contract) stay the same.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    ReviewHit,
    SearchRequest,
    SearchResponse,
)

API_VERSION = "0.1.0"

# Comma-separated list of allowed browser origins. Defaults cover a Next.js dev
# server (3000) and a Vite dev server (5173).
CORS_ORIGINS = os.getenv(
    "BONNUIT_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")

app = FastAPI(
    title="BonNuit API",
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


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION)


@app.post("/api/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    """Return the reviews most similar to `query`.

    STUB: replace with `bonnuit.retriever.Retriever(...).search(req.query, k=req.k)`
    and map each `SearchHit` to a `ReviewHit`.
    """
    stub = [
        ReviewHit(
            hotel_name="Hôtel Exemple",
            review="Quiet room at the back, ten minutes on foot from the museum.",
            score=0.83,
            locality=req.locality or "Paris",
        ),
        ReviewHit(
            hotel_name="Hôtel Placeholder",
            review="Good value for the location, though the walls are thin.",
            score=0.71,
            locality=req.locality or "Paris",
        ),
    ]
    return SearchResponse(query=req.query, hits=stub[: req.k])


@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """Retrieve context, then return a generated answer grounded in it.

    STUB: retrieve top-k reviews, then call the (not yet written)
    `bonnuit.generator` which prompts Claude and returns an answer + citations.
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
