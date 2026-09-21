"""Request/response models for the BonNuit API -- the frontend/backend contract.

FastAPI turns these into the OpenAPI schema at ``/openapi.json``; the frontend
generates its TypeScript types from that. Keep these separate from the internal
``bonnuit`` dataclasses so the wire format can stay stable while the
implementation changes. Any edit here affects both sides -- review accordingly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["a quiet hotel near the Louvre"])
    k: int = Field(5, ge=1, le=50, description="number of review hits to return")
    locality: str | None = Field(None, examples=["Paris"])


class ReviewHit(BaseModel):
    hotel_name: str
    review: str
    score: float = Field(..., description="cosine similarity, 0..1")
    locality: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[ReviewHit]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["family hotel near Sacré-Cœur, walkable"])
    k: int = Field(8, ge=1, le=50, description="reviews to retrieve as context")
    locality: str | None = Field(None, examples=["Paris"])


class Citation(BaseModel):
    hotel_name: str
    review: str


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: str
    version: str
