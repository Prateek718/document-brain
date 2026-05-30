"""Request/response models for the HTTP API."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from document_brain.schemas import RetrievalResult

Confidence = Literal["high", "medium", "low", "none"]


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievalResult]
    confidence: Confidence
    model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
