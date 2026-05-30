"""Typed data models used across the application.

We define schemas in a dedicated module so they can be imported from anywhere
without creating circular dependencies (ingestion uses them, retrieval uses
them, the FastAPI layer will too).
"""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """One chunk returned from a similarity search, with citation metadata."""

    text: str = Field(description="The chunk's text content.")
    score: float = Field(description="Cosine similarity to the query, in [-1, 1].")
    filename: str = Field(description="Source document filename.")
    page: int = Field(description="1-indexed page number within the source document.")
    chunk_index: int = Field(description="Position of this chunk in the document.")
