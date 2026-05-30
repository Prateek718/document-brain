"""Semantic retrieval over the Qdrant vector store.

Embeds a natural-language query using the same model used at ingestion time,
then searches Qdrant for the top-k most similar chunks. Returns typed results
with full citation metadata.
"""

import logging

from document_brain.config import settings
from document_brain.ingestion import embed_texts
from document_brain.schemas import RetrievalResult
from document_brain.vector_store import get_client

logger = logging.getLogger(__name__)


def search(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievalResult]:
    """Return the top-k chunks most semantically similar to the query.

    Args:
        query: Natural-language query string.
        top_k: Maximum number of chunks to return. Defaults to settings.default_top_k.
        score_threshold: Drop results below this cosine similarity. Defaults to
            settings.similarity_threshold. The threshold is the hallucination guard —
            an empty result list is honest "no relevant context."

    Returns:
        Ranked list (highest score first) of RetrievalResult objects.
        May be shorter than top_k if some results fall below the threshold.
        May be empty if no results clear the threshold.

    Raises:
        ValueError: If `query` is empty or whitespace-only.
    """
    if not query.strip():
        raise ValueError("query must be a non-empty string.")

    effective_top_k = top_k if top_k is not None else settings.default_top_k
    effective_threshold = (
        score_threshold if score_threshold is not None else settings.similarity_threshold
    )

    [query_vector] = embed_texts([query])
    client = get_client()

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=effective_top_k,
        score_threshold=effective_threshold,
        with_payload=True,
    )

    results = [
        RetrievalResult(
            text=str(point.payload["text"]),
            score=point.score,
            filename=str(point.payload["filename"]),
            page=int(point.payload["page"]),
            chunk_index=int(point.payload["chunk_index"]),
        )
        for point in response.points
        if point.payload is not None
    ]

    logger.info(
        "Query %r returned %d results above threshold %.2f.",
        query[:60],
        len(results),
        effective_threshold,
    )
    return results
