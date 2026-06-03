"""Qdrant vector store: connection, collection management, upserts, and search.

Encapsulates all I/O with Qdrant. The rest of the application talks to this
module rather than to qdrant-client directly, which keeps the storage backend
swappable and the business logic free of network concerns.
"""

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from document_brain.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a cached Qdrant client. First call constructs the connection."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )


def ensure_collection_exists() -> None:
    """Create the configured collection if it doesn't already exist.

    Idempotent: safe to call on every application startup. Uses cosine distance
    because the embedding model produces normalized vectors and our retrieval
    math expects angle-based similarity.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.qdrant_collection in existing:
        logger.info("Collection %r already exists.", settings.qdrant_collection)
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=settings.embedding_dimensions,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection %r.", settings.qdrant_collection)


def upsert_chunks(
    points: list[PointStruct],
    batch_size: int = 128,
) -> None:
    """Insert or update points (chunk vectors + payloads), batched.

    Idempotent at the point-id level: re-upserting a point with the same id
    overwrites the previous vector and payload.

    Points are upserted in batches because a single request carrying thousands
    of vectors can exceed the server's request limits — Qdrant drops the
    connection. Batching keeps each request small enough to succeed and makes
    large-document ingestion reliable.
    """
    client = get_client()
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
            wait=True,
        )
    logger.info("Upserted %d points to %r.", len(points), settings.qdrant_collection)
