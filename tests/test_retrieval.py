"""Tests for the retrieval layer.

Uses Qdrant's in-memory mode (`:memory:`) so tests exercise real similarity
math without touching the cloud cluster. The fixture seeds a tiny known
corpus before each test, so retrieval behavior is fully deterministic.
"""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from document_brain.ingestion import embed_texts
from document_brain.retrieval import search


@pytest.fixture
def seeded_qdrant() -> Generator[QdrantClient, None, None]:
    """In-memory Qdrant client preloaded with a known three-chunk corpus."""
    client = QdrantClient(":memory:")
    collection_name = "test_collection"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    chunks = [
        {
            "text": "The kilogram is the SI unit of mass in the International System of Units.",
            "filename": "physics.pdf",
            "page": 1,
            "chunk_index": 0,
        },
        {
            "text": (
                "Photosynthesis is the process by which plants convert "
                "sunlight into chemical energy."
            ),
            "filename": "biology.pdf",
            "page": 5,
            "chunk_index": 0,
        },
        {
            "text": (
                "The metre is defined as the length light travels in vacuum "
                "during a precise fraction of a second."
            ),
            "filename": "physics.pdf",
            "page": 2,
            "chunk_index": 1,
        },
    ]
    vectors = embed_texts([c["text"] for c in chunks])  # type: ignore[arg-type]

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=i, vector=vec, payload=chunk)
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ],
        wait=True,
    )

    with (
        patch("document_brain.retrieval.get_client", return_value=client),
        patch("document_brain.retrieval.settings.qdrant_collection", collection_name),
    ):
        yield client


def test_retrieves_semantically_similar_chunks(seeded_qdrant: QdrantClient) -> None:
    """A physics question should retrieve physics chunks ahead of biology."""
    results = search("What is the SI unit of mass?", score_threshold=0.0)
    assert len(results) >= 1
    assert results[0].filename == "physics.pdf"
    assert "kilogram" in results[0].text.lower()


def test_filters_out_irrelevant_chunks_via_threshold(seeded_qdrant: QdrantClient) -> None:
    """A query unrelated to any indexed content returns an empty list."""
    results = search("How do you bake a chocolate cake?", score_threshold=0.3)
    assert results == []


def test_top_k_caps_result_count(seeded_qdrant: QdrantClient) -> None:
    """top_k bounds the number of returned chunks."""
    results = search("physics", top_k=2, score_threshold=0.0)
    assert len(results) <= 2


def test_results_are_sorted_by_score_descending(seeded_qdrant: QdrantClient) -> None:
    """Most-similar result appears first."""
    results = search("physics units of measurement", score_threshold=0.0)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rejects_empty_query() -> None:
    """An empty or whitespace-only query raises ValueError before any I/O."""
    with pytest.raises(ValueError, match="query"):
        search("   ")
