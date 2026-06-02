"""Integration test against a real Qdrant engine.

Unit tests use the in-memory Qdrant client, which is lenient about point-id
formats; the real engine requires unsigned ints or UUID strings. This test
exercises the real ingest -> retrieve path against an actual Qdrant so that
contract (the kind of mismatch the in-memory client silently tolerates) is
verified in CI.

Requires a Qdrant reachable at QDRANT_URL (localhost:6333 locally, a service
container in CI). Marked `integration` so the default unit-test run skips it.
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def real_qdrant_collection() -> str:
    """Point the app at a real local Qdrant and a throwaway collection.

    Clears the cached client so it reconnects with test settings, creates a
    fresh collection, yields its name, and drops it on teardown.
    """
    os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://localhost:6333")
    os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY", "")

    # Import after env is set so settings pick up the test values.
    from document_brain import config
    from document_brain.vector_store import ensure_collection_exists, get_client

    config.settings.qdrant_url = os.environ["QDRANT_URL"]
    config.settings.qdrant_api_key = os.environ["QDRANT_API_KEY"]
    config.settings.qdrant_collection = "integration_test"

    get_client.cache_clear()  # drop any client cached against other settings
    client = get_client()
    # Clean slate in case a prior failed run left the collection behind.
    if client.collection_exists("integration_test"):
        client.delete_collection("integration_test")
    ensure_collection_exists()

    yield "integration_test"

    client.delete_collection("integration_test")
    get_client.cache_clear()


def test_real_qdrant_ingest_and_retrieve(real_qdrant_collection: str) -> None:
    """Ingesting a chunk into real Qdrant and retrieving it round-trips.

    This fails against the real engine if point ids are not valid UUIDs/ints,
    which is exactly the bug the in-memory unit tests cannot catch.
    """
    from document_brain.ingestion import ingest_document
    from document_brain.retrieval import search

    # Minimal valid single-page PDF containing known text.
    pdf_bytes = _make_test_pdf("The kilogram is the SI unit of mass.")

    chunks_ingested = ingest_document("integration_test.pdf", pdf_bytes)
    assert chunks_ingested >= 1

    results = search("What is the SI unit of mass?", top_k=3)
    assert len(results) >= 1
    assert any("kilogram" in r.text.lower() for r in results)


def _make_test_pdf(text: str) -> bytes:
    """Build a one-page PDF containing the given text."""
    import io

    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 700, text)
    c.save()
    return buffer.getvalue()
