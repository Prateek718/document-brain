"""HTTP-level tests for the FastAPI application."""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from document_brain.ingestion import embed_texts
from document_brain.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Test client with an in-memory Qdrant and a mocked LLM."""
    qdrant = QdrantClient(":memory:")
    collection = "test_api"
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    chunks = [
        {
            "text": "The kilogram is the SI unit of mass.",
            "filename": "physics.pdf",
            "page": 1,
            "chunk_index": 0,
        }
    ]
    vectors = embed_texts([c["text"] for c in chunks])  # type: ignore[arg-type]
    qdrant.upsert(
        collection_name=collection,
        points=[PointStruct(id=0, vector=vectors[0], payload=chunks[0])],
        wait=True,
    )

    async def fake_generate(question: str, chunks: list) -> str:  # type: ignore[type-arg]
        return f"Mocked answer for: {question}"

    with (
        patch("document_brain.retrieval.get_client", return_value=qdrant),
        patch("document_brain.retrieval.settings.qdrant_collection", collection),
        patch("document_brain.main.ensure_collection_exists"),
        patch("document_brain.main.generate_answer", side_effect=fake_generate),
        TestClient(app, headers={"X-API-Key": "test-api-key-padded-to-thirty-two-chars"}) as c,
    ):
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is mass measured in?"})
    assert response.status_code == 200
    body = response.json()
    assert "Mocked answer" in body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["filename"] == "physics.pdf"


def test_query_rejects_short_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": "ok"})
    assert response.status_code == 422  # Pydantic validation


def test_query_validates_top_k_bounds(client: TestClient) -> None:
    response = client.post("/query", json={"question": "valid question", "top_k": 99})
    assert response.status_code == 422


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("test.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_trace_id_in_response_headers(client: TestClient) -> None:
    response = client.get("/health")
    assert "x-trace-id" in response.headers
    assert len(response.headers["x-trace-id"]) == 8


def test_query_returns_confidence_high_for_strong_match(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is the SI unit of mass?"})
    assert response.status_code == 200
    assert response.json()["confidence"] in ("high", "medium")


def test_query_returns_confidence_none_when_no_matches(client: TestClient) -> None:
    response = client.post(
        "/query", json={"question": "completely unrelated topic about quantum chromodynamics"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == "none"
    assert body["sources"] == []


def test_query_falls_back_when_llm_fails(client: TestClient) -> None:
    """When the LLM raises, the endpoint still returns 200 with a fallback answer."""
    from unittest.mock import patch

    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = httpx.ConnectError("simulated network failure", request=request)

    with patch("document_brain.main.generate_answer", side_effect=error):
        response = client.post("/query", json={"question": "What is the SI unit of mass?"})

    assert response.status_code == 200
    body = response.json()
    assert "unavailable" in body["answer"].lower()
    assert len(body["sources"]) >= 1


def test_query_rejects_missing_api_key() -> None:
    """Protected endpoint returns 401 when no API key is supplied."""
    with (
        patch("document_brain.main.ensure_collection_exists"),
        TestClient(app) as unauth_client,
    ):
        response = unauth_client.post("/query", json={"question": "valid question"})
    assert response.status_code == 401


def test_query_rejects_wrong_api_key() -> None:
    """Protected endpoint returns 401 when the API key is incorrect."""
    with (
        patch("document_brain.main.ensure_collection_exists"),
        TestClient(app, headers={"X-API-Key": "wrong-key"}) as unauth_client,
    ):
        response = unauth_client.post("/query", json={"question": "valid question"})
    assert response.status_code == 401


def test_documents_rejects_missing_api_key() -> None:
    """Ingestion endpoint is also protected: 401 without a key."""
    with (
        patch("document_brain.main.ensure_collection_exists"),
        TestClient(app) as unauth_client,
    ):
        response = unauth_client.post(
            "/documents",
            files={"file": ("test.pdf", b"%PDF-fake", "application/pdf")},
        )
    assert response.status_code == 401


def test_root_serves_frontend(client: TestClient) -> None:
    """The root path serves the HTML frontend, unauthenticated."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
