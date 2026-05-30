"""Tests for LLM answer generation."""

import httpx
import pytest
from pytest_httpx import HTTPXMock

from document_brain.generation import _build_user_message, generate_answer
from document_brain.schemas import RetrievalResult


def _chunk(text: str, page: int = 1) -> RetrievalResult:
    return RetrievalResult(text=text, score=0.8, filename="doc.pdf", page=page, chunk_index=0)


def test_build_user_message_labels_sources() -> None:
    msg = _build_user_message("Q?", [_chunk("first"), _chunk("second", page=2)])
    assert "[Source 1]" in msg
    assert "[Source 2]" in msg
    assert "page: 1" in msg
    assert "page: 2" in msg


def test_build_user_message_handles_empty_chunks() -> None:
    msg = _build_user_message("Q?", [])
    assert "No relevant context" in msg
    assert "Q?" in msg


@pytest.mark.asyncio
async def test_generate_answer_returns_text(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json={"content": [{"type": "text", "text": "The answer is 42."}]},
    )
    result = await generate_answer("Q?", [_chunk("ctx")])
    assert result == "The answer is 42."


@pytest.mark.asyncio
async def test_generate_answer_raises_on_http_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        status_code=401,
        json={"error": "invalid api key"},
    )
    with pytest.raises(httpx.HTTPStatusError):
        await generate_answer("Q?", [_chunk("ctx")])
