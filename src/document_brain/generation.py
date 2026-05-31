"""LLM answer generation over retrieved chunks, with citations."""

import logging

import httpx

from document_brain.config import settings
from document_brain.schemas import RetrievalResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise document assistant. Answer using ONLY the provided context.

Rules:
- Cite sources inline as [Source N] where N matches the source number below.
- If the context does not contain the answer, say so plainly. Do not invent facts.
- Be concise. Use bullet points for multi-part answers.
"""


def _build_user_message(question: str, chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return (
            "No relevant context was found in the document database. "
            "Tell the user no information is available on this topic.\n\n"
            f"Question: {question}"
        )
    blocks = [
        f"[Source {i}] (file: {c.filename}, page: {c.page})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return f"CONTEXT:\n{'\n\n---\n\n'.join(blocks)}\n\nQUESTION: {question}"


async def generate_answer(question: str, chunks: list[RetrievalResult]) -> str:
    """Call Claude with the question and retrieved chunks; return the answer text."""
    user_msg = _build_user_message(question, chunks)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.anthropic_base_url}/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": settings.max_tokens,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            },
        )
        response.raise_for_status()
        data = response.json()

    answer = "".join(b["text"] for b in data["content"] if b["type"] == "text").strip()
    logger.info("Generated answer for query %r (chunks=%d).", question[:60], len(chunks))
    return answer
