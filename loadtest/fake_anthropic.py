"""A fake Anthropic /v1/messages endpoint for load testing.

Stands in for the real API so load tests exercise our full HTTP path
(routing, retrieval, httpx call, response parsing) without spending tokens.
It sleeps to imitate real Haiku latency — without that delay the test would
not reproduce the 'many requests parked on a slow upstream' condition that
is the whole point of load testing an async service.
"""

import asyncio
import os
import random

from fastapi import FastAPI

app = FastAPI()

# Mimic real Haiku latency. Override via env to test other scenarios.
_MIN_DELAY = float(os.getenv("FAKE_LLM_MIN_DELAY", "0.8"))
_MAX_DELAY = float(os.getenv("FAKE_LLM_MAX_DELAY", "1.5"))

_FAKE_ANSWER = (
    "Based on the provided context, the document discusses the key topic "
    "in detail. [Source 1] The relevant section confirms this. [Source 2]"
)


@app.post("/v1/messages")
async def messages() -> dict:
    # Jittered sleep imitates network + inference time of a real call.
    await asyncio.sleep(random.uniform(_MIN_DELAY, _MAX_DELAY))
    return {
        "id": "msg_fake_loadtest",
        "type": "message",
        "role": "assistant",
        "model": "fake-haiku",
        "content": [{"type": "text", "text": _FAKE_ANSWER}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1500, "output_tokens": 60},
    }
