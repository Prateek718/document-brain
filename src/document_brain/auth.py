"""API-key authentication for protected endpoints.

A shared-secret scheme: clients send the key in the X-API-Key header, which is
compared against the configured key in constant time. Applied as a route
dependency so protected endpoints reject unauthenticated requests before any
work (embedding, retrieval, LLM calls) is done.
"""

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from document_brain.config import settings


async def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Reject requests without a valid X-API-Key header.

    Uses secrets.compare_digest for a constant-time comparison, so the check
    does not leak information about the key via response timing.
    """
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
