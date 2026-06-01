"""FastAPI application entry point."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import pypdf
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from document_brain.api_schemas import IngestResponse, QueryRequest, QueryResponse
from document_brain.auth import require_api_key
from document_brain.config import settings
from document_brain.generation import generate_answer
from document_brain.ingestion import ingest_document
from document_brain.retrieval import search
from document_brain.vector_store import ensure_collection_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _confidence_from_chunks(chunks: list) -> str:  # type: ignore[type-arg]
    """Heuristic: confidence reflects the top retrieved chunk's similarity."""
    if not chunks:
        return "none"
    top = chunks[0].score
    if top >= 0.6:
        return "high"
    if top >= 0.45:
        return "medium"
    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run once at startup; ensure Qdrant collection exists."""
    ensure_collection_exists()
    logger.info("Document Brain ready.")
    yield


app = FastAPI(
    title="Document Brain",
    description="Production RAG API for document Q&A with source citations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
    trace = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    logger.info("[%s] %s %s", trace, request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("[%s] -> %s (%.1fms)", trace, response.status_code, elapsed_ms)
    response.headers["X-Trace-Id"] = trace
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def upload_document(
    file: Annotated[UploadFile, File()],
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")

    try:
        chunks = ingest_document(file.filename, pdf_bytes)
    except pypdf.errors.PdfReadError as e:
        raise HTTPException(status_code=422, detail=f"Invalid PDF: {e}") from e

    return IngestResponse(filename=file.filename, chunks_ingested=chunks)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        chunks = await run_in_threadpool(search, request.question, top_k=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        answer = await generate_answer(request.question, chunks)
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
        logger.error("LLM call failed: %s; returning retrieval-only fallback.", e)
        if chunks:
            answer = (
                "The LLM service is currently unavailable. Here are the most relevant "
                "passages found in the documents:\n\n"
                + "\n\n---\n\n".join(
                    f"[Source {i + 1}, page {c.page}] {c.text}" for i, c in enumerate(chunks)
                )
            )
        else:
            answer = "The LLM service is unavailable and no relevant context was found."

    return QueryResponse(
        answer=answer,
        sources=chunks,
        confidence=_confidence_from_chunks(chunks),  # type: ignore[arg-type]
        model=settings.llm_model,
    )
