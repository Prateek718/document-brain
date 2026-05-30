"""PDF ingestion: extract text from PDF documents page by page."""

import hashlib
import io
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Final, cast

import pypdf
from qdrant_client.models import PointStruct

from document_brain.vector_store import ensure_collection_exists, upsert_chunks

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict[str, str | int]]:
    """Extract text from a PDF, preserving page numbers for citations.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        List of dicts, one per non-empty page, each with keys:
            - "page": int, 1-indexed page number
            - "text": str, extracted text content, stripped of leading/trailing whitespace

        Pages with no extractable text are skipped (common for scanned PDFs
        or image-only pages without OCR).

    Raises:
        pypdf.errors.PdfReadError: If the bytes are not a valid PDF.
    """
    pdf_stream = io.BytesIO(pdf_bytes)
    reader = pypdf.PdfReader(pdf_stream)

    pages: list[dict[str, str | int]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": page_number, "text": text})

    return pages


# Default chunking parameters tuned for sentence-transformers/all-MiniLM-L6-v2,
# whose 256-token context comfortably fits ~500 characters of English prose.
DEFAULT_CHUNK_SIZE: Final = 500
DEFAULT_CHUNK_OVERLAP: Final = 50

# Separators tried in priority order. Earlier = more natural boundary.
_SEPARATORS: Final = ("\n\n", "\n", ". ", " ")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks at the most natural available boundary.

    The splitter tries separators in priority order (paragraph, line, sentence,
    word, hard cut) and recurses on the remainder so chunks longer than
    `chunk_size` are guaranteed not to occur in the output.

    Args:
        text: The text to split. Whitespace-only input returns an empty list.
        chunk_size: Target maximum chunk length in characters. Must be > overlap.
        chunk_overlap: Number of characters each chunk shares with the next.
            Provides context continuity for sentences spanning a boundary.

    Returns:
        List of chunk strings, each stripped of leading/trailing whitespace and
        non-empty. Chunks are ordered as they appear in the original text.

    Raises:
        ValueError: If chunk_overlap >= chunk_size (would cause infinite recursion).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Find the rightmost separator that lands inside the chunk window.
    # We search the slice [chunk_overlap : chunk_size] so the first chunk is
    # never absurdly short — at minimum it contains chunk_overlap characters.
    for sep in _SEPARATORS:
        split_at = text.rfind(sep, chunk_overlap, chunk_size)
        if split_at != -1:
            split_at += len(sep)  # include the separator in the previous chunk
            head = text[:split_at].strip()
            # Back up by chunk_overlap so the next chunk re-includes the tail of this one.
            tail_start = max(0, split_at - chunk_overlap)
            tail = text[tail_start:]
            return [head, *chunk_text(tail, chunk_size, chunk_overlap)]

    # No natural separator found — hard cut at chunk_size.
    head = text[:chunk_size].strip()
    tail = text[chunk_size - chunk_overlap :]
    return [head, *chunk_text(tail, chunk_size, chunk_overlap)]


@lru_cache(maxsize=1)
def _get_embedder() -> "SentenceTransformer":
    """Load and cache the sentence-transformer model.

    First call downloads weights (~80MB) and is slow; subsequent calls are
    instant. The model is process-local: each worker holds its own copy.
    """
    from sentence_transformers import SentenceTransformer

    from document_brain.config import settings

    logger.info("Loading embedding model %r...", settings.embedding_model)
    return cast("SentenceTransformer", SentenceTransformer(settings.embedding_model))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings into vectors using the configured model.

    Args:
        texts: Strings to embed. May be empty.

    Returns:
        List of vectors, same length as input, in the same order.
        Each vector is a list of floats of length `embedding_dimensions`.
    """
    if not texts:
        return []
    model = _get_embedder()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [vector.tolist() for vector in vectors]


def _make_point_id(document_id: str, chunk_index: int) -> str:
    """Build a stable, unique id for a chunk so re-ingesting is idempotent.

    Combining document_id with chunk_index means uploading the same PDF twice
    produces the same point ids, and Qdrant overwrites rather than duplicates.
    """
    raw = f"{document_id}:{chunk_index}"
    # Qdrant accepts UUIDs or 64-bit ints as point ids; we hash to a UUID-5-like
    # hex string for deterministic-but-unique ids.
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def ingest_document(filename: str, pdf_bytes: bytes) -> int:
    """Top-level ingestion: PDF → pages → chunks → embeddings → Qdrant.

    Args:
        filename: Logical name for the document. Used in citations and as part
            of point ids so re-uploading the same filename overwrites.
        pdf_bytes: Raw PDF bytes.

    Returns:
        Number of chunks ingested.

    Raises:
        pypdf.errors.PdfReadError: If the bytes are not a valid PDF.
    """
    ensure_collection_exists()

    pages = extract_text_from_pdf(pdf_bytes)
    if not pages:
        logger.warning("No extractable text in %r. Nothing ingested.", filename)
        return 0

    # Document id is a hash of the filename so re-ingesting the same file
    # produces the same point ids → idempotent upsert.
    document_id = hashlib.sha256(filename.encode()).hexdigest()[:16]

    # Build flat list of chunks across all pages, retaining page numbers.
    chunked: list[dict[str, str | int]] = []
    for page_info in pages:
        page_number = page_info["page"]
        assert isinstance(page_number, int)
        page_text = page_info["text"]
        assert isinstance(page_text, str)
        for chunk in chunk_text(page_text):
            chunked.append({"page": page_number, "text": chunk})

    if not chunked:
        logger.warning("PDF %r produced no chunks. Nothing ingested.", filename)
        return 0

    # Embed all chunks in one batched call — much faster than embedding one at a time.
    vectors = embed_texts([cast(str, c["text"]) for c in chunked])

    # Build Qdrant points.
    points = [
        PointStruct(
            id=_make_point_id(document_id, i),
            vector=vector,
            payload={
                "filename": filename,
                "document_id": document_id,
                "page": chunk["page"],
                "chunk_index": i,
                "text": chunk["text"],
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunked, vectors, strict=True))
    ]

    upsert_chunks(points)
    logger.info("Ingested %d chunks from %r.", len(points), filename)
    return len(points)
