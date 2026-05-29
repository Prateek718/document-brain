"""PDF ingestion: extract text from PDF documents page by page."""

import io
from typing import Final

import pypdf


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
