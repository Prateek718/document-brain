"""PDF ingestion: extract text from PDF documents page by page."""

import io

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
