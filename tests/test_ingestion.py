"""Tests for the PDF ingestion module."""

import io

import pypdf
import pytest

from document_brain.ingestion import chunk_text, extract_text_from_pdf


def _make_pdf(pages: list[str]) -> bytes:
    """Build a minimal in-memory PDF containing the given page texts.

    Used to give tests deterministic, fast PDFs without needing fixture files
    on disk.
    """
    writer = pypdf.PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=200, height=200)
        # pypdf can't write text directly onto a page, so we annotate it.
        # The text becomes extractable via extract_text().
        page.merge_page(_text_page(text))

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _text_page(text: str) -> pypdf.PageObject:
    """Helper: build a single page object containing the given text."""
    # Construct a tiny PDF in memory whose one page contains `text`,
    # then return that page object so it can be merged elsewhere.
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(50, 100, text)
    c.showPage()
    c.save()
    buf.seek(0)
    return pypdf.PdfReader(buf).pages[0]


def test_extracts_text_from_each_page() -> None:
    """Function returns one dict per non-empty page with correct numbering."""
    pdf_bytes = _make_pdf(["Hello world", "Second page content"])
    result = extract_text_from_pdf(pdf_bytes)

    assert len(result) == 2
    assert result[0] == {"page": 1, "text": "Hello world"}
    assert result[1] == {"page": 2, "text": "Second page content"}


def test_skips_pages_with_no_text() -> None:
    """Pages whose extracted text is empty are excluded from output."""
    pdf_bytes = _make_pdf(["Real content", "", "More content"])
    result = extract_text_from_pdf(pdf_bytes)

    # Three pages in, but the empty one is skipped.
    assert len(result) == 2
    # Page numbers still match the original PDF's pages, not the output index.
    assert result[0]["page"] == 1
    assert result[1]["page"] == 3


def test_strips_surrounding_whitespace() -> None:
    """Extracted text has leading/trailing whitespace removed."""
    pdf_bytes = _make_pdf(["   Padded text   "])
    result = extract_text_from_pdf(pdf_bytes)

    assert result[0]["text"] == "Padded text"


def test_raises_on_invalid_pdf_bytes() -> None:
    """Garbage input produces a clear, propagating error."""
    with pytest.raises(pypdf.errors.PdfReadError):
        extract_text_from_pdf(b"this is not a PDF")


def test_chunks_short_text_to_single_chunk() -> None:
    """Text shorter than chunk_size returns as one chunk."""
    result = chunk_text("Hello world.", chunk_size=500, chunk_overlap=50)
    assert result == ["Hello world."]


def test_chunks_empty_text_returns_empty_list() -> None:
    """Whitespace-only input produces no chunks."""
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
    assert chunk_text("   \n\n   ", chunk_size=500, chunk_overlap=50) == []


def test_chunks_respect_size_limit() -> None:
    """No chunk exceeds chunk_size, even with no natural separators."""
    text = "x" * 2000
    result = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert all(len(chunk) <= 500 for chunk in result)
    assert len(result) > 1


def test_chunks_split_at_paragraph_when_possible() -> None:
    """When a paragraph break fits within chunk_size, splitting happens there."""
    text = "First paragraph here." + "\n\n" + ("More text. " * 100)
    result = chunk_text(text, chunk_size=200, chunk_overlap=20)
    # First chunk should end at the paragraph break, not mid-sentence.
    assert result[0] == "First paragraph here."


def test_chunks_overlap_between_adjacent() -> None:
    """Consecutive chunks share approximately chunk_overlap characters."""
    text = "word " * 500  # produces long text with word boundaries
    result = chunk_text(text, chunk_size=200, chunk_overlap=40)
    # The end of chunk N should appear at the start of chunk N+1.
    for i in range(len(result) - 1):
        tail = result[i][-30:]  # last 30 chars
        # Some portion of tail should appear in the next chunk's start
        assert any(tail[j:] in result[i + 1][:60] for j in range(len(tail)))


def test_chunks_raises_when_overlap_too_large() -> None:
    """Misconfiguration is caught explicitly rather than infinite-looping."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("some text", chunk_size=50, chunk_overlap=50)
