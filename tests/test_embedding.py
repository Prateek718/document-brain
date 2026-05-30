"""Tests for the text embedding function."""

from document_brain.config import settings
from document_brain.ingestion import embed_texts


def test_embed_empty_list_returns_empty() -> None:
    """Embedding an empty list returns an empty list, not an error."""
    assert embed_texts([]) == []


def test_embed_produces_vectors_of_configured_dimension() -> None:
    """Every output vector has length equal to settings.embedding_dimensions."""
    vectors = embed_texts(["hello world", "another piece of text"])
    assert len(vectors) == 2
    assert all(len(v) == settings.embedding_dimensions for v in vectors)


def test_embed_is_deterministic() -> None:
    """The same input always produces the same vector."""
    [v1] = embed_texts(["deterministic input"])
    [v2] = embed_texts(["deterministic input"])
    assert v1 == v2


def test_embed_distinguishes_different_meanings() -> None:
    """Different inputs produce different vectors (a sanity check on the model)."""
    [v1, v2] = embed_texts(["the cat sat on the mat", "quantum mechanics is complex"])
    assert v1 != v2


def test_embed_returns_floats_not_other_types() -> None:
    """Vector elements are plain Python floats, ready for JSON serialization."""
    [vector] = embed_texts(["sample"])
    assert all(isinstance(x, float) for x in vector)
