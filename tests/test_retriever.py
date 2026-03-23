import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retriever import retrieve_chunks, format_context  # noqa: E402


def test_retrieve_chunks_returns_results():
    """Test retriever returns chunks for valid query"""
    chunks = retrieve_chunks(
        "Apple revenue growth",
        ticker="AAPL",
        k=3
    )
    assert chunks is not None
    assert len(chunks) <= 3


def test_retrieve_chunks_has_metadata():
    """Test each chunk has required metadata"""
    chunks = retrieve_chunks(
        "Microsoft earnings",
        ticker="MSFT",
        k=2
    )
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "ticker" in chunk["metadata"]
        assert "filed_date" in chunk["metadata"]


def test_format_context_not_empty():
    """Test format_context returns non-empty string"""
    chunks = retrieve_chunks(
        "Apple revenue",
        ticker="AAPL",
        k=2
    )
    if chunks:
        context = format_context(chunks)
        assert len(context) > 0
        assert "Source" in context
