"""
Module: test_chunker
Purpose: Unit tests for the regulatory document chunker.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies document
    lineage preservation through the chunking pipeline.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import pytest
from langchain_core.documents import Document

from esg_auditor.rag.chunker import (
    chunk_regulatory_document,
)

VALID_METADATA: dict[str, str | int] = {
    "source": "SEC-10K-TEST-2024",
    "document_type": "10-K",
    "company": "Test Corp",
    "ticker": "TEST",
    "date": "2024-01-01",
    "jurisdiction": "US-Federal",
    "section": "",
    "chunk_index": 0,
}


class TestChunkRegulatoryDocument:
    """Tests for chunk_regulatory_document."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Blank text should return an empty list."""
        result = chunk_regulatory_document(
            "", VALID_METADATA
        )
        assert result == []

    def test_whitespace_only_returns_empty_list(
        self,
    ) -> None:
        """Whitespace-only text returns empty list."""
        result = chunk_regulatory_document(
            "   \n\n  ", VALID_METADATA
        )
        assert result == []

    def test_missing_metadata_key_raises_value_error(
        self,
    ) -> None:
        """Missing required key should raise ValueError."""
        incomplete = {
            k: v
            for k, v in VALID_METADATA.items()
            if k != "source"
        }
        with pytest.raises(ValueError):
            chunk_regulatory_document(
                "Some text", incomplete
            )

    def test_missing_key_name_in_error_message(
        self,
    ) -> None:
        """Error message should name the missing key."""
        incomplete = {
            k: v
            for k, v in VALID_METADATA.items()
            if k != "jurisdiction"
        }
        with pytest.raises(
            ValueError, match="jurisdiction"
        ):
            chunk_regulatory_document(
                "Some text", incomplete
            )

    def test_each_chunk_has_chunk_id(self) -> None:
        """Every chunk must have a unique chunk_id."""
        text = "Item 1. Risk Factors\nSome risk content."
        result = chunk_regulatory_document(
            text, VALID_METADATA
        )
        assert len(result) > 0
        for doc in result:
            assert "chunk_id" in doc.metadata
            assert len(doc.metadata["chunk_id"]) == 36

    def test_each_chunk_has_chunk_index(self) -> None:
        """Every chunk must have a chunk_index."""
        text = "Item 1. Risk Factors\nContent here."
        result = chunk_regulatory_document(
            text, VALID_METADATA
        )
        for i, doc in enumerate(result):
            assert doc.metadata["chunk_index"] == i

    def test_chunk_size_does_not_exceed_limit(
        self,
    ) -> None:
        """Chunks should not exceed 1000 characters."""
        long_text = (
            "Item 1. Risk Factors\n" + "x " * 2000
        )
        result = chunk_regulatory_document(
            long_text, VALID_METADATA
        )
        for doc in result:
            assert len(doc.page_content) <= 1000

    def test_two_item_sections_produce_two_groups(
        self,
    ) -> None:
        """Two Item sections should produce chunks from both."""
        text = (
            "Item 1. Risk Factors\n"
            "Risk content paragraph one.\n\n"
            "Item 2. Properties\n"
            "Property content paragraph two."
        )
        result = chunk_regulatory_document(
            text, VALID_METADATA
        )
        sections = {
            doc.metadata["section"] for doc in result
        }
        assert len(sections) >= 2

    def test_returns_list_of_document_objects(
        self,
    ) -> None:
        """Return type should be a list of Documents."""
        text = "Item 1. Overview\nSome overview content."
        result = chunk_regulatory_document(
            text, VALID_METADATA
        )
        assert isinstance(result, list)
        for doc in result:
            assert isinstance(doc, Document)
