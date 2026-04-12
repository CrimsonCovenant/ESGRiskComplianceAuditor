"""
Module: test_rag_pipeline
Purpose: Integration tests for the full RAG pipeline (Qdrant).
SR 11-7 Relevance: Pillar 2 (Validation) — verifies end-to-end
    document ingestion and retrieval against a live Qdrant cluster.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import os

import pytest


def _is_real_qdrant() -> bool:
    """Check if QDRANT_URL points to a real cluster."""
    url = os.environ.get("QDRANT_URL", "http://localhost")
    return not (
        url.startswith("http://localhost")
        or "test-" in url
    )


pytestmark = pytest.mark.skipif(
    not _is_real_qdrant(),
    reason=(
        "Integration test requires a live Qdrant cluster "
        "(set QDRANT_URL to a real endpoint)"
    ),
)


class TestRagPipelineIntegration:
    """Integration tests requiring a live Qdrant cluster."""

    def test_ingest_and_search_cycle(self) -> None:
        """Ingest 3 documents and verify search returns results."""
        from esg_auditor.config import get_settings
        from esg_auditor.rag.ingest import (
            ingest_text,
        )
        from esg_auditor.tools.qdrant_search import (
            get_embedding_model,
            get_qdrant_client,
            search_regulatory_docs,
        )

        settings = get_settings()
        client = get_qdrant_client(
            settings.qdrant_url, settings.qdrant_api_key
        )
        model = get_embedding_model()
        collection = "test_integration_rag"

        metadata = {
            "source": "TEST-DOC-001",
            "document_type": "10-K",
            "company": "Test Corp",
            "ticker": "TEST",
            "date": "2024-01-01",
            "jurisdiction": "US-Federal",
            "section": "",
            "chunk_index": 0,
        }

        text = (
            "Item 1. Risk Factors\n"
            "Climate risk is a material concern "
            "for the company's operations.\n\n"
            "Item 2. ESG Disclosures\n"
            "The company has committed to net-zero "
            "emissions by 2050.\n\n"
            "Item 3. Governance\n"
            "The board oversees all ESG initiatives."
        )

        count = ingest_text(
            text, metadata, client, model, collection
        )
        assert count >= 3

        result = search_regulatory_docs.invoke(
            {"query": "climate risk emissions"}
        )
        assert "ERROR" not in result
