"""
Module: test_tools
Purpose: Unit tests for all external data integration tools.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies tool contracts
    without making real API calls.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from unittest.mock import MagicMock, patch

import pytest

from esg_auditor.core.exceptions import DataFetchError


class TestFetchEsgNews:
    """Tests for the marketaux.fetch_esg_news tool."""

    def test_returns_formatted_headlines_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mocked success returns formatted headlines."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test")
        monkeypatch.setenv("MARKETAUX_API_KEY", "test-key")

        mock_response = {
            "data": [
                {
                    "title": "Tesla ESG upgrade",
                    "entities": [
                        {"sentiment_score": 0.85}
                    ],
                },
                {
                    "title": "TSLA carbon report",
                    "entities": [],
                },
            ]
        }

        with patch(
            "esg_auditor.tools.marketaux._fetch_marketaux_news",
            return_value=mock_response,
        ):
            from esg_auditor.tools.marketaux import (
                fetch_esg_news,
            )

            result = fetch_esg_news.invoke("TSLA")

        assert "Tesla ESG upgrade" in result
        assert "sentiment: 0.85" in result
        assert "sentiment: N/A" in result

    def test_returns_error_string_on_data_fetch_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DataFetchError should be caught and returned as ERROR string."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test")
        monkeypatch.setenv("MARKETAUX_API_KEY", "test-key")

        with patch(
            "esg_auditor.tools.marketaux._fetch_marketaux_news",
            side_effect=DataFetchError("timeout"),
        ):
            from esg_auditor.tools.marketaux import (
                fetch_esg_news,
            )

            result = fetch_esg_news.invoke("TSLA")

        assert result.startswith("ERROR:")

    def test_returns_no_news_message_when_data_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty data list returns informative message."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test")
        monkeypatch.setenv("MARKETAUX_API_KEY", "test-key")

        with patch(
            "esg_auditor.tools.marketaux._fetch_marketaux_news",
            return_value={"data": []},
        ):
            from esg_auditor.tools.marketaux import (
                fetch_esg_news,
            )

            result = fetch_esg_news.invoke("TSLA")

        assert "No ESG news" in result


class TestSearchSecFilings:
    """Tests for the sec_edgar.search_sec_filings tool."""

    def test_returns_filing_list_on_success(self) -> None:
        """Mocked success returns formatted filings."""
        mock_hits = [
            {
                "_id": "0001193125-24-012345",
                "_source": {
                    "form_type": "10-K",
                    "file_description": "Annual Report",
                    "file_date": "2024-03-15",
                },
            }
        ]

        with patch(
            "esg_auditor.tools.sec_edgar._search_edgar",
            return_value=mock_hits,
        ):
            from esg_auditor.tools.sec_edgar import (
                search_sec_filings,
            )

            result = search_sec_filings.invoke(
                {"company_name": "Apple Inc."}
            )

        assert "10-K" in result
        assert "Annual Report" in result

    def test_returns_error_string_on_http_failure(
        self,
    ) -> None:
        """DataFetchError should return ERROR string."""
        with patch(
            "esg_auditor.tools.sec_edgar._search_edgar",
            side_effect=DataFetchError("HTTP 500"),
        ):
            from esg_auditor.tools.sec_edgar import (
                search_sec_filings,
            )

            result = search_sec_filings.invoke(
                {"company_name": "Apple Inc."}
            )

        assert result.startswith("ERROR:")

    def test_includes_accession_number_in_output(
        self,
    ) -> None:
        """Output must include accession number for audit."""
        mock_hits = [
            {
                "_id": "0001193125-24-099999",
                "_source": {
                    "form_type": "10-Q",
                    "file_description": "Quarterly",
                    "file_date": "2024-06-30",
                },
            }
        ]

        with patch(
            "esg_auditor.tools.sec_edgar._search_edgar",
            return_value=mock_hits,
        ):
            from esg_auditor.tools.sec_edgar import (
                search_sec_filings,
            )

            result = search_sec_filings.invoke(
                {"company_name": "Apple Inc."}
            )

        assert "0001193125-24-099999" in result


class TestGetYfinanceEsgScore:
    """Tests for the yfinance_tools.get_yfinance_esg_score tool."""

    def test_returns_error_when_sustainability_is_none(
        self,
    ) -> None:
        """None sustainability data returns ERROR string."""
        with patch(
            "esg_auditor.tools.yfinance_tools"
            "._get_yfinance_sustainability",
            side_effect=DataFetchError("No data"),
        ):
            from esg_auditor.tools.yfinance_tools import (
                get_yfinance_esg_score,
            )

            result = get_yfinance_esg_score.invoke("AAPL")

        assert result.startswith("ERROR:")

    def test_includes_freshness_warning_in_output(
        self,
    ) -> None:
        """Output must include freshness warning."""
        mock_data = {
            "totalEsg": {0: 25.0},
            "environmentScore": {0: 8.0},
        }

        with patch(
            "esg_auditor.tools.yfinance_tools"
            "._get_yfinance_sustainability",
            return_value=mock_data,
        ):
            from esg_auditor.tools.yfinance_tools import (
                get_yfinance_esg_score,
            )

            result = get_yfinance_esg_score.invoke("AAPL")

        assert "verify data freshness" in result.lower()


class TestAnalyzeSentimentEsg:
    """Tests for the finbert.analyze_sentiment_esg tool."""

    def test_returns_formatted_lines_on_success(
        self,
    ) -> None:
        """Mocked FinBERT returns formatted output."""
        mock_results = [
            {
                "text": "Strong sustainability",
                "sentiment": "positive",
                "sentiment_score": 0.95,
                "esg_category": "Environmental",
                "esg_confidence": 0.88,
            }
        ]

        mock_models = MagicMock()
        mock_models.analyze.return_value = mock_results

        with patch(
            "esg_auditor.tools.finbert.get_finbert_models",
            return_value=mock_models,
        ):
            from esg_auditor.tools.finbert import (
                analyze_sentiment_esg,
            )

            result = analyze_sentiment_esg.invoke(
                {"texts": ["Strong sustainability"]}
            )

        assert "Environmental" in result
        assert "positive" in result

    def test_returns_error_on_empty_texts(self) -> None:
        """Empty texts list returns ERROR string."""
        from esg_auditor.tools.finbert import (
            analyze_sentiment_esg,
        )

        result = analyze_sentiment_esg.invoke(
            {"texts": []}
        )
        assert result.startswith("ERROR:")
        assert "empty" in result.lower()


class TestSearchRegulatoryDocs:
    """Tests for qdrant_search.search_regulatory_docs."""

    def test_returns_formatted_chunks_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mocked Qdrant returns formatted chunks."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test")

        mock_point = MagicMock()
        mock_point.payload = {
            "document_type": "10-K",
            "jurisdiction": "US-Federal",
            "section": "Item 1A",
            "text": "Climate risk disclosure...",
            "source": "SEC-10K-AAPL-2024",
        }

        mock_results = MagicMock()
        mock_results.points = [mock_point]

        mock_client = MagicMock()
        mock_client.query_points.return_value = (
            mock_results
        )

        mock_embed = MagicMock()
        mock_embed.embed.return_value = iter(
            [[0.1] * 384]
        )

        with (
            patch(
                "esg_auditor.tools.qdrant_search"
                ".get_qdrant_client",
                return_value=mock_client,
            ),
            patch(
                "esg_auditor.tools.qdrant_search"
                ".get_embedding_model",
                return_value=mock_embed,
            ),
        ):
            from esg_auditor.tools.qdrant_search import (
                search_regulatory_docs,
            )

            result = search_regulatory_docs.invoke(
                {"query": "climate risk"}
            )

        assert "10-K" in result
        assert "Item 1A" in result

    def test_includes_source_id_in_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Output must include source ID for SR 11-7."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test")

        mock_point = MagicMock()
        mock_point.payload = {
            "document_type": "CSRD",
            "jurisdiction": "EU",
            "section": "Section 2",
            "text": "ESG reporting standards",
            "source": "CSRD-EU-2024-001",
        }

        mock_results = MagicMock()
        mock_results.points = [mock_point]

        mock_client = MagicMock()
        mock_client.query_points.return_value = (
            mock_results
        )

        mock_embed = MagicMock()
        mock_embed.embed.return_value = iter(
            [[0.1] * 384]
        )

        with (
            patch(
                "esg_auditor.tools.qdrant_search"
                ".get_qdrant_client",
                return_value=mock_client,
            ),
            patch(
                "esg_auditor.tools.qdrant_search"
                ".get_embedding_model",
                return_value=mock_embed,
            ),
        ):
            from esg_auditor.tools.qdrant_search import (
                search_regulatory_docs,
            )

            result = search_regulatory_docs.invoke(
                {"query": "ESG reporting"}
            )

        assert "CSRD-EU-2024-001" in result

    def test_returns_error_string_on_embedding_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EmbeddingError should be caught and returned."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test")

        mock_embed = MagicMock()
        mock_embed.embed.side_effect = RuntimeError(
            "model load failed"
        )

        with (
            patch(
                "esg_auditor.tools.qdrant_search"
                ".get_qdrant_client",
                return_value=MagicMock(),
            ),
            patch(
                "esg_auditor.tools.qdrant_search"
                ".get_embedding_model",
                return_value=mock_embed,
            ),
        ):
            from esg_auditor.tools.qdrant_search import (
                search_regulatory_docs,
            )

            result = search_regulatory_docs.invoke(
                {"query": "test query"}
            )

        assert "ERROR" in result
