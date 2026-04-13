"""
Module: test_finbert_eval
Purpose: Unit tests for the FinBERT accuracy evaluation pipeline.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies the NLP model
    evaluation pipeline's output contract and input guards.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from esg_auditor.eval.finbert_eval import (
    LABEL_MAP,
    evaluate_finbert_accuracy,
)


def _mock_phrasebank() -> dict[str, list]:
    """Return a minimal fake phrasebank dataset."""
    return {
        "sentence": [
            "Revenue increased significantly.",
            "Losses were reported this quarter.",
            "The company maintained operations.",
        ]
        * 100,  # 300 sentences total
        "label": [2, 0, 1] * 100,
    }


def _mock_sentiment_pipe(
    texts: list[str], **kwargs: object
) -> list[dict[str, str | float]]:
    """Return matching predictions for test data."""
    label_cycle = ["positive", "negative", "neutral"]
    return [
        {"label": label_cycle[i % 3], "score": 0.95}
        for i in range(len(texts))
    ]


class TestEvaluateFinbertAccuracy:
    """Tests for evaluate_finbert_accuracy function."""

    @patch(
        "esg_auditor.eval.finbert_eval.get_finbert_models"
    )
    @patch(
        "esg_auditor.eval.finbert_eval._load_phrasebank"
    )
    @patch(
        "esg_auditor.eval.finbert_eval.get_settings"
    )
    def test_returns_expected_keys(
        self,
        mock_settings: MagicMock,
        mock_load_phrasebank: MagicMock,
        mock_get_models: MagicMock,
    ) -> None:
        """Result dict has all required keys."""
        mock_load_phrasebank.return_value = (
            _mock_phrasebank()
        )

        mock_models = MagicMock()
        mock_models.sentiment_pipe = (
            _mock_sentiment_pipe
        )
        mock_get_models.return_value = mock_models

        result = evaluate_finbert_accuracy()
        expected_keys = {
            "accuracy",
            "sample_size",
            "negative_f1",
            "neutral_f1",
            "positive_f1",
            "dataset",
            "model",
        }
        assert set(result.keys()) == expected_keys

    @patch(
        "esg_auditor.eval.finbert_eval.get_finbert_models"
    )
    @patch(
        "esg_auditor.eval.finbert_eval._load_phrasebank"
    )
    @patch(
        "esg_auditor.eval.finbert_eval.get_settings"
    )
    def test_accuracy_is_float_between_0_and_1(
        self,
        mock_settings: MagicMock,
        mock_load_phrasebank: MagicMock,
        mock_get_models: MagicMock,
    ) -> None:
        """Accuracy value is a float in [0.0, 1.0]."""
        mock_load_phrasebank.return_value = (
            _mock_phrasebank()
        )

        mock_models = MagicMock()
        mock_models.sentiment_pipe = (
            _mock_sentiment_pipe
        )
        mock_get_models.return_value = mock_models

        result = evaluate_finbert_accuracy()
        assert isinstance(result["accuracy"], float)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_label_map_has_all_three_labels(
        self,
    ) -> None:
        """LABEL_MAP covers positive, negative, neutral."""
        assert set(LABEL_MAP.keys()) == {
            "positive",
            "negative",
            "neutral",
        }

    @patch(
        "esg_auditor.eval.finbert_eval.get_finbert_models"
    )
    @patch(
        "esg_auditor.eval.finbert_eval._load_phrasebank"
    )
    @patch(
        "esg_auditor.eval.finbert_eval.get_settings"
    )
    def test_sample_size_capped_at_200_by_default(
        self,
        mock_settings: MagicMock,
        mock_load_phrasebank: MagicMock,
        mock_get_models: MagicMock,
    ) -> None:
        """Default mode samples 200 sentences max."""
        mock_load_phrasebank.return_value = (
            _mock_phrasebank()
        )

        mock_models = MagicMock()
        mock_models.sentiment_pipe = (
            _mock_sentiment_pipe
        )
        mock_get_models.return_value = mock_models

        result = evaluate_finbert_accuracy(full=False)
        assert result["sample_size"] == 200
