"""
Module: test_ragas_eval
Purpose: Unit tests for the RAGAS evaluation pipeline.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies the
    evaluation pipeline's input validation and output
    contract before production use.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from unittest.mock import MagicMock, patch

import pytest

from esg_auditor.core.exceptions import StructuredOutputError
from esg_auditor.eval.ragas_eval import run_evaluation


VALID_EVAL_DATA: dict[str, list] = {
    "question": ["What is SR 11-7?"],
    "answer": [
        "SR 11-7 is Federal Reserve guidance "
        "on model risk management."
    ],
    "contexts": [
        [
            "SR 11-7 requires three pillars of "
            "model risk management."
        ]
    ],
    "ground_truth": [
        "SR 11-7 establishes requirements for "
        "model development, validation, and "
        "governance."
    ],
}

def _make_mock_ragas_result() -> MagicMock:
    """Build a mock EvaluationResult with _scores_dict."""
    mock = MagicMock()
    mock._scores_dict = {
        "faithfulness": [0.85],
        "llm_context_precision_with_reference": [0.90],
        "context_recall": [0.88],
    }
    return mock


class TestRunEvaluation:
    """Tests for run_evaluation function."""

    def test_raises_on_empty_questions(self) -> None:
        """Empty question list raises StructuredOutputError."""
        data = {**VALID_EVAL_DATA, "question": []}
        with pytest.raises(
            StructuredOutputError,
            match="at least one",
        ):
            run_evaluation(data, MagicMock())

    def test_raises_on_missing_keys(self) -> None:
        """Missing required key raises StructuredOutputError."""
        data = {"question": ["q"], "answer": ["a"]}
        with pytest.raises(
            StructuredOutputError,
            match="missing required keys",
        ):
            run_evaluation(data, MagicMock())

    def test_raises_on_missing_contexts(self) -> None:
        """Missing contexts key raises StructuredOutputError."""
        data = {
            "question": ["q"],
            "answer": ["a"],
            "ground_truth": ["gt"],
        }
        with pytest.raises(
            StructuredOutputError,
            match="missing required keys",
        ):
            run_evaluation(data, MagicMock())

    @patch("esg_auditor.eval.ragas_eval.evaluate")
    @patch(
        "esg_auditor.eval.ragas_eval.ChatAnthropic"
    )
    def test_returns_expected_keys(
        self,
        mock_anthropic: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Mocked RAGAS returns dict with required metric keys."""
        mock_evaluate.return_value = _make_mock_ragas_result()
        result = run_evaluation(
            VALID_EVAL_DATA, MagicMock()
        )
        assert "faithfulness" in result
        assert "context_precision" in result
        assert "context_recall" in result
        assert "response_relevancy" not in result

    @patch("esg_auditor.eval.ragas_eval.evaluate")
    @patch(
        "esg_auditor.eval.ragas_eval.ChatAnthropic"
    )
    def test_metric_values_are_floats(
        self,
        mock_anthropic: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """All returned metric values are floats."""
        mock_evaluate.return_value = _make_mock_ragas_result()
        result = run_evaluation(
            VALID_EVAL_DATA, MagicMock()
        )
        for key, value in result.items():
            assert isinstance(value, float), (
                f"{key} should be float, got {type(value)}"
            )

    @patch("esg_auditor.eval.ragas_eval.evaluate")
    @patch(
        "esg_auditor.eval.ragas_eval.ChatAnthropic"
    )
    def test_wraps_unexpected_exception(
        self,
        mock_anthropic: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Unexpected exceptions wrap in StructuredOutputError."""
        mock_evaluate.side_effect = RuntimeError(
            "API down"
        )
        with pytest.raises(
            StructuredOutputError,
            match="RAGAS evaluation failed",
        ):
            run_evaluation(
                VALID_EVAL_DATA, MagicMock()
            )
