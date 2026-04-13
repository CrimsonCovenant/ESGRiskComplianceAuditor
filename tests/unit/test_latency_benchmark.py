"""
Module: test_latency_benchmark
Purpose: Unit tests for the latency benchmark pipeline.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies the latency
    benchmark's output contract and safety guards.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from esg_auditor.core.exceptions import ConfigurationError
from esg_auditor.eval.latency_benchmark import (
    run_latency_benchmark,
)


def _make_mock_settings(
    api_key: str = "sk-real-key-12345",
) -> MagicMock:
    """Build a mock Settings with a given API key."""
    settings = MagicMock()
    settings.anthropic_api_key = api_key
    return settings


def _make_slow_graph() -> MagicMock:
    """Return a mock graph whose invoke sleeps 10ms."""
    graph = MagicMock()

    def _slow_invoke(*args: object, **kwargs: object) -> None:
        time.sleep(0.01)

    graph.invoke = _slow_invoke
    return graph


class TestRunLatencyBenchmark:
    """Tests for run_latency_benchmark function."""

    @patch(
        "esg_auditor.eval.latency_benchmark.build_graph"
    )
    def test_returns_expected_percentile_keys(
        self,
        mock_build_graph: MagicMock,
    ) -> None:
        """Result has all required percentile keys."""
        mock_build_graph.return_value = (
            _make_slow_graph()
        )
        settings = _make_mock_settings()

        result = run_latency_benchmark(
            n_runs=10, settings=settings
        )
        expected_keys = {
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "mean_ms",
            "min_ms",
            "max_ms",
            "n_runs",
            "query",
        }
        assert set(result.keys()) == expected_keys

    def test_raises_on_test_api_key(self) -> None:
        """ConfigurationError when key starts with test-."""
        settings = _make_mock_settings(
            api_key="test-placeholder-key"
        )
        with pytest.raises(
            ConfigurationError,
            match="ANTHROPIC_API_KEY",
        ):
            run_latency_benchmark(settings=settings)

    @patch(
        "esg_auditor.eval.latency_benchmark.build_graph"
    )
    def test_n_runs_matches_output(
        self,
        mock_build_graph: MagicMock,
    ) -> None:
        """result n_runs equals the argument passed in."""
        mock_build_graph.return_value = (
            _make_slow_graph()
        )
        settings = _make_mock_settings()

        result = run_latency_benchmark(
            n_runs=15, settings=settings
        )
        assert result["n_runs"] == 15

    @patch(
        "esg_auditor.eval.latency_benchmark.build_graph"
    )
    def test_all_times_are_positive(
        self,
        mock_build_graph: MagicMock,
    ) -> None:
        """p50, p95, p99 are all positive floats."""
        mock_build_graph.return_value = (
            _make_slow_graph()
        )
        settings = _make_mock_settings()

        result = run_latency_benchmark(
            n_runs=10, settings=settings
        )
        assert result["p50_ms"] > 0
        assert result["p95_ms"] > 0
        assert result["p99_ms"] > 0
