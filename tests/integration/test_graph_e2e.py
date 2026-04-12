"""
Module: test_graph_e2e
Purpose: End-to-end integration test for the full agent graph.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies the full
    agent pipeline produces valid results with a real LLM.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""



import pytest
from langchain_core.messages import AIMessage

from esg_auditor.config import get_settings


def _has_real_api_key() -> bool:
    """Check if a real Anthropic API key is available."""
    try:
        settings = get_settings()
        key = settings.anthropic_api_key
        return bool(key) and not key.startswith("test-")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_real_api_key(),
    reason="E2E test requires a real ANTHROPIC_API_KEY",
)


class TestGraphEndToEnd:
    """End-to-end test requiring a real Claude API call."""

    def test_graph_responds_to_audit_request(
        self,
    ) -> None:
        """Full graph invocation should produce results.

        NOTE: Does NOT assert specific content — LLM
        output is non-deterministic.
        """
        from esg_auditor.agents.graph import (
            build_graph,
            make_initial_state,
        )

        settings = get_settings()
        graph = build_graph(settings)

        result = graph.invoke(
            make_initial_state(
                "Briefly summarise ESG risks "
                "for Apple (AAPL)"
            ),
            config={
                "configurable": {
                    "thread_id": "e2e-test-001"
                }
            },
        )

        # Structural assertions only
        assert result["state_version"] >= 1
        assert len(result["executed_agents"]) > 0
        assert isinstance(
            result["messages"][-1], AIMessage
        )
        assert len(result["messages"][-1].content) > 0
