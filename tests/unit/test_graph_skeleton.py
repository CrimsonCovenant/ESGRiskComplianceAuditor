"""
Module: test_graph_skeleton
Purpose: Unit tests for graph construction and initial state.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies that the graph
    wiring is correct and state initialisation is consistent.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import pytest

from esg_auditor.agents.graph import build_graph, make_initial_state
from esg_auditor.config import get_settings


class TestGraphSkeleton:
    """Tests for graph construction and make_initial_state."""

    def test_build_graph_returns_compiled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_graph() should return a compiled graph."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_make_initial_state_structure(self) -> None:
        """make_initial_state should produce a valid AgentState."""
        state = make_initial_state("test message")
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "test message"
        assert state["current_agent"] == ""
        assert state["iteration_count"] == 0
        assert state["state_version"] == 0
        assert state["created_by"] == ""
        assert state["executed_agents"] == []
        assert state["client_profile"] == {}
        assert state["audit_request"] == {}
        assert state["research_results"] == {}
        assert state["esg_report"] == {}
