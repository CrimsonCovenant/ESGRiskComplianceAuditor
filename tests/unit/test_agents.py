"""
Module: test_agents
Purpose: Unit tests for agent logic (all LLM calls mocked).
SR 11-7 Relevance: Pillar 2 (Validation) — verifies agent
    behaviour contracts without making real API calls.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from esg_auditor.config import get_settings


class TestAdvisorNode:
    """Tests for the _advisor_node function."""

    def test_advisor_node_increments_state_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Advisor should increment state_version by 1."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from esg_auditor.agents.graph import _advisor_node

        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(
            content="Test response"
        )

        state = {
            "messages": [
                HumanMessage(content="test query")
            ],
            "current_agent": "",
            "iteration_count": 0,
            "state_version": 5,
            "created_by": "",
            "executed_agents": [],
            "client_profile": {},
            "audit_request": {},
            "research_results": {},
            "esg_report": {},
        }

        result = _advisor_node(
            state,
            advisor_model=mock_model,
            max_iterations=10,
        )

        assert result["state_version"] == 6

    def test_advisor_node_appends_to_executed_agents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Advisor should append 'advisor' to executed_agents."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from esg_auditor.agents.graph import _advisor_node

        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(
            content="Response"
        )

        state = {
            "messages": [
                HumanMessage(content="test query")
            ],
            "current_agent": "",
            "iteration_count": 0,
            "state_version": 0,
            "created_by": "",
            "executed_agents": ["previous"],
            "client_profile": {},
            "audit_request": {},
            "research_results": {},
            "esg_report": {},
        }

        result = _advisor_node(
            state,
            advisor_model=mock_model,
            max_iterations=10,
        )

        assert "advisor" in result["executed_agents"]
        assert result["executed_agents"] == [
            "previous",
            "advisor",
        ]

    def test_advisor_returns_shutdown_at_iteration_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At max iterations, advisor returns without LLM call."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from esg_auditor.agents.graph import _advisor_node

        mock_model = MagicMock()

        state = {
            "messages": [
                HumanMessage(content="test query")
            ],
            "current_agent": "",
            "iteration_count": 10,
            "state_version": 0,
            "created_by": "",
            "executed_agents": [],
            "client_profile": {},
            "audit_request": {},
            "research_results": {},
            "esg_report": {},
        }

        result = _advisor_node(
            state,
            advisor_model=mock_model,
            max_iterations=10,
        )

        # LLM should NOT have been called
        mock_model.invoke.assert_not_called()
        assert "iteration limit" in (
            result["messages"][0].content.lower()
        )


class TestShouldContinue:
    """Tests for _should_continue routing."""

    def test_returns_end_at_iteration_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return END when at max iterations."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from langgraph.graph import END

        from esg_auditor.agents.graph import (
            _should_continue,
        )

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "1",
                            "name": "test",
                            "args": {},
                        }
                    ],
                )
            ],
            "iteration_count": 10,
            "state_version": 0,
            "created_by": "",
            "executed_agents": [],
            "current_agent": "",
            "client_profile": {},
            "audit_request": {},
            "research_results": {},
            "esg_report": {},
        }

        result = _should_continue(
            state, max_iterations=10
        )
        assert result == END

    def test_returns_tools_when_tool_calls_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return 'tools' when tool calls detected."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from esg_auditor.agents.graph import (
            _should_continue,
        )

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "1",
                            "name": "test",
                            "args": {},
                        }
                    ],
                )
            ],
            "iteration_count": 0,
            "state_version": 0,
            "created_by": "",
            "executed_agents": [],
            "current_agent": "",
            "client_profile": {},
            "audit_request": {},
            "research_results": {},
            "esg_report": {},
        }

        result = _should_continue(
            state, max_iterations=10
        )
        assert result == "tools"


class TestBuildGraph:
    """Tests for build_graph factory."""

    def test_build_graph_returns_compiled_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_graph() should return a compiled graph."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from esg_auditor.agents.graph import build_graph

        settings = get_settings()
        graph = build_graph(settings)

        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_get_advisor_tools_returns_two_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_get_advisor_tools should return exactly 2 tools."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-k")
        monkeypatch.setenv("QDRANT_URL", "http://localhost")
        monkeypatch.setenv("QDRANT_API_KEY", "test-k")
        get_settings.cache_clear()

        from esg_auditor.agents.graph import (
            _get_advisor_tools,
        )

        settings = get_settings()
        tools = _get_advisor_tools(settings)

        assert len(tools) == 2

        tool_names = {t.name for t in tools}
        assert "consult_analyst" in tool_names
        assert "get_client_profile" in tool_names
