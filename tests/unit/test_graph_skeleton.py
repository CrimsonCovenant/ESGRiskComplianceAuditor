"""
Module: test_graph_skeleton
Purpose: Unit tests for the Phase 1 LangGraph skeleton.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies that the graph
    wiring is correct, placeholder nodes produce expected outputs,
    and state versioning fields are properly managed.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from esg_auditor.agents.graph import build_graph, make_initial_state


class TestGraphSkeleton:
    """Tests for the Phase 1 graph skeleton."""

    def test_build_graph_returns_compiled(self) -> None:
        """build_graph() should return a compiled graph object."""
        graph = build_graph()
        assert graph is not None

    def test_graph_smoke(self) -> None:
        """Full smoke test for the Phase 1 graph skeleton.

        Verifies:
        - Last message is '[Advisor placeholder]'
        - current_agent is 'advisor'
        - state_version incremented to 1
        - created_by set to 'advisor'
        - executed_agents contains ['advisor']
        - iteration_count incremented to 1
        """
        graph = build_graph()
        result = graph.invoke(
            make_initial_state("test audit request"),
            config={"configurable": {"thread_id": "smoke-001"}},
        )
        assert (
            result["messages"][-1].content
            == "[Advisor placeholder]"
        )
        assert result["current_agent"] == "advisor"
        assert result["state_version"] == 1
        assert result["created_by"] == "advisor"
        assert result["executed_agents"] == ["advisor"]
        assert result["iteration_count"] == 1

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
