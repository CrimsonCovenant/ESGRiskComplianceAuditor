"""
Module: graph
Purpose: LangGraph StateGraph assembly for the multi-agent ESG auditor.
SR 11-7 Relevance: Pillar 1 (Development) + Pillar 3 (Governance) —
    the graph definition is the executable specification of the agent
    workflow. Every node increments state_version and records
    created_by to establish an immutable audit lineage.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from esg_auditor.core.state import AgentState


def _placeholder_advisor(state: AgentState) -> dict:
    """Placeholder advisor node.

    SR 11-7 Pillar 3: increments state_version and records
    created_by on every transition, establishing the immutable
    audit lineage.
    """
    return {
        "messages": [AIMessage(content="[Advisor placeholder]")],
        "current_agent": "advisor",
        "state_version": state["state_version"] + 1,
        "created_by": "advisor",
        "executed_agents": (
            state["executed_agents"] + ["advisor"]
        ),
        "iteration_count": state["iteration_count"] + 1,
    }


def _placeholder_analyst(state: AgentState) -> dict:
    """Placeholder analyst node.

    SR 11-7 Pillar 3: increments state_version and records
    created_by on every transition, establishing the immutable
    audit lineage.
    """
    return {
        "messages": [AIMessage(content="[Analyst placeholder]")],
        "current_agent": "analyst",
        "state_version": state["state_version"] + 1,
        "created_by": "analyst",
        "executed_agents": (
            state["executed_agents"] + ["analyst"]
        ),
        "iteration_count": state["iteration_count"] + 1,
    }


def _placeholder_client(state: AgentState) -> dict:
    """Placeholder client node.

    SR 11-7 Pillar 3: increments state_version and records
    created_by on every transition, establishing the immutable
    audit lineage.
    """
    return {
        "messages": [AIMessage(content="[Client placeholder]")],
        "current_agent": "client",
        "state_version": state["state_version"] + 1,
        "created_by": "client",
        "executed_agents": (
            state["executed_agents"] + ["client"]
        ),
        "iteration_count": state["iteration_count"] + 1,
    }


def _should_continue(state: AgentState) -> str:
    """Route to END unconditionally.

    Phase 3 replaces this with real tool-call routing logic.
    """
    return END


def build_graph() -> CompiledStateGraph:
    """Assemble and compile the LangGraph StateGraph.

    Factory function — the compiled graph is NOT stored at module
    level. Call this function each time you need a fresh graph
    instance (e.g. per-request or per-test).

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("advisor", _placeholder_advisor)
    workflow.add_node("analyst", _placeholder_analyst)
    workflow.add_node("client", _placeholder_client)

    # Wire edges - Phase 1: START -> advisor -> END
    workflow.add_edge(START, "advisor")
    workflow.add_conditional_edges(
        "advisor", _should_continue, {END: END}
    )

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def make_initial_state(user_message: str) -> AgentState:
    """Create a zeroed AgentState for a new conversation thread.

    Args:
        user_message: The initial user query to seed the
            conversation.

    Returns:
        A fully initialised AgentState dict with the user
        message as the first entry in messages.
    """
    return AgentState(
        messages=[HumanMessage(content=user_message)],
        current_agent="",
        iteration_count=0,
        state_version=0,
        created_by="",
        executed_agents=[],
        client_profile={},
        audit_request={},
        research_results={},
        esg_report={},
    )
