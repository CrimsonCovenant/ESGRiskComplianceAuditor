"""
Module: state
Purpose: LangGraph shared state definition for the multi-agent graph.
SR 11-7 Relevance: Pillar 1 (Development) + Pillar 3 (Governance) —
    every field is logged at each state transition via LangSmith
    tracing. state_version and created_by provide an immutable audit
    lineage; executed_agents enables saga compensation rollback.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state threaded through the LangGraph graph.

    Fields are grouped by concern:
    - Message accumulation (LangGraph reducer)
    - Orchestration control
    - Immutable snapshot audit trail (Pattern 1)
    - Saga compensation tracking (Pattern 4)
    - Domain data containers
    """

    # Message accumulator (LangGraph reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    # Orchestration
    current_agent: str
    iteration_count: int  # loop guard

    # Immutable snapshot audit trail (Pattern 1)
    state_version: int  # increments at every handoff
    created_by: str  # agent that produced current version

    # Saga compensation tracking (Pattern 4 - used in Phase 3)
    executed_agents: list[str]  # ordered list for rollback

    # Domain data
    client_profile: dict  # ClientProfile fields
    audit_request: dict  # AuditRequest fields
    research_results: dict  # Analyst agent findings
    esg_report: dict  # ESGReport fields (pre-validation)
