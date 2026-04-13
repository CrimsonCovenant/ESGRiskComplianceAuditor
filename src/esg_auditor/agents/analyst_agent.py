"""
Module: analyst_agent
Purpose: Analyst sub-graph for ESG research using Phase 2 tools.
SR 11-7 Relevance: Pillar 2 (Validation) — the analyst is the
    data gathering and classification node. All tool outputs are
    logged with source identifiers for audit traceability.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import logging
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from esg_auditor.agents.prompts import ANALYST_SYSTEM_PROMPT
from esg_auditor.config import Settings
from esg_auditor.core.exceptions import AgentRoutingError
from esg_auditor.tools.finbert import analyze_sentiment_esg
from esg_auditor.tools.finnhub_tools import (
    get_finnhub_esg_score,
)
from esg_auditor.tools.marketaux import fetch_esg_news
from esg_auditor.tools.qdrant_search import (
    search_regulatory_docs,
)
from esg_auditor.tools.sec_edgar import search_sec_filings
from esg_auditor.tools.yfinance_tools import (
    get_yfinance_esg_score,
)

logger = logging.getLogger(__name__)

ALL_ANALYST_TOOLS: list = [
    fetch_esg_news,
    search_sec_filings,
    get_finnhub_esg_score,
    get_yfinance_esg_score,
    analyze_sentiment_esg,
    search_regulatory_docs,
]


class _AnalystState(TypedDict):
    """Internal state for the Analyst sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages]


def run_analyst(
    research_query: str,
    settings: Settings,
) -> str:
    """Execute the Analyst ReAct loop for ESG research.

    Constructs a fresh ChatAnthropic and a nested sub-graph
    with a ReAct-style tool-calling loop. The Analyst has
    access to all Phase 2 data tools.

    Args:
        research_query: The research task from the Advisor.
        settings: Application settings with LLM credentials.

    Returns:
        The Analyst's final response as a string.

    Raises:
        AgentRoutingError: If the analyst loop exceeds max
            iterations or encounters an API error.
    """
    try:
        llm = ChatAnthropic(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            temperature=0,
            api_key=settings.anthropic_api_key,
        )
        analyst_model = llm.bind_tools(ALL_ANALYST_TOOLS)
    except Exception as exc:
        raise AgentRoutingError(
            f"Analyst LLM construction failed: {exc}"
        ) from exc

    max_iterations = settings.max_agent_iterations
    iteration_count = 0

    def _analyst_llm_node(
        state: _AnalystState,
    ) -> dict:
        """Invoke the Analyst LLM with tool bindings."""
        nonlocal iteration_count
        iteration_count += 1

        if iteration_count > max_iterations:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Analyst reached maximum "
                            "iteration limit. Returning "
                            "findings collected so far."
                        )
                    )
                ]
            }

        response = analyst_model.invoke(
            state["messages"]
        )
        return {"messages": [response]}

    def _should_continue(state: _AnalystState) -> str:
        """Route based on tool calls in last message."""
        last = state["messages"][-1]
        if (
            isinstance(last, AIMessage)
            and last.tool_calls
        ):
            if iteration_count >= max_iterations:
                return END
            return "tools"
        return END

    # Build the nested sub-graph
    workflow = StateGraph(_AnalystState)
    workflow.add_node("analyst_llm", _analyst_llm_node)
    workflow.add_node(
        "tools", ToolNode(ALL_ANALYST_TOOLS)
    )

    workflow.add_edge(START, "analyst_llm")
    workflow.add_conditional_edges(
        "analyst_llm",
        _should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "analyst_llm")

    graph = workflow.compile(
        checkpointer=InMemorySaver()
    )

    try:
        initial_state: _AnalystState = {
            "messages": [
                SystemMessage(
                    content=ANALYST_SYSTEM_PROMPT
                ),
                HumanMessage(content=research_query),
            ]
        }

        # Include parent context in thread_id for
        # LangSmith trace nesting.
        analyst_thread_id = (
            f"analyst-{abs(hash(research_query)) % 1_000_000:06d}"
        )

        result = graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": analyst_thread_id,
                }
            },
        )
        final_message = result["messages"][-1]

        if isinstance(final_message, AIMessage):
            return final_message.content or ""

        return str(final_message.content)

    except AgentRoutingError:
        raise
    except Exception as exc:
        raise AgentRoutingError(
            f"Analyst execution failed: {exc}"
        ) from exc
