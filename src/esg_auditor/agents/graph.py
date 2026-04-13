"""
Module: graph
Purpose: LangGraph StateGraph assembly for the multi-agent ESG auditor.
SR 11-7 Relevance: Pillar 1 (Development) + Pillar 3 (Governance) —
    the graph definition is the executable specification of the agent
    workflow. Every node increments state_version and records
    created_by to establish an immutable audit lineage.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import logging
import os
from functools import partial

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool as tool_decorator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from esg_auditor.agents.analyst_agent import run_analyst
from esg_auditor.agents.client_agent import (
    generate_client_profile,
)
from esg_auditor.agents.prompts import ADVISOR_SYSTEM_PROMPT
from esg_auditor.config import Settings, get_settings
from esg_auditor.core.state import AgentState

logger = logging.getLogger(__name__)


def _get_advisor_tools(settings: Settings) -> list:
    """Construct @tool wrappers with settings injected.

    Uses closures to inject settings into the Analyst and
    Client agent calls so the Advisor graph does not need
    module-level singletons.

    Args:
        settings: Application settings for LLM credentials.

    Returns:
        List of two tool-decorated functions.
    """

    @tool_decorator
    def consult_analyst(research_query: str) -> str:
        """Delegate ESG research to the Analyst agent.

        The Analyst runs six tools: Finnhub/yfinance ESG
        scores, Marketaux news sentiment, FinBERT NLP
        classification, SEC EDGAR filings, and Qdrant
        regulatory knowledge base. Always call this tool
        for every ESG audit request — never answer from
        your own knowledge.

        Args:
            research_query: Specific research question
                including company name, ticker, and ESG
                dimensions to investigate.

        Returns:
            Formatted analyst findings with source
            citations.
        """
        result = run_analyst(research_query, settings)
        return f"[ANALYST AGENT RESULT]\n{result}"

    @tool_decorator
    def get_client_profile(
        client_description: str,
    ) -> str:
        """Generate a structured investor profile via the Client agent.

        Must be called before any ESG suitability
        assessment. The Client agent produces a validated
        ClientProfile including risk tolerance, assets,
        holdings, investment horizon, and ESG preference.

        Args:
            client_description: Context about the client
                or their query. If unknown, pass a default
                institutional investor description.

        Returns:
            JSON-formatted ClientProfile with suitability
            parameters.
        """
        result = generate_client_profile(
            client_description, settings
        )
        return f"[CLIENT AGENT RESULT]\n{result}"

    return [consult_analyst, get_client_profile]


def _build_advisor_model(
    settings: Settings, tools: list
) -> ChatAnthropic:
    """Construct and bind the advisor LLM.

    Called once inside build_graph(). No module-level LLM.

    Args:
        settings: Application settings with LLM credentials.
        tools: List of tools to bind to the model.

    Returns:
        ChatAnthropic with tools bound.
    """
    llm = ChatAnthropic(
        model=settings.default_model,
        max_tokens=settings.max_tokens,
        temperature=0,
        api_key=settings.anthropic_api_key,
    )
    return llm.bind_tools(tools)


def _advisor_node(
    state: AgentState,
    *,
    advisor_model: ChatAnthropic,
    max_iterations: int,
) -> dict:
    """Advisor node — the only LLM-backed node in the graph.

    Checks iteration guard first, then invokes the advisor
    model with the full message history.

    SR 11-7 Pillar 3: increments state_version and records
    created_by on every transition.

    Args:
        state: Current graph state.
        advisor_model: Bound ChatAnthropic instance.
        max_iterations: Maximum allowed iterations.

    Returns:
        State update dict.
    """
    current_iter = state["iteration_count"]

    if current_iter >= max_iterations:
        logger.warning(
            "Advisor reached iteration limit (%d). "
            "Returning safe shutdown.",
            max_iterations,
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Maximum analysis iterations "
                        "reached. Please review partial "
                        "results above. Human review "
                        "recommended before any action."
                    )
                )
            ],
            "current_agent": "advisor",
            "state_version": state["state_version"] + 1,
            "created_by": "advisor",
            "executed_agents": (
                state["executed_agents"] + ["advisor"]
            ),
            "iteration_count": current_iter + 1,
        }

    # On the first pass, append a compliance reminder
    # so the LLM calls both tools before answering.
    system_content = ADVISOR_SYSTEM_PROMPT
    if current_iter == 0:
        system_content += (
            "\n\nREMINDER FOR THIS REQUEST: You have "
            "just received a new audit request. Before "
            "writing any response, you must call "
            "get_client_profile AND consult_analyst. "
            "Do not write a final response until both "
            "tools have returned their results."
        )

    messages = [
        SystemMessage(content=system_content),
        *state["messages"],
    ]

    response = advisor_model.invoke(messages)

    # Determine which sub-agents were delegated to this
    # pass by inspecting tool_calls on the response. This
    # ensures the audit trail reflects all three agents
    # when they contribute, satisfying SR 11-7 traceability.
    sub_agents_called: list[str] = []
    if (
        hasattr(response, "tool_calls")
        and response.tool_calls
    ):
        for tc in response.tool_calls:
            if tc["name"] == "consult_analyst":
                sub_agents_called.append("analyst")
            elif tc["name"] == "get_client_profile":
                sub_agents_called.append("client")

    return {
        "messages": [response],
        "current_agent": "advisor",
        "state_version": state["state_version"] + 1,
        "created_by": "advisor",
        "executed_agents": (
            state["executed_agents"]
            + ["advisor"]
            + sub_agents_called
        ),
        "iteration_count": current_iter + 1,
    }


def _should_continue(
    state: AgentState,
    *,
    max_iterations: int,
) -> str:
    """Route based on tool calls in last message.

    Returns "tools" if the last message has tool_calls,
    END otherwise. Also returns END if iteration limit
    is reached.

    Args:
        state: Current graph state.
        max_iterations: Maximum allowed iterations.

    Returns:
        "tools" or END.
    """
    if state["iteration_count"] >= max_iterations:
        return END

    last_message = state["messages"][-1]
    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"
    return END


def _compensate_workflow(state: AgentState) -> None:
    """Walk executed_agents in reverse for saga rollback.

    Phase 3 stub: logs intent for each agent. Full
    side-effect rollback will be added per agent as
    they gain stateful operations.

    Args:
        state: Current graph state with executed_agents.
    """
    agents = state.get("executed_agents", [])
    for agent_name in reversed(agents):
        logger.info(
            "Saga compensation: rolling back '%s'.",
            agent_name,
        )


def build_graph(
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Assemble and compile the full LangGraph StateGraph.

    Factory function — constructs LLM, tools, and graph
    fresh each call. No module-level singletons.

    Args:
        settings: Optional settings override. Uses
            get_settings() if None.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    if settings is None:
        settings = get_settings()

    advisor_tools = _get_advisor_tools(settings)
    advisor_model = _build_advisor_model(
        settings, advisor_tools
    )
    max_iterations = settings.max_agent_iterations

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node(
        "advisor",
        partial(
            _advisor_node,
            advisor_model=advisor_model,
            max_iterations=max_iterations,
        ),
    )
    workflow.add_node("tools", ToolNode(advisor_tools))

    # Wire edges
    workflow.add_edge(START, "advisor")
    workflow.add_conditional_edges(
        "advisor",
        partial(
            _should_continue,
            max_iterations=max_iterations,
        ),
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "advisor")

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def make_initial_state(user_message: str) -> AgentState:
    """Create a zeroed AgentState for a new conversation.

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


def configure_tracing(settings: Settings) -> None:
    """Configure LangSmith tracing from settings.

    Called ONCE at application startup from app.py.
    This is the ONLY function that touches os.environ
    in the agents package.

    Args:
        settings: Application settings with tracing config.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = (
        settings.langchain_tracing_v2
    )
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = (
            settings.langsmith_api_key
        )
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = (
            settings.langsmith_project
        )
    logger.info(
        "LangSmith tracing configured: %s",
        settings.langchain_tracing_v2,
    )
