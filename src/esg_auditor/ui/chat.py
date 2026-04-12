"""
Module: ui.chat
Purpose: Streamlit chat interface for multi-agent ESG audit conversations.
SR 11-7 Relevance: Pillar 3 (Governance) — every interaction appended to
    session audit_log for the Audit Trail page.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import uuid

import streamlit as st

from esg_auditor.agents.graph import (
    build_graph,
    make_initial_state,
)
from esg_auditor.config import get_settings
from esg_auditor.ui.components import build_audit_entry


def _extract_content(content: str | list) -> str:
    """Extract plain text from Claude message content.

    Claude can return content as a plain string or as a
    list of content blocks (e.g.
    [{"type": "text", "text": "..."}]). This function
    normalises both to a plain string.

    Args:
        content: Message content from AIMessage.

    Returns:
        Plain text string.
    """
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )
    return str(content)


def render_chat() -> None:
    """Render the ESG Audit Chat page.

    Initialises the LangGraph graph and session state on
    first render. Appends every interaction to
    st.session_state.audit_log for the Audit Trail page.
    """
    st.header("💬 ESG Audit Chat")
    st.caption(
        "Enter a company name and ticker to run an "
        "ESG compliance audit. "
        "Example: *Audit Tesla (TSLA) for ESG compliance*"
    )

    settings = get_settings()

    # Initialise session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(settings)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the ESG Portfolio "
                    "Auditor. Enter a company name and "
                    "ticker symbol to begin an audit. "
                    "\n\n⚠️ *All output is advisory only "
                    "and does not constitute investment "
                    "advice. Human review is required "
                    "before any action.*"
                ),
            }
        ]
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new input
    if prompt := st.chat_input(
        "e.g., Audit Microsoft (MSFT) for ESG compliance"
    ):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                "Running multi-agent ESG audit..."
            ):
                thread_id = (
                    f"session-"
                    f"{st.session_state.session_id}"
                )
                result = (
                    st.session_state.graph.invoke(
                        make_initial_state(prompt),
                        config={
                            "configurable": {
                                "thread_id": thread_id
                            }
                        },
                    )
                )

            response = _extract_content(
                result["messages"][-1].content
            )
            st.markdown(response)

            # Agent trace expander
            with st.expander(
                "🔍 Agent trace", expanded=False
            ):
                st.write(
                    "**State version:**",
                    result.get("state_version", 0),
                )
                st.write(
                    "**Agents executed:**",
                    result.get("executed_agents", []),
                )
                st.write(
                    "**Iterations:**",
                    result.get("iteration_count", 0),
                )

        # Update session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        st.session_state.last_report = result.get(
            "esg_report", {}
        )
        st.session_state.audit_log.append(
            build_audit_entry(result, thread_id)
        )
