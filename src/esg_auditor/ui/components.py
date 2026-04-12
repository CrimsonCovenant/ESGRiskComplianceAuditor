"""
Module: ui.components
Purpose: Shared UI components and audit trail rendering.
SR 11-7 Relevance: Pillar 3 (Governance) — the audit trail is a
    compliance deliverable capturing all agent interactions.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import json
from datetime import UTC, datetime

import streamlit as st


def build_audit_entry(
    result: dict, thread_id: str
) -> dict:
    """Build an audit log entry from a completed graph result.

    SR 11-7 Pillar 3: captures the required data elements for
    each agent interaction.

    Args:
        result: Completed AgentState dict from graph.invoke().
        thread_id: LangGraph thread identifier for this session.

    Returns:
        Audit entry dict with all traceability fields populated.
    """
    return {
        "timestamp": datetime.now(
            UTC
        ).isoformat(),
        "thread_id": thread_id,
        "state_version": result.get(
            "state_version", 0
        ),
        "created_by": result.get("created_by", ""),
        "executed_agents": result.get(
            "executed_agents", []
        ),
        "iteration_count": result.get(
            "iteration_count", 0
        ),
        "current_agent": result.get(
            "current_agent", ""
        ),
    }


def render_audit_trail(
    audit_log: list[dict],
) -> None:
    """Render the Audit Trail page.

    Displays each agent interaction as an expandable entry
    with full traceability fields. Provides a JSON export
    button for compliance reporting.

    Args:
        audit_log: List of audit entry dicts from the session.
    """
    st.header("🔍 Audit Trail")
    st.caption(
        "SR 11-7 compliant record of all agent "
        "interactions in this session."
    )

    if not audit_log:
        st.info(
            "No audit interactions recorded yet. "
            "Run an audit from the Chat page first."
        )
        return

    st.metric("Total Interactions", len(audit_log))

    for entry in audit_log:
        title = (
            f"[{entry['timestamp']}] "
            f"Thread: {entry['thread_id']} "
            f"| v{entry['state_version']}"
        )
        with st.expander(title, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    "**State version:**",
                    entry["state_version"],
                )
                st.write(
                    "**Created by:**",
                    entry["created_by"],
                )
            with col2:
                st.write(
                    "**Iteration count:**",
                    entry["iteration_count"],
                )
                st.write(
                    "**Current agent:**",
                    entry["current_agent"],
                )
            st.write(
                "**Agents executed:**",
                entry["executed_agents"],
            )
            st.json(entry)

    st.divider()
    st.download_button(
        label="📥 Export Audit Log (JSON)",
        data=json.dumps(audit_log, indent=2),
        file_name="esg_audit_log.json",
        mime="application/json",
    )
