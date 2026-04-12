"""
Module: app
Purpose: Streamlit application entry point for the ESG Portfolio Auditor.
SR 11-7 Relevance: Pillar 3 (Governance) — configures LangSmith tracing
    at startup before any agent interaction occurs, ensuring the full
    session is captured in the audit trail.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import streamlit as st

from esg_auditor.agents.graph import configure_tracing
from esg_auditor.config import get_settings
from esg_auditor.ui.chat import render_chat
from esg_auditor.ui.components import render_audit_trail
from esg_auditor.ui.dashboard import render_dashboard

settings = get_settings()
configure_tracing(settings)  # Must be first — before any st.* calls

st.set_page_config(
    page_title="ESG Portfolio Auditor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.sidebar.radio(
    "Navigation",
    ["💬 Chat", "📊 Dashboard", "🔍 Audit Trail"],
    index=0,
)

if page == "💬 Chat":
    render_chat()
elif page == "📊 Dashboard":
    render_dashboard(
        st.session_state.get("last_report", {})
    )
elif page == "🔍 Audit Trail":
    render_audit_trail(
        st.session_state.get("audit_log", [])
    )
