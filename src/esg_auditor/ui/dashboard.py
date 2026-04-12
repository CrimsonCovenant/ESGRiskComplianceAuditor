"""
Module: ui.dashboard
Purpose: Streamlit dashboard for ESG audit report visualisation.
SR 11-7 Relevance: Pillar 2 (Validation) — dashboard surfaces model
    output for human review, including confidence scores and
    regulatory flags.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import plotly.graph_objects as go
import streamlit as st


def esg_gauge(score: float, title: str) -> go.Figure:
    """Create a Plotly gauge chart for an ESG score.

    Args:
        score: ESG score value (0-100 scale).
        title: Display title for the gauge.

    Returns:
        Plotly Figure with indicator gauge.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {
                        "range": [0, 40],
                        "color": "#ff4444",
                    },
                    {
                        "range": [40, 70],
                        "color": "#ffaa00",
                    },
                    {
                        "range": [70, 100],
                        "color": "#00cc66",
                    },
                ],
            },
        )
    )
    fig.update_layout(height=250, margin={"t": 50, "b": 0, "l": 30, "r": 30})
    return fig


_RISK_COLORS: dict[str, str] = {
    "low": "#00cc66",
    "medium": "#ffaa00",
    "high": "#ff4444",
    "critical": "#990000",
}


def render_dashboard(report: dict) -> None:
    """Render the ESG Dashboard page.

    Displays ESG scores, gauge charts, key findings,
    regulatory flags, and sources from the last audit.

    Args:
        report: ESG report dict from session state.
            Empty dict when no audit has been run.
    """
    st.header("📊 ESG Dashboard")

    if not report:
        st.info(
            "No audit results yet. Run an audit from "
            "the Chat page first."
        )
        return

    # Company header
    company = report.get("company_name", "Unknown")
    ticker = report.get("ticker", "N/A")
    st.subheader(f"{company} ({ticker})")

    # Risk level badge
    risk_level = report.get("risk_level", "medium")
    risk_color = _RISK_COLORS.get(
        risk_level.lower(), "#ffaa00"
    )
    st.markdown(
        f'<span style="background-color: {risk_color}; '
        f"color: white; padding: 4px 12px; "
        f"border-radius: 4px; font-weight: bold;"
        f'">{risk_level.upper()}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Metric cards
    pillar = report.get("pillar_scores", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Overall",
            f"{report.get('overall_score', 0):.1f}",
        )
    with c2:
        st.metric(
            "Environmental",
            f"{pillar.get('environmental', 0):.1f}",
        )
    with c3:
        st.metric(
            "Social",
            f"{pillar.get('social', 0):.1f}",
        )
    with c4:
        st.metric(
            "Governance",
            f"{pillar.get('governance', 0):.1f}",
        )

    # Gauge charts
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            esg_gauge(
                pillar.get("environmental", 0),
                "Environmental",
            ),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            esg_gauge(
                pillar.get("social", 0), "Social"
            ),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            esg_gauge(
                pillar.get("governance", 0),
                "Governance",
            ),
            use_container_width=True,
        )

    # Key Findings
    with st.expander(
        "📋 Key Findings", expanded=True
    ):
        findings = report.get("key_findings", [])
        if findings:
            for finding in findings:
                st.markdown(f"- {finding}")
        else:
            st.write("No key findings recorded.")

    # Regulatory Flags
    with st.expander("⚠️ Regulatory Flags"):
        flags = report.get("regulatory_flags", [])
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.write("No regulatory flags raised.")

    # Sources
    with st.expander("📚 Sources"):
        sources = report.get("sources", [])
        if sources:
            for source in sources:
                st.markdown(f"- {source}")
        else:
            st.write("No sources recorded.")

    # Disclaimer
    st.divider()
    st.warning(
        "⚠️ This output is advisory only and does not "
        "constitute investment advice. Human review "
        "required before any action is taken."
    )
