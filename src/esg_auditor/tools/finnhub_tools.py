"""
Module: finnhub_tools
Purpose: Finnhub API integration for ESG score retrieval.
SR 11-7 Relevance: Pillar 2 (Validation) — ESG score provenance
    is tracked for audit trail accuracy.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import finnhub
from langchain_core.tools import tool
from ratelimit import limits, sleep_and_retry

from esg_auditor.config import get_settings
from esg_auditor.core.exceptions import DataFetchError


@sleep_and_retry
@limits(calls=60, period=60)
def _get_finnhub_esg(
    ticker: str, api_key: str
) -> dict:
    """Fetch ESG scores from Finnhub.

    Constructs a fresh Finnhub client inside the function body
    to avoid module-level singletons.

    Args:
        ticker: Stock ticker symbol.
        api_key: Finnhub API key from Settings.

    Returns:
        ESG score dict from Finnhub API.

    Raises:
        DataFetchError: On HTTP 403, empty response, or any
            Finnhub client exception.
    """
    try:
        client = finnhub.Client(api_key=api_key)
        result = client.company_esg_score(symbol=ticker)
    except finnhub.FinnhubRequestException as exc:
        if "403" in str(exc):
            raise DataFetchError(
                "Finnhub ESG requires Premium subscription"
            ) from exc
        raise DataFetchError(
            f"Finnhub request error: {exc}"
        ) from exc
    except finnhub.FinnhubAPIException as exc:
        raise DataFetchError(
            f"Finnhub API error: {exc}"
        ) from exc

    if not result:
        raise DataFetchError(
            "Finnhub returned empty ESG response"
        )
    return result


@tool
def get_finnhub_esg_score(ticker: str) -> str:
    """Get ESG scores for a ticker from Finnhub.

    Returns structured ESG score breakdown. Falls back to
    yfinance if Finnhub is unavailable.
    Rate limit: 60 requests/minute.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        Formatted ESG score breakdown, or ERROR string
        suggesting yfinance fallback.
    """
    settings = get_settings()
    try:
        data = _get_finnhub_esg(
            ticker, settings.finnhub_api_key
        )
    except DataFetchError as exc:
        return (
            "ERROR: Finnhub ESG unavailable — "
            f"use get_yfinance_esg_score as fallback. "
            f"Reason: {exc}"
        )

    lines: list[str] = [
        f"Finnhub ESG Scores for {ticker}:",
    ]
    total = data.get("totalEsg", "N/A")
    env = data.get("environmentScore", "N/A")
    social = data.get("socialScore", "N/A")
    gov = data.get("governanceScore", "N/A")
    lines.append(f"Total ESG: {total}")
    lines.append(f"Environmental: {env}")
    lines.append(f"Social: {social}")
    lines.append(f"Governance: {gov}")
    return "\n".join(lines)
