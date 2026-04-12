"""
Module: yfinance_tools
Purpose: yfinance ESG sustainability data retrieval (fallback).
SR 11-7 Relevance: Pillar 2 (Validation) — fallback ESG data
    source with explicit freshness warnings for auditors.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import yfinance
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from esg_auditor.core.exceptions import DataFetchError

_FRESHNESS_WARNING = (
    "\n[Source: yfinance/Sustainalytics "
    "— verify data freshness before use]"
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    reraise=True,
)
def _get_yfinance_sustainability(
    ticker: str,
) -> dict:
    """Fetch sustainability data from yfinance.

    Retries up to 3 times with exponential backoff to
    handle Yahoo Finance rate limits.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict of sustainability metrics.

    Raises:
        DataFetchError: If sustainability data is None,
            empty, or any exception occurs.
    """
    try:
        tk = yfinance.Ticker(ticker)
        sustainability = tk.sustainability
    except Exception as exc:
        raise DataFetchError(
            f"yfinance error for {ticker}: {exc}"
        ) from exc

    if sustainability is None or sustainability.empty:
        raise DataFetchError(
            f"No sustainability data for {ticker}"
        )

    return sustainability.to_dict()


@tool
def get_yfinance_esg_score(ticker: str) -> str:
    """Get ESG sustainability scores from yfinance.

    Fallback data source when Finnhub is unavailable.
    Includes a freshness warning per SR 11-7 guidelines.

    Args:
        ticker: Stock ticker symbol, e.g. "MSFT".

    Returns:
        Formatted sustainability metrics with freshness
        warning, or ERROR string.
    """
    try:
        data = _get_yfinance_sustainability(ticker)
    except DataFetchError as exc:
        return (
            f"ERROR: yfinance ESG unavailable — {exc}"
        )

    lines: list[str] = [
        f"yfinance Sustainability for {ticker}:"
    ]
    for key, values in list(data.items())[:10]:
        if isinstance(values, dict):
            val = next(iter(values.values()), "N/A")
        else:
            val = values
        lines.append(f"  {key}: {val}")

    lines.append(_FRESHNESS_WARNING)
    return "\n".join(lines)
