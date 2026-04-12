"""
Module: marketaux
Purpose: Marketaux API integration for ESG news retrieval.
SR 11-7 Relevance: Pillar 2 (Validation) — all news sources
    are logged with entity sentiment for data provenance.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import requests
from langchain_core.tools import tool
from ratelimit import limits, sleep_and_retry

from esg_auditor.config import get_settings
from esg_auditor.core.exceptions import DataFetchError

_ONE_DAY_SECONDS = 86_400
_ESG_QUERY = (
    "ESG OR sustainability OR environmental "
    "OR governance OR social OR emissions"
)


@sleep_and_retry
@limits(calls=100, period=_ONE_DAY_SECONDS)
def _fetch_marketaux_news(
    ticker: str,
    query: str,
    api_key: str,
) -> dict:
    """Call Marketaux API with rate limiting.

    Args:
        ticker: Stock ticker symbol.
        query: Search query string.
        api_key: Marketaux API token from Settings.

    Returns:
        Raw JSON response dict.

    Raises:
        DataFetchError: On HTTP error or timeout.
    """
    params = {
        "api_token": api_key,
        "symbols": ticker,
        "search": query,
        "filter_entities": "true",
        "language": "en",
        "limit": 10,
    }
    try:
        resp = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        raise DataFetchError(
            f"Marketaux HTTP error: {exc}"
        ) from exc
    except requests.Timeout as exc:
        raise DataFetchError(
            "Marketaux request timed out"
        ) from exc


@tool
def fetch_esg_news(ticker: str) -> str:
    """Fetch recent ESG news for a ticker via Marketaux.

    Returns formatted headlines with entity sentiment scores.
    Rate limit: 100 requests/day.

    Args:
        ticker: Stock ticker symbol, e.g. "TSLA".

    Returns:
        Newline-separated headlines with sentiment, or
        ERROR string.
    """
    settings = get_settings()
    try:
        data = _fetch_marketaux_news(
            ticker, _ESG_QUERY, settings.marketaux_api_key
        )
    except DataFetchError as exc:
        return f"ERROR: Marketaux fetch failed — {exc}"

    articles = data.get("data", [])
    if not articles:
        return "No ESG news found for this ticker."

    lines: list[str] = []
    for article in articles[:10]:
        sentiment = "N/A"
        if article.get("entities"):
            sentiment = article["entities"][0].get(
                "sentiment_score", "N/A"
            )
        lines.append(
            f"- {article['title']} (sentiment: {sentiment})"
        )
    return "\n".join(lines)
