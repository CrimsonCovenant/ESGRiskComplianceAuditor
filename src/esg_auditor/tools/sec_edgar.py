"""
Module: sec_edgar
Purpose: SEC EDGAR full-text search for ESG disclosures.
SR 11-7 Relevance: Pillar 1 (Development) — SEC filings are
    primary source documents for regulatory compliance analysis.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from datetime import date

import requests
from langchain_core.tools import tool
from ratelimit import limits, sleep_and_retry

from esg_auditor.core.exceptions import DataFetchError

_SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_USER_AGENT = "ESGAuditor research@esgauditor.app"


@sleep_and_retry
@limits(calls=10, period=1)
def _search_edgar(
    company_name: str, query: str
) -> list[dict]:
    """Query SEC EDGAR full-text search.

    Args:
        company_name: Legal company name for the query.
        query: ESG topic query string.

    Returns:
        List of hit dicts from EDGAR response.

    Raises:
        DataFetchError: On HTTP error or timeout.
    """
    params = {
        "q": f'"{company_name}" AND ({query})',
        "forms": "10-K,10-Q",
        "dateRange": "custom",
        "startdt": "2024-01-01",
        "enddt": date.today().isoformat(),
    }
    try:
        resp = requests.get(
            _SEC_SEARCH_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("hits", {}).get("hits", [])
    except requests.HTTPError as exc:
        raise DataFetchError(
            f"SEC EDGAR HTTP error: {exc}"
        ) from exc
    except requests.Timeout as exc:
        raise DataFetchError(
            "SEC EDGAR request timed out"
        ) from exc


@tool
def search_sec_filings(
    company_name: str,
    query: str = "ESG climate risk emissions",
) -> str:
    """Search SEC EDGAR filings for ESG disclosures.

    No API key required. Returns top 5 matching filings.
    Rate limit: 10 requests/second.

    Args:
        company_name: Legal name of the company.
        query: ESG topic to search for.

    Returns:
        Formatted filing list with accession numbers, or
        ERROR string.
    """
    try:
        hits = _search_edgar(company_name, query)
    except DataFetchError as exc:
        return f"ERROR: SEC EDGAR search failed — {exc}"

    if not hits:
        return "No SEC filings found matching this query."

    lines: list[str] = []
    for hit in hits[:5]:
        src = hit.get("_source", {})
        accession = hit.get("_id", "N/A")
        lines.append(
            f"- {src.get('form_type', 'N/A')}: "
            f"{src.get('file_description', 'N/A')} "
            f"({src.get('file_date', 'N/A')}) "
            f"[accession: {accession}]"
        )
    return "\n".join(lines)
