"""
Module: conftest
Purpose: Shared pytest fixtures for the ESG Auditor test suite.
SR 11-7 Relevance: Pillar 2 (Validation) — provides reproducible
    test configuration so validation results are deterministic.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from collections.abc import Generator

import pytest

from esg_auditor.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Clear the get_settings LRU cache before and after each test.

    Autouse ensures no test ever sees stale cached config from a
    previous test, regardless of whether it explicitly requests
    the settings fixture.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Generator[
    Settings, None, None
]:
    """Return a Settings instance with test environment variables.

    Injects minimum required env vars so that Settings can be
    constructed without a real .env file.
    """
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY", "test-key-for-unit-tests"
    )
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")

    # Clear again after monkeypatch has set env vars
    get_settings.cache_clear()

    yield get_settings()
