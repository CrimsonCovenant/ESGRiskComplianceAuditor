"""
Module: config
Purpose: Application configuration loaded from environment variables.
SR 11-7 Relevance: Pillar 1 (Development) — centralises all runtime
    parameters so model behaviour is reproducible and auditable.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the ESG Auditor.

    All values are loaded from environment variables or a .env file.
    Required fields have no default and will raise ValidationError
    if absent.
    """

    # LLM
    anthropic_api_key: str
    default_model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096

    # Vector store
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "esg_regulatory_docs"

    # Financial data APIs
    marketaux_api_key: str = ""
    finnhub_api_key: str = ""

    # Observability / SR 11-7 Pillar 3
    langsmith_api_key: str = ""
    langsmith_project: str = "esg-auditor"
    langchain_tracing_v2: str = "false"

    # Application behaviour
    environment: str = "development"
    log_level: str = "INFO"
    max_agent_iterations: int = 10
    model_risk_tier: str = "Tier_2"

    # Circuit breaker (Phase 3 uses these thresholds)
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_half_open_max_calls: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Use this factory everywhere instead of a module-level singleton so
    tests can clear the cache and inject different config values.
    """
    return Settings()
