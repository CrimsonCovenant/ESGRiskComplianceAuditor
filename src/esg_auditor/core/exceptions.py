"""
Module: exceptions
Purpose: Custom exception hierarchy for the ESG Auditor.
SR 11-7 Relevance: Pillar 2 (Validation) — typed exceptions allow the
    audit trail to record failure modes precisely.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""


class ESGAuditorError(Exception):
    """Base exception for all ESG Auditor errors."""


class ConfigurationError(ESGAuditorError):
    """Raised when required configuration is missing or invalid."""


class DataFetchError(ESGAuditorError):
    """Raised when a financial data API call fails."""


class EmbeddingError(ESGAuditorError):
    """Raised when document embedding or vector search fails."""


class AgentRoutingError(ESGAuditorError):
    """Raised when the supervisor cannot route to a valid agent."""


class StructuredOutputError(ESGAuditorError):
    """Raised when LLM output fails Pydantic validation."""


class HandoffValidationError(ESGAuditorError):
    """Raised when data fails schema validation at an agent boundary."""


class CircuitBreakerError(ESGAuditorError):
    """Raised when a call is rejected because the circuit is OPEN."""


class CompensationError(ESGAuditorError):
    """Raised when a compensate() rollback call itself fails."""


class StateVersionError(ESGAuditorError):
    """Raised when an agent receives an unexpected state version."""
