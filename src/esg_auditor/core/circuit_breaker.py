"""
Module: circuit_breaker
Purpose: Abstract base class for agent-level circuit breakers.
SR 11-7 Relevance: Pillar 3 (Governance) — circuit state transitions
    are logged via the observability layer so auditors can see when
    an agent started degrading.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

import abc
import enum


class CircuitState(enum.Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerBase(abc.ABC):
    """Abstract contract for all agent-level circuit breakers.

    Concrete implementations are built in Phase 3. Every agent call
    that touches an external service (LLM, API, vector store) must
    be wrapped in a subclass of this base.

    SR 11-7 Pillar 3: state transitions must be logged via the
    observability layer so auditors can see when an agent started
    degrading.
    """

    def __init__(self, name: str) -> None:
        """Initialise the circuit breaker in CLOSED state.

        Args:
            name: Human-readable identifier for this breaker
                (e.g. "advisor-llm", "qdrant-search").
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count: int = 0

    @abc.abstractmethod
    def call(self, *args: object, **kwargs: object) -> object:
        """Execute the protected call."""

    @abc.abstractmethod
    def _on_success(self) -> None:
        """Handle a successful call — reset failure count."""

    @abc.abstractmethod
    def _on_failure(self) -> None:
        """Handle a failed call — increment count, trip if needed."""

    @abc.abstractmethod
    def compensate(self) -> None:
        """Saga rollback hook — undo side effects of prior calls."""
