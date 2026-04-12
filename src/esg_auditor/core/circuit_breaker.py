"""
Module: circuit_breaker
Purpose: Abstract base class and concrete circuit breaker implementation.
SR 11-7 Relevance: Pillar 3 (Governance) — circuit state transitions
    are logged via the observability layer so auditors can see when
    an agent started degrading.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import abc
import enum
import logging
import time
from collections.abc import Callable
from typing import Any

from esg_auditor.core.exceptions import CircuitBreakerError

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerBase(abc.ABC):
    """Abstract contract for all agent-level circuit breakers.

    Every agent call that touches an external service (LLM, API,
    vector store) must be wrapped in a subclass of this base.

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


class CircuitBreaker(CircuitBreakerBase):
    """Concrete circuit breaker with CLOSED/OPEN/HALF_OPEN state machine.

    SR 11-7 Pillar 3: every state transition is logged so auditors
    can see when an agent started degrading and when it recovered.

    Args:
        name: Human-readable name for logging.
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds to wait before testing recovery.
        half_open_max_calls: Max calls allowed in HALF_OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
    ) -> None:
        """Initialise with configurable thresholds.

        Args:
            name: Circuit breaker identifier.
            failure_threshold: Failures before opening circuit.
            recovery_timeout: Seconds before half-open probe.
            half_open_max_calls: Max calls in half-open state.
        """
        super().__init__(name)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._opened_at: float | None = None
        self._half_open_calls: int = 0

    def call(
        self,
        func: Callable[..., Any],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Execute func through the circuit breaker.

        Args:
            func: The callable to protect.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            The result of func(*args, **kwargs).

        Raises:
            CircuitBreakerError: If circuit is OPEN (fail-fast)
                or HALF_OPEN with max probe calls reached.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is OPEN "
                    f"— failing fast. Retry after "
                    f"{self.recovery_timeout}s."
                )

        if self.state == CircuitState.HALF_OPEN:
            if (
                self._half_open_calls
                >= self.half_open_max_calls
            ):
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is HALF_OPEN "
                    f"— max probe calls reached."
                )
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Reset circuit to CLOSED on successful call."""
        if self.state != CircuitState.CLOSED:
            logger.info(
                "Circuit '%s' CLOSED after recovery.",
                self.name,
            )
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._half_open_calls = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        """Increment failure count and open if threshold hit."""
        self.failure_count += 1
        logger.warning(
            "Circuit '%s' failure %d/%d.",
            self.name,
            self.failure_count,
            self.failure_threshold,
        )
        if self.failure_count >= self.failure_threshold:
            self._open_circuit()

    def _open_circuit(self) -> None:
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        logger.error(
            "Circuit '%s' OPENED after %d failures.",
            self.name,
            self.failure_count,
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN for probe."""
        self.state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        logger.info(
            "Circuit '%s' → HALF_OPEN "
            "(testing recovery).",
            self.name,
        )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._opened_at is None:
            return False
        return (
            time.monotonic() - self._opened_at
        ) >= self.recovery_timeout

    def compensate(self) -> None:
        """Saga rollback hook — reset circuit on compensation.

        Called by the orchestrator when rolling back a failed
        workflow. Closing the circuit allows the next workflow
        attempt to try again.
        """
        logger.info(
            "Circuit '%s' compensate() called "
            "— resetting to CLOSED.",
            self.name,
        )
        self._on_success()
