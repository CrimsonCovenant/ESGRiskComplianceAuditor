"""
Module: test_circuit_breaker
Purpose: Unit tests for the circuit breaker ABC and concrete impl.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies that the circuit
    breaker interface and state machine work correctly.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import time

import pytest

from esg_auditor.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerBase,
    CircuitState,
)
from esg_auditor.core.exceptions import CircuitBreakerError


class _ConcreteBreaker(CircuitBreakerBase):
    """Minimal concrete subclass for testing the ABC."""

    def call(
        self, *args: object, **kwargs: object
    ) -> object:
        """No-op call implementation."""
        return None

    def _on_success(self) -> None:
        """No-op success handler."""

    def _on_failure(self) -> None:
        """No-op failure handler."""

    def compensate(self) -> None:
        """No-op compensation handler."""


class TestCircuitBreakerBase:
    """Tests for the CircuitBreakerBase abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """CircuitBreakerBase is abstract."""
        with pytest.raises(TypeError):
            CircuitBreakerBase("test")  # type: ignore[abstract]

    def test_concrete_subclass_instantiation(
        self,
    ) -> None:
        """A concrete subclass should instantiate."""
        breaker = _ConcreteBreaker("test-breaker")
        assert breaker.name == "test-breaker"

    def test_initial_state_is_closed(self) -> None:
        """New breakers start in CLOSED state."""
        breaker = _ConcreteBreaker("test-breaker")
        assert breaker.state == CircuitState.CLOSED

    def test_initial_failure_count_is_zero(self) -> None:
        """New breakers start with zero failures."""
        breaker = _ConcreteBreaker("test-breaker")
        assert breaker.failure_count == 0

    def test_compensate_is_abstract(self) -> None:
        """Subclass without compensate() raises TypeError."""

        class _Incomplete(CircuitBreakerBase):
            """Missing compensate()."""

            def call(
                self, *args: object, **kwargs: object
            ) -> object:
                """No-op."""
                return None

            def _on_success(self) -> None:
                """No-op."""

            def _on_failure(self) -> None:
                """No-op."""

        with pytest.raises(TypeError):
            _Incomplete("missing")  # type: ignore[abstract]


class TestCircuitBreaker:
    """Tests for the concrete CircuitBreaker."""

    def test_successful_call_returns_result(
        self,
    ) -> None:
        """Successful function should return its result."""
        cb = CircuitBreaker("test", failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_opens_after_failure_threshold(
        self,
    ) -> None:
        """Circuit should OPEN after N consecutive failures."""
        cb = CircuitBreaker("test", failure_threshold=3)

        def _failing() -> None:
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(_failing)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_open_circuit_raises_circuit_breaker_error(
        self,
    ) -> None:
        """OPEN circuit should fail fast without calling func."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=9999,
        )

        def _failing() -> None:
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(_failing)

        assert cb.state == CircuitState.OPEN

        call_count = 0

        def _should_not_be_called() -> str:
            nonlocal call_count
            call_count += 1
            return "hello"

        with pytest.raises(CircuitBreakerError):
            cb.call(_should_not_be_called)

        assert call_count == 0

    def test_circuit_transitions_to_half_open_after_timeout(
        self,
    ) -> None:
        """After timeout, circuit should transition through HALF_OPEN."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=1,
        )

        def _failing() -> None:
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(_failing)

        assert cb.state == CircuitState.OPEN

        # Simulate timeout by backdating _opened_at
        cb._opened_at = time.monotonic() - 120

        # Successful call should recover to CLOSED
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_successful_call_resets_failure_count(
        self,
    ) -> None:
        """Success after failures should reset count."""
        cb = CircuitBreaker("test", failure_threshold=5)

        def _failing() -> None:
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(_failing)

        assert cb.failure_count == 2

        cb.call(lambda: "ok")
        assert cb.failure_count == 0

    def test_compensate_resets_circuit_to_closed(
        self,
    ) -> None:
        """compensate() should reset state to CLOSED."""
        cb = CircuitBreaker("test", failure_threshold=2)

        def _failing() -> None:
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(_failing)

        assert cb.state == CircuitState.OPEN

        cb.compensate()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
