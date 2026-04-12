"""
Module: test_circuit_breaker
Purpose: Unit tests for the CircuitBreakerBase abstract contract.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies that the circuit
    breaker interface is correctly defined and enforced.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

import pytest

from esg_auditor.core.circuit_breaker import (
    CircuitBreakerBase,
    CircuitState,
)


class _ConcreteBreaker(CircuitBreakerBase):
    """Minimal concrete subclass for testing the ABC."""

    def call(self, *args: object, **kwargs: object) -> object:
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
        """CircuitBreakerBase is abstract and cannot be created."""
        with pytest.raises(TypeError):
            CircuitBreakerBase("test")  # type: ignore[abstract]

    def test_concrete_subclass_instantiation(self) -> None:
        """A concrete subclass should instantiate successfully."""
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
        """Subclass without compensate() should raise TypeError."""

        class _Incomplete(CircuitBreakerBase):
            """Missing compensate() to test abstractness."""

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
            _Incomplete("missing-compensate")  # type: ignore[abstract]
