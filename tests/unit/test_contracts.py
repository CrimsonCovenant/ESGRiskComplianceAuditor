"""
Module: test_contracts
Purpose: Unit tests for the inter-agent handoff validator.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies that data
    contract violations are caught and reported with agent names.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

import pytest
from pydantic import BaseModel

from esg_auditor.core.contracts import validate_handoff
from esg_auditor.core.exceptions import HandoffValidationError


class _SampleSchema(BaseModel):
    """Minimal schema for testing validate_handoff."""

    name: str
    value: int


class TestValidateHandoff:
    """Tests for the validate_handoff function."""

    def test_valid_data_passes(self) -> None:
        """Valid data should return a model instance."""
        data = {"name": "test", "value": 42}
        result = validate_handoff(
            data, _SampleSchema, "producer", "consumer"
        )
        assert isinstance(result, _SampleSchema)
        assert result.name == "test"
        assert result.value == 42

    def test_invalid_data_raises_handoff_error(self) -> None:
        """Invalid data should raise HandoffValidationError."""
        data = {"name": "test", "value": "not-an-int"}
        with pytest.raises(HandoffValidationError):
            validate_handoff(
                data,
                _SampleSchema,
                "analyst",
                "advisor",
            )

    def test_error_message_contains_both_agents(self) -> None:
        """Error message should name both producing and consuming agents."""
        data = {"name": "test"}  # missing 'value'
        with pytest.raises(HandoffValidationError, match="analyst"):
            validate_handoff(
                data,
                _SampleSchema,
                "analyst",
                "advisor",
            )
        with pytest.raises(HandoffValidationError, match="advisor"):
            validate_handoff(
                data,
                _SampleSchema,
                "analyst",
                "advisor",
            )

    def test_missing_field_raises_handoff_error(self) -> None:
        """Missing required field should raise HandoffValidationError."""
        data = {"name": "only-name"}
        with pytest.raises(HandoffValidationError):
            validate_handoff(
                data,
                _SampleSchema,
                "client",
                "advisor",
            )
