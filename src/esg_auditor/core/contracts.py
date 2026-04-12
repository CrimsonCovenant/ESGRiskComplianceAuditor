"""
Module: contracts
Purpose: Inter-agent handoff validation for data contracts.
SR 11-7 Relevance: Pillar 2 (Validation) — ensures every agent
    boundary exchange is schema-validated so malformed data never
    propagates silently through the pipeline.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from pydantic import BaseModel, ValidationError

from esg_auditor.core.exceptions import HandoffValidationError


def validate_handoff(
    data: dict,
    schema: type[BaseModel],
    producing_agent: str,
    consuming_agent: str,
) -> BaseModel:
    """Validate an inter-agent handoff against the consumer's schema.

    Args:
        data: Raw dict output from producing_agent.
        schema: Pydantic model the consuming_agent expects.
        producing_agent: Name of the agent that produced data.
        consuming_agent: Name of the agent that will consume data.

    Returns:
        Validated Pydantic model instance.

    Raises:
        HandoffValidationError: If data does not match schema.
            The error message names both agents to aid audit
            triage.
    """
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise HandoffValidationError(
            f"Handoff from '{producing_agent}' to "
            f"'{consuming_agent}' failed schema validation: "
            f"{exc}"
        ) from exc
