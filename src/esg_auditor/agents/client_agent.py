"""
Module: client_agent
Purpose: Client profile generation via structured LLM output.
SR 11-7 Relevance: Pillar 1 (Development) — client profile is part
    of model input documentation and suitability audit trail.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from esg_auditor.agents.prompts import CLIENT_SYSTEM_PROMPT
from esg_auditor.config import Settings
from esg_auditor.core.exceptions import (
    AgentRoutingError,
    StructuredOutputError,
)
from esg_auditor.core.schemas import ClientProfile

logger = logging.getLogger(__name__)


def generate_client_profile(
    client_description: str,
    settings: Settings,
) -> str:
    """Generate a structured investor profile via LLM.

    Constructs a fresh ChatAnthropic instance (no module-level
    singleton) and uses with_structured_output for reliable
    Pydantic extraction.

    Args:
        client_description: Natural language description of
            the client's investment profile.
        settings: Application settings with LLM credentials.

    Returns:
        JSON string of the validated ClientProfile.

    Raises:
        StructuredOutputError: If LLM output fails Pydantic
            validation.
        AgentRoutingError: On Anthropic API errors.
    """
    try:
        llm = ChatAnthropic(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            temperature=0,
            api_key=settings.anthropic_api_key,
        )
        structured_llm = llm.with_structured_output(
            ClientProfile
        )

        messages = [
            SystemMessage(content=CLIENT_SYSTEM_PROMPT),
            HumanMessage(content=client_description),
        ]

        result = structured_llm.invoke(messages)

        if isinstance(result, ClientProfile):
            logger.info(
                "Client profile generated: %s",
                result.client_id,
            )
            return result.model_dump_json()

        raise StructuredOutputError(
            "LLM did not return a valid ClientProfile"
        )

    except ValidationError as exc:
        raise StructuredOutputError(
            f"Client profile validation failed: {exc}"
        ) from exc
    except StructuredOutputError:
        raise
    except Exception as exc:
        raise AgentRoutingError(
            f"Client agent API error: {exc}"
        ) from exc
