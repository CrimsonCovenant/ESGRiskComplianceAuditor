"""
Module: schemas
Purpose: Pydantic v2 data models defining the system's input/output
    contracts.
SR 11-7 Relevance: Pillar 1 (Development) — schemas enforce typed,
    validated data boundaries between agents and ensure reproducible
    structured outputs.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    """Categorical risk classification for ESG assessments."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ESGCategory(str, Enum):
    """ESG classification category from FinBERT analysis."""

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    NONE = "none"


class ESGPillarScore(BaseModel):
    """Numeric scores for each ESG pillar (0–100 scale)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    environmental: Annotated[float, Field(ge=0.0, le=100.0)]
    social: Annotated[float, Field(ge=0.0, le=100.0)]
    governance: Annotated[float, Field(ge=0.0, le=100.0)]


class ESGReport(BaseModel):
    """Structured ESG audit report produced by the analyst agent.

    Used with with_structured_output() for reliable LLM
    extraction.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    company_name: str = Field(
        ..., description="Company being audited"
    )
    ticker: str = Field(
        ..., description="Stock ticker symbol"
    )
    pillar_scores: ESGPillarScore
    overall_score: Annotated[float, Field(ge=0.0, le=100.0)]
    risk_level: RiskLevel
    key_findings: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Top findings from the ESG audit",
    )
    regulatory_flags: list[str] = Field(default_factory=list)
    sentiment_summary: str = Field(
        ..., description="FinBERT sentiment analysis summary"
    )
    sources: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @field_validator("ticker", mode="after")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        """Normalise ticker to uppercase and strip whitespace."""
        return v.upper().strip()


class AuditRequest(BaseModel):
    """Input schema for ESG audit requests from the UI layer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    company_name: str = Field(..., min_length=1)
    ticker: str
    focus_areas: list[str] = Field(
        default_factory=lambda: [
            "environmental",
            "social",
            "governance",
        ]
    )
    include_sentiment: bool = True
    include_regulatory: bool = True


class ClientProfile(BaseModel):
    """LLM-generated investor persona for suitability analysis.

    Represents a synthetic or real client profile used by the
    client agent to assess investment suitability against ESG
    audit results.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    client_id: str
    age: int = Field(ge=18, le=100)
    risk_tolerance: RiskLevel
    total_assets_usd: float = Field(ge=0.0)
    current_holdings: list[str]
    investment_horizon_years: int = Field(ge=1, le=50)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
