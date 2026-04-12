"""
Module: test_schemas
Purpose: Unit tests for Pydantic v2 data models in core.schemas.
SR 11-7 Relevance: Pillar 2 (Validation) — verifies that input/output
    contracts reject invalid data and normalise fields correctly.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-11
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from esg_auditor.core.schemas import (
    AuditRequest,
    ClientProfile,
    ESGPillarScore,
    ESGReport,
    RiskLevel,
)


class TestESGReport:
    """Tests for the ESGReport schema."""

    def test_valid_instantiation(self) -> None:
        """A fully valid ESGReport should instantiate without error."""
        report = ESGReport(
            company_name="Apple Inc.",
            ticker="aapl",
            pillar_scores=ESGPillarScore(
                environmental=75.0,
                social=80.0,
                governance=85.0,
            ),
            overall_score=80.0,
            risk_level=RiskLevel.LOW,
            key_findings=["Strong governance framework"],
            sentiment_summary="Positive overall sentiment",
            sources=["SEC 10-K 2024"],
        )
        assert report.company_name == "Apple Inc."
        assert report.ticker == "AAPL"
        assert report.overall_score == 80.0
        assert isinstance(report.generated_at, datetime)

    def test_ticker_uppercasing(self) -> None:
        """Ticker field should be uppercased and stripped."""
        report = ESGReport(
            company_name="Tesla Inc.",
            ticker="  tsla  ",
            pillar_scores=ESGPillarScore(
                environmental=60.0,
                social=55.0,
                governance=70.0,
            ),
            overall_score=61.7,
            risk_level=RiskLevel.MEDIUM,
            key_findings=["Emissions targets on track"],
            sentiment_summary="Mixed sentiment",
        )
        assert report.ticker == "TSLA"

    def test_overall_score_out_of_range_high(self) -> None:
        """overall_score above 100 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ESGReport(
                company_name="Test Corp",
                ticker="TST",
                pillar_scores=ESGPillarScore(
                    environmental=50.0,
                    social=50.0,
                    governance=50.0,
                ),
                overall_score=150.0,
                risk_level=RiskLevel.HIGH,
                key_findings=["Finding"],
                sentiment_summary="Neutral",
            )

    def test_overall_score_out_of_range_low(self) -> None:
        """overall_score below 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ESGReport(
                company_name="Test Corp",
                ticker="TST",
                pillar_scores=ESGPillarScore(
                    environmental=50.0,
                    social=50.0,
                    governance=50.0,
                ),
                overall_score=-5.0,
                risk_level=RiskLevel.HIGH,
                key_findings=["Finding"],
                sentiment_summary="Neutral",
            )

    def test_generated_at_is_utc(self) -> None:
        """generated_at default should be timezone-aware UTC."""
        report = ESGReport(
            company_name="Test Corp",
            ticker="TST",
            pillar_scores=ESGPillarScore(
                environmental=50.0,
                social=50.0,
                governance=50.0,
            ),
            overall_score=50.0,
            risk_level=RiskLevel.LOW,
            key_findings=["Finding"],
            sentiment_summary="Neutral",
        )
        assert report.generated_at.tzinfo == UTC


class TestAuditRequest:
    """Tests for the AuditRequest schema."""

    def test_default_focus_areas(self) -> None:
        """Default focus_areas should include all three pillars."""
        request = AuditRequest(
            company_name="Apple Inc.",
            ticker="AAPL",
        )
        assert request.focus_areas == [
            "environmental",
            "social",
            "governance",
        ]

    def test_custom_focus_areas(self) -> None:
        """Custom focus_areas should override defaults."""
        request = AuditRequest(
            company_name="Apple Inc.",
            ticker="AAPL",
            focus_areas=["environmental"],
        )
        assert request.focus_areas == ["environmental"]

    def test_default_flags(self) -> None:
        """include_sentiment and include_regulatory default True."""
        request = AuditRequest(
            company_name="Test",
            ticker="TST",
        )
        assert request.include_sentiment is True
        assert request.include_regulatory is True


class TestClientProfile:
    """Tests for the ClientProfile schema."""

    def test_valid_profile(self) -> None:
        """A valid ClientProfile should instantiate without error."""
        profile = ClientProfile(
            client_id="CLT-001",
            age=35,
            risk_tolerance=RiskLevel.MEDIUM,
            total_assets_usd=500_000.0,
            current_holdings=["AAPL", "MSFT"],
            investment_horizon_years=10,
        )
        assert profile.client_id == "CLT-001"
        assert profile.age == 35

    def test_age_below_minimum(self) -> None:
        """Age below 18 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ClientProfile(
                client_id="CLT-002",
                age=17,
                risk_tolerance=RiskLevel.LOW,
                total_assets_usd=100_000.0,
                current_holdings=[],
                investment_horizon_years=5,
            )

    def test_age_above_maximum(self) -> None:
        """Age above 100 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ClientProfile(
                client_id="CLT-003",
                age=101,
                risk_tolerance=RiskLevel.HIGH,
                total_assets_usd=1_000_000.0,
                current_holdings=["GOOG"],
                investment_horizon_years=3,
            )

    def test_age_boundary_minimum(self) -> None:
        """Age exactly 18 should be valid."""
        profile = ClientProfile(
            client_id="CLT-004",
            age=18,
            risk_tolerance=RiskLevel.LOW,
            total_assets_usd=10_000.0,
            current_holdings=[],
            investment_horizon_years=30,
        )
        assert profile.age == 18

    def test_age_boundary_maximum(self) -> None:
        """Age exactly 100 should be valid."""
        profile = ClientProfile(
            client_id="CLT-005",
            age=100,
            risk_tolerance=RiskLevel.LOW,
            total_assets_usd=2_000_000.0,
            current_holdings=["BND"],
            investment_horizon_years=1,
        )
        assert profile.age == 100
