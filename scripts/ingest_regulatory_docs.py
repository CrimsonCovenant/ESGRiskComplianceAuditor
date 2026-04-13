#!/usr/bin/env python3
"""
Script: ingest_regulatory_docs
Purpose: One-time ingest of seed regulatory documents into Qdrant Cloud.
SR 11-7 Relevance: Pillar 1 (Development) — establishes the RAG knowledge
    base required for regulatory compliance citation in ESG audit reports.
    Run once before first production use; re-run when corpus is updated.
Usage:
    python scripts/ingest_regulatory_docs.py
    python scripts/ingest_regulatory_docs.py --dry-run
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""
from __future__ import annotations  # isort:skip_file

import argparse
import sys
from pathlib import Path

# Add src/ to path so package imports work without pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from esg_auditor.config import get_settings
from esg_auditor.rag.ingest import ingest_text
from esg_auditor.tools.qdrant_search import (
    ensure_collection,
    get_embedding_model,
    get_qdrant_client,
)


SEED_DOCUMENTS: list[dict[str, str | dict[str, str]]] = [
    {
        "text": (
            "SR 11-7: Guidance on Model Risk Management, issued by the "
            "Board of Governors of the Federal Reserve System and the "
            "Office of the Comptroller of the Currency on April 4, 2011, "
            "establishes supervisory guidance on model risk management "
            "for banking organisations. The guidance defines a model as "
            "a quantitative method, system, or approach that applies "
            "statistical, economic, financial, or mathematical theories, "
            "techniques, and assumptions to process input data into "
            "quantitative estimates. Model risk arises from the potential "
            "for adverse consequences from decisions based on incorrect "
            "or misused model outputs and reports.\n\n"
            "The guidance is organised around three pillars. Pillar 1 "
            "(Model Development and Implementation) requires that models "
            "be developed with sound theory, robust data, and rigorous "
            "testing. Documentation must include the model's purpose, "
            "design, methodology, assumptions, limitations, and "
            "intended use. Developers must maintain version control and "
            "record all changes to model code and parameters.\n\n"
            "Pillar 2 (Model Validation) mandates that an effective "
            "validation framework include evaluation of conceptual "
            "soundness, ongoing monitoring, and outcomes analysis. "
            "Validation should be performed by qualified staff who are "
            "independent of model development and business line use. "
            "Validation activities include sensitivity analysis, "
            "benchmarking against alternative models, and back-testing "
            "against realised outcomes.\n\n"
            "Pillar 3 (Model Governance, Policies, and Controls) "
            "requires that institutions establish governance frameworks "
            "with clear roles and responsibilities for model risk "
            "management. Boards of directors should ensure adequate "
            "resources for model risk management and set limits on model "
            "usage. A comprehensive model inventory must be maintained, "
            "and internal audit should assess the effectiveness of the "
            "overall model risk management framework. The guidance "
            "applies to all models used for decision-making, financial "
            "reporting, and regulatory compliance, including AI and "
            "machine learning systems that meet the definition of a model."
        ),
        "metadata": {
            "source": "SR-11-7-FRB-2011",
            "document_type": "SR-Letter",
            "company": "",
            "ticker": "",
            "date": "2011-04-04",
            "jurisdiction": "US-Federal",
            "section": "Full Guidance Summary",
            "chunk_index": 0,
        },
    },
    {
        "text": (
            "The Task Force on Climate-related Financial Disclosures "
            "(TCFD) published its final recommendations on June 29, 2017, "
            "providing a framework for companies to disclose climate-related "
            "risks and opportunities. The recommendations are structured "
            "around four thematic areas that represent core elements of "
            "how organisations operate.\n\n"
            "Governance: Disclose the organisation's governance around "
            "climate-related risks and opportunities. This includes the "
            "board's oversight of climate-related risks and the role of "
            "management in assessing and managing those risks. Companies "
            "should describe the board committees responsible for climate "
            "oversight and the frequency with which they are informed.\n\n"
            "Strategy: Disclose the actual and potential impacts of "
            "climate-related risks and opportunities on the organisation's "
            "businesses, strategy, and financial planning where such "
            "information is material. Organisations should describe "
            "climate-related risks over the short, medium, and long term "
            "and the resilience of their strategy under different climate "
            "scenarios, including a 2°C or lower scenario.\n\n"
            "Risk Management: Disclose how the organisation identifies, "
            "assesses, and manages climate-related risks. Companies should "
            "describe processes for identifying and assessing climate risks, "
            "how these processes are integrated into overall risk management, "
            "and the criteria used to determine materiality of climate risks.\n\n"
            "Metrics and Targets: Disclose the metrics and targets used to "
            "assess and manage relevant climate-related risks and "
            "opportunities where such information is material. Organisations "
            "should disclose Scope 1, Scope 2, and, if appropriate, "
            "Scope 3 greenhouse gas emissions, along with related risks. "
            "Cross-industry climate-related metrics include transition risks, "
            "physical risks, climate-related opportunities, capital "
            "deployment, and internal carbon prices."
        ),
        "metadata": {
            "source": "TCFD-Recommendations-2017",
            "document_type": "TCFD",
            "company": "",
            "ticker": "",
            "date": "2017-06-29",
            "jurisdiction": "Global",
            "section": "Recommendations Summary",
            "chunk_index": 0,
        },
    },
    {
        "text": (
            "On March 6, 2024, the U.S. Securities and Exchange Commission "
            "adopted the final rule on climate-related disclosures for "
            "public companies (Release No. 33-11275). The rule requires "
            "registrants to disclose material climate-related risks, "
            "governance structures, risk management processes, and "
            "greenhouse gas emissions in registration statements and "
            "annual reports.\n\n"
            "Large accelerated filers are required to disclose Scope 1 "
            "and Scope 2 greenhouse gas emissions when material. These "
            "emissions must be reported in gross terms, without the use "
            "of purchased offsets. Large accelerated filers must obtain "
            "limited assurance for the reported emissions data beginning "
            "in fiscal years starting 2029, transitioning to reasonable "
            "assurance in fiscal years starting 2033.\n\n"
            "All registrants must describe their governance of climate "
            "risks, including board oversight and management's role. "
            "Companies must also describe material climate-related risks "
            "including physical risks (acute and chronic) and transition "
            "risks (regulatory, technology, market, reputational). If a "
            "company has adopted a transition plan, the material aspects "
            "of that plan must be disclosed.\n\n"
            "Financial statement effects of severe weather events and "
            "other natural conditions must be disclosed in a note to the "
            "financial statements when the aggregate impact exceeds one "
            "percent of the relevant financial statement line item. The "
            "rule provides a safe harbour for forward-looking statements "
            "related to transition plans, scenario analysis, internal "
            "carbon prices, and climate-related targets and goals."
        ),
        "metadata": {
            "source": "SEC-ClimateRule-2024",
            "document_type": "SEC-Rule",
            "company": "",
            "ticker": "",
            "date": "2024-03-06",
            "jurisdiction": "US-Federal",
            "section": "Final Rule Summary",
            "chunk_index": 0,
        },
    },
    {
        "text": (
            "FINRA's 2026 Annual Regulatory Oversight Report includes a "
            "dedicated section on generative AI and agentic AI systems in "
            "broker-dealer operations. FINRA observes that member firms are "
            "increasingly deploying large language model-based systems for "
            "customer communication drafting, compliance monitoring, "
            "research report generation, and portfolio recommendation "
            "support.\n\n"
            "FINRA emphasises that firms must maintain supervisory systems "
            "reasonably designed to achieve compliance with applicable "
            "securities laws when deploying AI-generated communications. "
            "Key requirements include: (1) all AI-generated customer "
            "communications must be reviewed and approved by a registered "
            "principal before distribution, (2) firms must log all prompts, "
            "model inputs, and model outputs for examination purposes, "
            "(3) model versioning must be maintained so that any output "
            "can be reproduced using the same model version and inputs.\n\n"
            "For agentic AI systems that take autonomous actions, FINRA "
            "highlights heightened supervisory obligations. Firms must "
            "implement circuit breakers or kill switches that halt "
            "autonomous agent operations when predefined thresholds are "
            "breached. Human-in-the-loop oversight is required for any "
            "agent action that could result in a trade execution, account "
            "modification, or fund transfer. Firms should conduct regular "
            "testing of agent decision paths and document the reasoning "
            "chain for each material recommendation."
        ),
        "metadata": {
            "source": "FINRA-2026-RAOR-GenAI",
            "document_type": "FINRA-Guidance",
            "company": "",
            "ticker": "",
            "date": "2026-01-15",
            "jurisdiction": "US-Federal",
            "section": "GenAI Oversight Section",
            "chunk_index": 0,
        },
    },
    {
        "text": (
            "The Corporate Sustainability Reporting Directive (CSRD), "
            "adopted by the European Parliament and Council on December 14, "
            "2022 (Directive 2022/2464), significantly expands the scope "
            "and depth of sustainability reporting obligations for "
            "companies operating in the European Union.\n\n"
            "The CSRD introduces the principle of double materiality, "
            "requiring companies to report on both how sustainability "
            "matters affect the company (financial materiality or "
            "outside-in perspective) and how the company impacts people "
            "and the environment (impact materiality or inside-out "
            "perspective). Both dimensions must be assessed and reported.\n\n"
            "Reporting is governed by the European Sustainability "
            "Reporting Standards (ESRS), developed by EFRAG. The ESRS "
            "cover cross-cutting standards (ESRS 1 and ESRS 2) and "
            "topical standards spanning Environmental (E1-E5), Social "
            "(S1-S4), and Governance (G1) dimensions. Key environmental "
            "topics include climate change, pollution, water and marine "
            "resources, biodiversity, and circular economy. Social topics "
            "cover own workforce, workers in the value chain, affected "
            "communities, and consumers.\n\n"
            "The CSRD applies to large EU companies (meeting two of three "
            "criteria: 250+ employees, EUR 50M+ net turnover, EUR 25M+ "
            "total assets), listed SMEs (with transitional provisions), "
            "and non-EU companies with substantial EU activity (EUR 150M+ "
            "net turnover in the EU). Sustainability reports must be "
            "included in the management report, published in machine-"
            "readable XHTML format, and subject to limited assurance by "
            "an independent auditor. First reports are due in 2025 for "
            "the largest companies (those already subject to NFRD), "
            "with phased implementation through 2028 for other entities."
        ),
        "metadata": {
            "source": "EU-CSRD-2022",
            "document_type": "CSRD",
            "company": "",
            "ticker": "",
            "date": "2022-12-14",
            "jurisdiction": "EU",
            "section": "Directive Summary",
            "chunk_index": 0,
        },
    },
]


def main(dry_run: bool = False) -> None:
    """Ingest seed regulatory documents into Qdrant.

    Args:
        dry_run: If True, validate metadata only without
            writing to Qdrant.
    """
    settings = get_settings()
    client = get_qdrant_client(
        settings.qdrant_url, settings.qdrant_api_key
    )
    embedding_model = get_embedding_model()

    if not dry_run:
        ensure_collection(client, settings.collection_name)

    total_chunks = 0
    for doc in SEED_DOCUMENTS:
        text: str = doc["text"]  # type: ignore[assignment]
        metadata: dict[str, str | int] = doc["metadata"]  # type: ignore[assignment]
        source = metadata.get("source", "unknown")

        if dry_run:
            print(f"[DRY RUN] Validated: {source}")
            continue

        count = ingest_text(
            text,
            metadata,
            client,
            embedding_model,
            settings.collection_name,
        )
        total_chunks += count
        print(f"✓ Ingested: {source} ({count} chunks)")

    if dry_run:
        print(
            f"\nDry run complete — {len(SEED_DOCUMENTS)} "
            "documents validated."
        )
    else:
        print(
            f"\nTotal: {total_chunks} chunks ingested "
            f"into {settings.collection_name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest seed regulatory documents "
        "into Qdrant."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate documents without writing to Qdrant.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
