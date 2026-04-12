"""
Module: finbert
Purpose: Dual-FinBERT pipeline for ESG sentiment and classification.
SR 11-7 Relevance: Pillar 2 (Validation) — NLP classification
    confidence scores are logged for audit trail accuracy.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from functools import lru_cache

import torch
from langchain_core.tools import tool
from transformers import pipeline


class FinBERTModels:
    """Container for the dual FinBERT inference pipelines.

    Loads both models on CPU (device=-1) with float32 for
    compatibility with HF Spaces free tier.
    """

    def __init__(self) -> None:
        """Load sentiment and ESG classification pipelines."""
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,
            torch_dtype=torch.float32,
        )
        self.esg_pipe = pipeline(
            "text-classification",
            model="yiyanghkust/finbert-esg",
            device=-1,
            torch_dtype=torch.float32,
        )

    def analyze(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[dict]:
        """Run both pipelines on a list of texts.

        Args:
            texts: List of text strings to analyse.
            batch_size: Batch size for inference.

        Returns:
            List of dicts with keys: text, sentiment,
            sentiment_score, esg_category, esg_confidence.
        """
        sentiments = self.sentiment_pipe(
            texts,
            batch_size=batch_size,
            truncation=True,
        )
        esg_labels = self.esg_pipe(
            texts,
            batch_size=batch_size,
            truncation=True,
        )

        results: list[dict] = []
        for text, sent, esg in zip(
            texts, sentiments, esg_labels
        ):
            results.append(
                {
                    "text": text[:100],
                    "sentiment": sent["label"],
                    "sentiment_score": round(
                        sent["score"], 3
                    ),
                    "esg_category": esg["label"],
                    "esg_confidence": round(
                        esg["score"], 3
                    ),
                }
            )
        return results


@lru_cache(maxsize=1)
def get_finbert_models() -> FinBERTModels:
    """Load both FinBERT models once; cached for process lifetime.

    Returns:
        FinBERTModels instance with both pipelines ready.
    """
    return FinBERTModels()


@tool
def analyze_sentiment_esg(texts: list[str]) -> str:
    """Analyse financial texts for sentiment and ESG category.

    Uses dual FinBERT pipelines: ProsusAI/finbert for sentiment
    and yiyanghkust/finbert-esg for ESG classification.

    Args:
        texts: List of financial text snippets to analyse.

    Returns:
        One line per text with ESG category, sentiment, and
        score, or ERROR string.
    """
    if not texts:
        return "ERROR: texts list is empty"

    try:
        models = get_finbert_models()
        results = models.analyze(texts)
    except Exception as exc:
        return (
            f"ERROR: FinBERT inference failed — {exc}"
        )

    lines: list[str] = []
    for r in results:
        lines.append(
            f"- [{r['esg_category']}] "
            f"{r['sentiment']} ({r['sentiment_score']}): "
            f"{r['text']}"
        )
    return "\n".join(lines)
