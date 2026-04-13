"""
Module: eval.ragas_eval
Purpose: RAGAS evaluation pipeline for ESG audit RAG quality measurement.
SR 11-7 Relevance: Pillar 2 (Validation) — RAGAS metrics (Faithfulness,
    Context Precision, Context Recall) are the ongoing monitoring KPIs
    required for production model validation. Target: all metrics >= 0.8.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from __future__ import annotations

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
)
from ragas.metrics._context_precision import (
    LLMContextPrecisionWithReference,
)

from esg_auditor.config import Settings
from esg_auditor.core.exceptions import StructuredOutputError


def run_evaluation(
    eval_data: dict[str, list],
    settings: Settings,
) -> dict[str, float]:
    """Run RAGAS evaluation on ESG audit outputs.

    SR 11-7 Pillar 2: RAGAS metrics are the quantitative
    validation KPIs. Target thresholds: Faithfulness >= 0.8,
    Context Precision >= 0.8, Context Recall >= 0.8.

    Args:
        eval_data: Dict with keys 'question', 'answer',
            'contexts', 'ground_truth'. Each value is a
            list of equal length.
        settings: Application settings for model config.

    Returns:
        Dict with keys: faithfulness, context_precision,
        context_recall. All values are floats in
        range [0.0, 1.0].

    Raises:
        StructuredOutputError: If evaluation fails or
            data is invalid.
    """
    required_keys = {
        "question",
        "answer",
        "contexts",
        "ground_truth",
    }
    missing = required_keys - set(eval_data.keys())
    if missing:
        raise StructuredOutputError(
            f"eval_data missing required keys: {missing}"
        )
    if not eval_data["question"]:
        raise StructuredOutputError(
            "eval_data must contain at least one example"
        )

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
            )
        )
        precision_metric = LLMContextPrecisionWithReference()
        dataset = Dataset.from_dict(eval_data)
        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                precision_metric,
                LLMContextRecall(),
            ],
            llm=evaluator_llm,
        )
        # ragas 0.4.3 returns EvaluationResult where
        # _scores_dict values are lists (per-sample).
        # Compute mean for aggregate scores.
        scores = result._scores_dict
        def _mean(vals: list) -> float:
            clean = [v for v in vals if v is not None]
            return sum(clean) / len(clean) if clean else 0.0

        return {
            "faithfulness": round(
                _mean(scores["faithfulness"]), 4
            ),
            "context_precision": round(
                _mean(scores[precision_metric.name]), 4
            ),
            "context_recall": round(
                _mean(scores["context_recall"]), 4
            ),
        }
    except StructuredOutputError:
        raise
    except Exception as exc:
        raise StructuredOutputError(
            f"RAGAS evaluation failed: {exc}"
        ) from exc
