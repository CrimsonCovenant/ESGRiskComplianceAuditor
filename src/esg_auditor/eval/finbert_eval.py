"""
Module: eval.finbert_eval
Purpose: FinBERT sentiment accuracy evaluation on financial phrasebank.
SR 11-7 Relevance: Pillar 2 (Validation) — NLP model accuracy on a
    standard financial benchmark establishes baseline performance for
    audit trail confidence score interpretation.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from __future__ import annotations

import io
import random
import re
import zipfile

import requests
from sklearn.metrics import accuracy_score, f1_score

from esg_auditor.config import Settings, get_settings
from esg_auditor.tools.finbert import get_finbert_models

LABEL_MAP: dict[str, int] = {
    "positive": 2,
    "negative": 0,
    "neutral": 1,
}

PHRASEBANK_DATASET = "financial_phrasebank"
PHRASEBANK_CONFIG = "sentences_allagree"

_PHRASEBANK_ZIP_URL = (
    "https://huggingface.co/datasets/financial_phrasebank"
    "/resolve/main/data/FinancialPhraseBank-v1.0.zip"
)

_PHRASEBANK_CACHE: dict[str, list] | None = None

_LABEL_STR_TO_INT = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def _load_phrasebank() -> dict[str, list]:
    """Download and parse the FinancialPhraseBank sentences_allagree.

    datasets 4.x dropped support for loading scripts, so we fetch
    the raw ZIP from the HF Hub, extract sentences_allagree.txt,
    and parse the ``sentence@label`` format.
    """
    global _PHRASEBANK_CACHE  # noqa: PLW0603
    if _PHRASEBANK_CACHE is not None:
        return _PHRASEBANK_CACHE

    resp = requests.get(_PHRASEBANK_ZIP_URL, timeout=30)
    resp.raise_for_status()

    sentences: list[str] = []
    labels: list[int] = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        target = None
        for name in zf.namelist():
            if "Sentences_AllAgree" in name and name.endswith(
                ".txt"
            ):
                target = name
                break
        if target is None:
            raise FileNotFoundError(
                "Sentences_AllAgree.txt not found in ZIP"
            )
        raw = zf.read(target).decode("utf-8", errors="replace")
        for line in raw.strip().splitlines():
            # Format: "sentence text@label"
            match = re.match(
                r"^(.+)@(positive|negative|neutral)\s*$",
                line,
            )
            if match:
                sentences.append(match.group(1).strip())
                labels.append(
                    _LABEL_STR_TO_INT[match.group(2)]
                )

    _PHRASEBANK_CACHE = {
        "sentence": sentences,
        "label": labels,
    }
    return _PHRASEBANK_CACHE


def evaluate_finbert_accuracy(
    settings: Settings | None = None,
    full: bool = False,
) -> dict[str, float | int | str]:
    """Evaluate FinBERT sentiment accuracy on financial phrasebank.

    SR 11-7 Pillar 2: Establishes baseline accuracy for the NLP
    model used to classify ESG news sentiment, enabling confidence
    score interpretation in audit reports.

    Args:
        settings: Application settings. Uses get_settings()
            if None.
        full: If True, evaluate all sentences. If False
            (default), evaluate a random 200-sentence sample.

    Returns:
        Dict with accuracy, per-class F1 scores, sample size,
        and model/dataset metadata.
    """
    if settings is None:
        settings = get_settings()

    data = _load_phrasebank()
    sentences: list[str] = list(data["sentence"])
    true_labels: list[int] = list(data["label"])

    if not full:
        indices = random.sample(
            range(len(sentences)),
            min(200, len(sentences)),
        )
        sentences = [sentences[i] for i in indices]
        true_labels = [true_labels[i] for i in indices]

    models = get_finbert_models()
    predictions_raw = models.sentiment_pipe(
        sentences, truncation=True
    )
    pred_labels = [
        LABEL_MAP[p["label"].lower()]
        for p in predictions_raw
    ]

    accuracy = accuracy_score(true_labels, pred_labels)
    f1_per_class = f1_score(
        true_labels,
        pred_labels,
        average=None,
        labels=[0, 1, 2],
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "sample_size": len(sentences),
        "negative_f1": round(float(f1_per_class[0]), 4),
        "neutral_f1": round(float(f1_per_class[1]), 4),
        "positive_f1": round(float(f1_per_class[2]), 4),
        "dataset": (
            f"{PHRASEBANK_DATASET}/{PHRASEBANK_CONFIG}"
        ),
        "model": "ProsusAI/finbert",
    }
