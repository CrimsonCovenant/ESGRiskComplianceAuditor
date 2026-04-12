"""
Module: embedder
Purpose: Document and query embedding using fastembed.
SR 11-7 Relevance: Pillar 1 (Development) — embedding model
    version is pinned for reproducibility across audits.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from fastembed import TextEmbedding
from langchain_core.documents import Document

from esg_auditor.core.exceptions import EmbeddingError

EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS: int = 384

_BATCH_SIZE = 32


def embed_documents(
    documents: list[Document],
    model: TextEmbedding,
) -> list[tuple[Document, list[float]]]:
    """Embed a list of documents in batches.

    Processes documents in batches of 32 to avoid OOM on
    large corpora.

    Args:
        documents: List of Document objects to embed.
        model: fastembed TextEmbedding model instance.

    Returns:
        List of (Document, vector) tuples.

    Raises:
        EmbeddingError: On any fastembed or iteration error.
    """
    results: list[tuple[Document, list[float]]] = []
    try:
        for i in range(0, len(documents), _BATCH_SIZE):
            batch = documents[i : i + _BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            embeddings = list(model.embed(texts))
            for doc, emb in zip(batch, embeddings):
                results.append((doc, list(emb)))
    except StopIteration as exc:
        raise EmbeddingError(
            "Embedding iterator exhausted unexpectedly"
        ) from exc
    except Exception as exc:
        raise EmbeddingError(
            f"Document embedding failed: {exc}"
        ) from exc
    return results


def embed_query(
    query: str, model: TextEmbedding
) -> list[float]:
    """Embed a single query string.

    Args:
        query: Natural language query to embed.
        model: fastembed TextEmbedding model instance.

    Returns:
        Embedding vector as list of floats.

    Raises:
        EmbeddingError: On any fastembed error.
    """
    try:
        embeddings = list(model.embed([query]))
        return list(embeddings[0])
    except StopIteration as exc:
        raise EmbeddingError(
            "Query embedding returned no results"
        ) from exc
    except Exception as exc:
        raise EmbeddingError(
            f"Query embedding failed: {exc}"
        ) from exc
