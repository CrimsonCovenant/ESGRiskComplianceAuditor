"""
Module: ingest
Purpose: Document ingestion pipeline for Qdrant vector store.
SR 11-7 Relevance: Pillar 1 (Development) — every ingested
    document receives a UUID for audit traceability.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import uuid

from fastembed import TextEmbedding
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
)
from qdrant_client.models import PointStruct

from esg_auditor.core.exceptions import EmbeddingError
from esg_auditor.rag.chunker import (
    chunk_regulatory_document,
)
from esg_auditor.rag.embedder import embed_documents
from esg_auditor.tools.qdrant_search import (
    ensure_collection,
)

_UPSERT_BATCH_SIZE = 100


def ingest_documents(
    documents: list[Document],
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
) -> int:
    """Ingest documents into Qdrant with embeddings.

    Ensures the target collection exists, embeds all
    documents, and upserts in batches of 100.

    Args:
        documents: List of Document objects to ingest.
        client: Connected QdrantClient instance.
        embedding_model: fastembed TextEmbedding model.
        collection_name: Target Qdrant collection.

    Returns:
        Count of successfully ingested documents.

    Raises:
        EmbeddingError: On embedding or Qdrant upsert
            failure.
    """
    ensure_collection(client, collection_name)

    doc_vectors = embed_documents(
        documents, embedding_model
    )

    points: list[PointStruct] = []
    for doc, vector in doc_vectors:
        payload = {
            **doc.metadata,
            "text": doc.page_content,
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    try:
        for i in range(
            0, len(points), _UPSERT_BATCH_SIZE
        ):
            batch = points[i : i + _UPSERT_BATCH_SIZE]
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )
    except UnexpectedResponse as exc:
        raise EmbeddingError(
            f"Qdrant upsert failed: {exc}"
        ) from exc

    return len(points)


def ingest_text(
    text: str,
    metadata: dict[str, str | int],
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
) -> int:
    """Convenience wrapper: chunk text then ingest.

    Args:
        text: Raw document text to chunk and ingest.
        metadata: Document metadata for chunking.
        client: Connected QdrantClient instance.
        embedding_model: fastembed TextEmbedding model.
        collection_name: Target Qdrant collection.

    Returns:
        Count of chunks ingested.

    Raises:
        EmbeddingError: On embedding or upsert failure.
        ValueError: If metadata is missing required keys.
    """
    chunks = chunk_regulatory_document(text, metadata)
    if not chunks:
        return 0
    return ingest_documents(
        chunks, client, embedding_model, collection_name
    )
