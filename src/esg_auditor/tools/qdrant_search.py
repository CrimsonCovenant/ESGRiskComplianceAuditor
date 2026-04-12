"""
Module: qdrant_search
Purpose: Qdrant vector store client and regulatory document search.
SR 11-7 Relevance: Pillar 3 (Governance) — retrieval sources are
    logged per query for audit trail traceability.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
)
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from esg_auditor.config import get_settings
from esg_auditor.core.exceptions import EmbeddingError

_EMBEDDING_DIMS = 384


@lru_cache(maxsize=1)
def get_qdrant_client(
    qdrant_url: str, qdrant_api_key: str
) -> QdrantClient:
    """Return a cached Qdrant client.

    Parameters are primitive strings (hashable) so
    @lru_cache works correctly.

    Args:
        qdrant_url: Qdrant cluster URL from Settings.
        qdrant_api_key: Qdrant API key from Settings.

    Returns:
        Connected QdrantClient instance.
    """
    return QdrantClient(
        url=qdrant_url, api_key=qdrant_api_key
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Return a cached bge-small-en-v1.5 embedding model.

    Returns:
        TextEmbedding instance for 384-dim embeddings.
    """
    return TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


def ensure_collection(
    client: QdrantClient, collection_name: str
) -> None:
    """Create the regulatory docs collection if absent.

    Sets up 384-dim cosine vectors and keyword payload
    indexes for document_type, jurisdiction, and section.

    Args:
        client: Connected QdrantClient.
        collection_name: Target collection name.

    Raises:
        EmbeddingError: On any Qdrant operation failure.
    """
    try:
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        if collection_name in names:
            return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=_EMBEDDING_DIMS,
                distance=Distance.COSINE,
            ),
        )

        for field in (
            "document_type",
            "jurisdiction",
            "section",
        ):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
    except UnexpectedResponse as exc:
        raise EmbeddingError(
            f"Qdrant collection setup failed: {exc}"
        ) from exc


@tool
def search_regulatory_docs(
    query: str,
    document_type: str | None = None,
    jurisdiction: str | None = None,
) -> str:
    """Search regulatory documents in Qdrant vector store.

    Embeds the query with bge-small-en-v1.5 and returns
    the top 5 matching document chunks with metadata.

    Args:
        query: Natural language search query.
        document_type: Optional filter (e.g. "10-K").
        jurisdiction: Optional filter (e.g. "US-Federal").

    Returns:
        Formatted chunks with source IDs for SR 11-7
        retrieval audit trail, or ERROR string.
    """
    settings = get_settings()
    try:
        client = get_qdrant_client(
            settings.qdrant_url, settings.qdrant_api_key
        )
        model = get_embedding_model()

        embeddings = list(model.embed([query]))
        query_vector = list(embeddings[0])

        conditions: list[FieldCondition] = []
        if document_type:
            conditions.append(
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=document_type),
                )
            )
        if jurisdiction:
            conditions.append(
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(value=jurisdiction),
                )
            )

        query_filter = (
            Filter(must=conditions) if conditions else None
        )

        results = client.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=5,
        )

        hits = results.points
    except (
        UnexpectedResponse,
        StopIteration,
        RuntimeError,
    ) as exc:
        return f"ERROR: Qdrant search failed — {exc}"
    except EmbeddingError as exc:
        return f"ERROR: Embedding failed — {exc}"

    if not hits:
        return "No regulatory documents found."

    lines: list[str] = []
    for hit in hits:
        payload = hit.payload or {}
        doc_type = payload.get("document_type", "N/A")
        juris = payload.get("jurisdiction", "N/A")
        section = payload.get("section", "N/A")
        text = payload.get("text", "")[:200]
        source_id = payload.get("source", "N/A")
        lines.append(
            f"[{doc_type} | {juris}] {section}:\n"
            f"{text}\n"
            f"Source: {source_id}"
        )
    return "\n\n".join(lines)
