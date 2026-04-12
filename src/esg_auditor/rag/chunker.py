"""
Module: chunker
Purpose: Regulatory document chunking with section-aware splitting.
SR 11-7 Relevance: Pillar 1 (Development) — document lineage is
    preserved through chunking via metadata propagation and
    unique chunk IDs.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

REGULATORY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
)

METADATA_SCHEMA: dict[str, str | int] = {
    "source": "",
    "document_type": "",
    "company": "",
    "ticker": "",
    "date": "",
    "jurisdiction": "",
    "section": "",
    "chunk_index": 0,
}

_SECTION_PATTERN = re.compile(
    r"(?=(?:Item|ITEM|Section|SECTION)\s+\d+[A-Za-z]?[\.:])"
)


def chunk_regulatory_document(
    text: str,
    metadata: dict[str, str | int],
) -> list[Document]:
    """Split a regulatory document into section-aware chunks.

    Validates metadata keys against METADATA_SCHEMA, splits
    on SEC/regulatory section headers, then applies recursive
    character splitting within each section.

    Args:
        text: Full document text.
        metadata: Document metadata dict. Must contain all
            keys from METADATA_SCHEMA.

    Returns:
        List of Document objects with enriched metadata
        including chunk_id, chunk_index, and section title.
        Returns empty list if text is blank.

    Raises:
        ValueError: If required metadata keys are missing.
    """
    missing = [
        k for k in METADATA_SCHEMA if k not in metadata
    ]
    if missing:
        raise ValueError(
            "chunk_regulatory_document: missing required "
            f"metadata keys: {missing}"
        )

    if not text.strip():
        return []

    sections = _SECTION_PATTERN.split(text)
    documents: list[Document] = []
    chunk_index = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if not body:
            body = title
            title = metadata.get("section", "") or "General"

        chunks = REGULATORY_SPLITTER.split_text(body)

        for chunk_text in chunks:
            chunk_meta = {
                **metadata,
                "section": title,
                "chunk_index": chunk_index,
                "chunk_id": str(uuid.uuid4()),
            }
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_meta,
                )
            )
            chunk_index += 1

    return documents
