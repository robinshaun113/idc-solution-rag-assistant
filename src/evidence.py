"""Evidence metadata shared by retrieval, generation and API responses."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document


def make_evidence_id(source: str, page: int | str | None, ordinal: int, text: str) -> str:
    """Create a stable, short identifier for one retrievable evidence block."""
    raw = f"{source}|{page}|{ordinal}|{text.strip()}".encode("utf-8")
    return f"ev_{hashlib.sha256(raw).hexdigest()[:16]}"


def backfill_legacy_evidence_metadata(documents: list[Document]) -> list[Document]:
    """Add order-independent IDs to results loaded from a pre-V4 index.

    Retrieval rank is deliberately excluded: rank may change between queries
    and must never change the identifier shown to a user.
    """
    for doc in documents:
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        raw = f"{source}|{page}|{doc.page_content.strip()}".encode("utf-8")
        doc.metadata.update(
            {
                "source": source,
                "document_id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:12],
                "chunk_index": doc.metadata.get("chunk_index"),
                "evidence_id": f"ev_{hashlib.sha256(raw).hexdigest()[:16]}",
            }
        )
    return documents


def attach_evidence_metadata(documents: list[Document]) -> list[Document]:
    """Ensure every chunk has source/page/document/evidence identifiers."""
    for ordinal, doc in enumerate(documents):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        chunk_index = doc.metadata.get("chunk_index", ordinal)
        doc.metadata.update(
            {
                "source": source,
                "document_id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:12],
                "chunk_index": chunk_index,
                "evidence_id": make_evidence_id(source, page, chunk_index, doc.page_content),
            }
        )
    return documents


def evidence_payload(doc: Document, preview_chars: int = 180) -> dict:
    """Return the public, machine-readable evidence contract."""
    page = doc.metadata.get("page")
    return {
        "evidence_id": doc.metadata.get("evidence_id", ""),
        "source": doc.metadata.get("source", "unknown"),
        "page": page + 1 if isinstance(page, int) else page,
        "chunk_index": doc.metadata.get("chunk_index"),
        "preview": doc.page_content[:preview_chars].replace("\n", " ").strip(),
    }
