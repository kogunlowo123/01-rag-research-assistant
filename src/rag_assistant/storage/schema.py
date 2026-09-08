"""Relational schema.

The database is the single source of truth for documents, chunks and their
vectors. Vectors are stored as ``float32`` little-endian blobs next to the chunk
they belong to, in the same transaction, so an index can never contain a vector
whose chunk was rolled back.

Scaling limit, stated plainly
-----------------------------
Dense search loads a tenant's vectors into a NumPy matrix and scans it. That is
exact, has no index to rebuild or tune, and is fast enough for corpora up to
roughly a few hundred thousand chunks on commodity hardware. Beyond that,
switch to ``pgvector`` with an HNSW index: the schema already carries the
vectors in PostgreSQL-compatible form, and
:class:`~rag_assistant.indexing.vector_store.VectorStore` is the single
interface that would need a second implementation. ``docs/architecture.md``
records the measurement behind that threshold.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every mapped class."""


class DocumentRow(Base):
    """A document and its ingestion lifecycle state."""

    __tablename__ = "documents"
    __table_args__ = (
        # Deduplication is per tenant: two tenants uploading the same public
        # paper must each get their own document, and neither may learn that
        # the other has it.
        UniqueConstraint("tenant_id", "content_sha256", name="uq_documents_tenant_digest"),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
        Index("ix_documents_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(2048), nullable=False, default="")
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    doc_metadata: Mapped[dict[str, str]] = mapped_column("metadata", JSON, default=dict)
    acl: Mapped[list[str]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkRow(Base):
    """A retrievable span of a document, with its dense vector and term postings."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        Index("ix_chunks_tenant", "tenant_id"),
        Index("ix_chunks_document_ordinal", "document_id", "ordinal"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict[str, str]] = mapped_column("metadata", JSON, default=dict)

    #: float32 little-endian vector. Stored beside the chunk so an index entry
    #: cannot outlive the text it describes.
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    #: Identifier of the provider and model that produced ``embedding``. Vectors
    #: from different models are not comparable, so search filters on this.
    embedding_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Number of terms in the chunk, used by BM25 length normalisation.
    term_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


class PostingRow(Base):
    """Inverted index posting: one row per (tenant, term, chunk).

    A real inverted index rather than a scan over chunk text. Term statistics
    for BM25 are aggregated from these rows, which keeps sparse retrieval
    correct after deletions without a rebuild.
    """

    __tablename__ = "postings"
    __table_args__ = (
        Index("ix_postings_tenant_term", "tenant_id", "term"),
        Index("ix_postings_chunk", "chunk_id"),
        UniqueConstraint("chunk_id", "term", name="uq_postings_chunk_term"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    term: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    frequency: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class SessionRow(Base):
    """A conversation thread. Turns are stored so follow-up queries can be rewritten."""

    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    turns: Mapped[list[list[str]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )


class AuditRow(Base):
    """Append-only record of security-relevant events.

    Written for every ingestion, every query, and every injection finding. The
    payload never contains document or query text — only identifiers, rule ids,
    scores and decisions — so the audit log is safe to retain for longer than
    the content it describes.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
        Index("ix_audit_event", "event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_key_id: Mapped[str] = mapped_column(String(32), nullable=False, default="anonymous")
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attributes: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


__all__ = ["AuditRow", "Base", "ChunkRow", "DocumentRow", "PostingRow", "SessionRow"]
