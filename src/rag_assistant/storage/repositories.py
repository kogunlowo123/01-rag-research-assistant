"""Repositories: the only place that knows SQL.

Every method takes an explicit ``tenant_id`` and every query filters on it. That
repetition is deliberate — it makes tenant isolation reviewable by reading this
one file, rather than by trusting that each call site remembered to filter.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.domain.models import (
    Chunk,
    Document,
    DocumentStatus,
    QuerySession,
    SourceKind,
)
from rag_assistant.storage.schema import AuditRow, ChunkRow, DocumentRow, PostingRow, SessionRow

#: A stored conversation turn is a (query, answer) pair.
_TURN_ARITY = 2

if TYPE_CHECKING:
    from datetime import datetime


def pack_vector(values: Sequence[float]) -> bytes:
    """Serialise a vector as little-endian float32.

    float32 halves storage against float64 and costs nothing in retrieval
    quality: embedding models emit float32, and cosine ranking is insensitive to
    the last bits of mantissa. The format is explicit rather than
    platform-native so a database file is portable between architectures.
    """
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    """Deserialise a little-endian float32 vector."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))


def _to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        tenant_id=row.tenant_id,
        title=row.title,
        source_kind=SourceKind(row.source_kind),
        source_ref=row.source_ref,
        media_type=row.media_type,
        content_sha256=row.content_sha256,
        byte_size=row.byte_size,
        status=DocumentStatus(row.status),
        chunk_count=row.chunk_count,
        metadata=dict(row.doc_metadata or {}),
        acl=tuple(row.acl or ()),
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        tenant_id=row.tenant_id,
        ordinal=row.ordinal,
        text=row.text,
        token_estimate=row.token_estimate,
        section_path=tuple(row.section_path or ()),
        page=row.page,
        metadata=dict(row.chunk_metadata or {}),
    )


class DocumentRepository:
    """Reads and writes document rows."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def add(self, document: Document) -> Document:
        """Insert a new document row."""
        row = DocumentRow(
            id=document.id,
            tenant_id=document.tenant_id,
            title=document.title,
            source_kind=str(document.source_kind),
            source_ref=document.source_ref,
            media_type=document.media_type,
            content_sha256=document.content_sha256,
            byte_size=document.byte_size,
            status=str(document.status),
            chunk_count=document.chunk_count,
            doc_metadata=dict(document.metadata),
            acl=list(document.acl),
            error=document.error,
        )
        self._session.add(row)
        await self._session.flush()
        return _to_document(row)

    async def get(self, tenant_id: str, document_id: str) -> Document | None:
        """Fetch one document within a tenant."""
        row = await self._session.scalar(
            select(DocumentRow).where(
                DocumentRow.id == document_id,
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.status != str(DocumentStatus.DELETED),
            )
        )
        return _to_document(row) if row else None

    async def find_by_digest(self, tenant_id: str, digest: str) -> Document | None:
        """Find an existing document with the same content, for deduplication."""
        row = await self._session.scalar(
            select(DocumentRow).where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.content_sha256 == digest,
                DocumentRow.status != str(DocumentStatus.DELETED),
            )
        )
        return _to_document(row) if row else None

    async def list(
        self,
        tenant_id: str,
        *,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        """List documents for a tenant, newest first."""
        statement = select(DocumentRow).where(
            DocumentRow.tenant_id == tenant_id,
            DocumentRow.status != str(DocumentStatus.DELETED),
        )
        if status is not None:
            statement = statement.where(DocumentRow.status == str(status))
        statement = statement.order_by(DocumentRow.created_at.desc()).limit(limit).offset(offset)
        rows = (await self._session.scalars(statement)).all()
        return [_to_document(row) for row in rows]

    async def count(self, tenant_id: str) -> int:
        """Count the live documents for a tenant."""
        total = await self._session.scalar(
            select(func.count())
            .select_from(DocumentRow)
            .where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.status != str(DocumentStatus.DELETED),
            )
        )
        return int(total or 0)

    async def set_status(
        self,
        tenant_id: str,
        document_id: str,
        status: DocumentStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Advance a document's lifecycle state."""
        values: dict[str, Any] = {"status": str(status), "error": error}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        await self._session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == document_id, DocumentRow.tenant_id == tenant_id)
            .values(**values)
        )

    async def soft_delete(self, tenant_id: str, document_id: str) -> bool:
        """Mark a document deleted and remove its chunks.

        The document row is retained as a tombstone so that an audit trail
        referencing it stays resolvable, while the chunks — the only part that
        can reach a prompt — are removed immediately along with their postings
        via the schema's cascade.
        """
        result = await self._session.execute(
            update(DocumentRow)
            .where(
                DocumentRow.id == document_id,
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.status != str(DocumentStatus.DELETED),
            )
            .values(status=str(DocumentStatus.DELETED), chunk_count=0)
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]  # CursorResult at runtime
            return False
        await self._session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
        return True


class ChunkRepository:
    """Reads and writes chunk rows, their vectors and their postings."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def add_many(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        provider: str,
        postings: Sequence[dict[str, int]],
    ) -> None:
        """Insert chunks, vectors and postings in one transaction.

        Writing all three together is what keeps dense and sparse retrieval
        consistent: there is no window in which a chunk is searchable by one and
        not the other.
        """
        if len(chunks) != len(vectors) or len(chunks) != len(postings):
            msg = "chunks, vectors and postings must be the same length"
            raise ValueError(msg)

        self._session.add_all(
            [
                ChunkRow(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    tenant_id=chunk.tenant_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    token_estimate=chunk.token_estimate,
                    section_path=list(chunk.section_path),
                    page=chunk.page,
                    chunk_metadata=dict(chunk.metadata),
                    embedding=pack_vector(vector),
                    embedding_provider=provider,
                    term_count=sum(terms.values()),
                )
                for chunk, vector, terms in zip(chunks, vectors, postings, strict=True)
            ]
        )
        self._session.add_all(
            [
                PostingRow(
                    tenant_id=chunk.tenant_id,
                    term=term,
                    chunk_id=chunk.id,
                    frequency=frequency,
                )
                for chunk, terms in zip(chunks, postings, strict=True)
                for term, frequency in terms.items()
            ]
        )
        await self._session.flush()

    async def get_many(self, tenant_id: str, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        """Fetch chunks by id within a tenant."""
        ids = list(chunk_ids)
        if not ids:
            return {}
        rows = (
            await self._session.scalars(
                select(ChunkRow).where(ChunkRow.tenant_id == tenant_id, ChunkRow.id.in_(ids))
            )
        ).all()
        return {row.id: _to_chunk(row) for row in rows}

    async def neighbours(
        self, tenant_id: str, document_id: str, ordinal: int, window: int
    ) -> list[Chunk]:
        """Fetch chunks adjacent to ``ordinal`` in reading order."""
        if window <= 0:
            return []
        rows = (
            await self._session.scalars(
                select(ChunkRow)
                .where(
                    ChunkRow.tenant_id == tenant_id,
                    ChunkRow.document_id == document_id,
                    ChunkRow.ordinal.between(ordinal - window, ordinal + window),
                )
                .order_by(ChunkRow.ordinal)
            )
        ).all()
        return [_to_chunk(row) for row in rows]

    async def load_vectors(
        self, tenant_id: str, provider: str
    ) -> tuple[list[str], list[list[float]]]:
        """Load every vector for a tenant that was produced by ``provider``.

        Filtering on the provider is what stops a model change from silently
        ranking new vectors against incomparable old ones.
        """
        rows = (
            await self._session.execute(
                select(ChunkRow.id, ChunkRow.embedding).where(
                    ChunkRow.tenant_id == tenant_id,
                    ChunkRow.embedding_provider == provider,
                    ChunkRow.embedding.is_not(None),
                )
            )
        ).all()
        ids: list[str] = []
        vectors: list[list[float]] = []
        for chunk_id, blob in rows:
            if blob is None:
                continue
            ids.append(chunk_id)
            vectors.append(unpack_vector(blob))
        return ids, vectors

    async def term_statistics(
        self, tenant_id: str, terms: Sequence[str]
    ) -> dict[str, list[tuple[str, int]]]:
        """Return postings for the requested terms within a tenant."""
        if not terms:
            return {}
        rows = (
            await self._session.execute(
                select(PostingRow.term, PostingRow.chunk_id, PostingRow.frequency).where(
                    PostingRow.tenant_id == tenant_id,
                    PostingRow.term.in_(list(terms)),
                )
            )
        ).all()
        grouped: dict[str, list[tuple[str, int]]] = {}
        for term, chunk_id, frequency in rows:
            grouped.setdefault(term, []).append((chunk_id, frequency))
        return grouped

    async def corpus_statistics(self, tenant_id: str) -> tuple[int, float]:
        """Return the chunk count and mean term count for BM25 normalisation."""
        row = (
            await self._session.execute(
                select(func.count(ChunkRow.id), func.avg(ChunkRow.term_count)).where(
                    ChunkRow.tenant_id == tenant_id
                )
            )
        ).one()
        count = int(row[0] or 0)
        average = float(row[1] or 0.0)
        return count, average

    async def lengths(self, tenant_id: str, chunk_ids: Sequence[str]) -> dict[str, int]:
        """Return term counts for the given chunks."""
        if not chunk_ids:
            return {}
        rows = (
            await self._session.execute(
                select(ChunkRow.id, ChunkRow.term_count).where(
                    ChunkRow.tenant_id == tenant_id, ChunkRow.id.in_(list(chunk_ids))
                )
            )
        ).all()
        return {chunk_id: int(count) for chunk_id, count in rows}

    async def document_acls(
        self, tenant_id: str, document_ids: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        """Return the ACL of each document, for post-retrieval authorisation."""
        if not document_ids:
            return {}
        rows = (
            await self._session.execute(
                select(DocumentRow.id, DocumentRow.acl, DocumentRow.title).where(
                    DocumentRow.tenant_id == tenant_id,
                    DocumentRow.id.in_(list(document_ids)),
                )
            )
        ).all()
        return {row[0]: tuple(row[1] or ()) for row in rows}

    async def document_titles(self, tenant_id: str, document_ids: Sequence[str]) -> dict[str, str]:
        """Return the title of each document, for citation rendering."""
        if not document_ids:
            return {}
        rows = (
            await self._session.execute(
                select(DocumentRow.id, DocumentRow.title).where(
                    DocumentRow.tenant_id == tenant_id,
                    DocumentRow.id.in_(list(document_ids)),
                )
            )
        ).all()
        return {row[0]: row[1] for row in rows}


class SessionRepository:
    """Reads and writes conversation sessions."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def get(self, tenant_id: str, session_id: str) -> QuerySession | None:
        """Fetch a conversation session within a tenant."""
        row = await self._session.scalar(
            select(SessionRow).where(SessionRow.id == session_id, SessionRow.tenant_id == tenant_id)
        )
        if row is None:
            return None
        turns = tuple((pair[0], pair[1]) for pair in (row.turns or []) if len(pair) == _TURN_ARITY)
        return QuerySession(
            id=row.id, tenant_id=row.tenant_id, turns=turns, created_at=row.created_at
        )

    async def upsert(self, session: QuerySession, *, max_turns: int = 20) -> None:
        """Create or update a session, retaining at most ``max_turns`` turns.

        Bounding the history is a memory and cost control: an unbounded thread
        would grow the rewriting prompt without limit.
        """
        turns = [list(turn) for turn in session.turns[-max_turns:]]
        existing = await self._session.get(SessionRow, session.id)
        if existing is None:
            self._session.add(SessionRow(id=session.id, tenant_id=session.tenant_id, turns=turns))
        else:
            if existing.tenant_id != session.tenant_id:
                msg = "session belongs to a different tenant"
                raise ValueError(msg)
            existing.turns = turns
        await self._session.flush()


class AuditRepository:
    """Appends security-relevant events."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def record(
        self,
        *,
        tenant_id: str,
        actor_key_id: str,
        event: str,
        outcome: str,
        request_id: str | None = None,
        subject_id: str | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        """Append one audit event."""
        self._session.add(
            AuditRow(
                tenant_id=tenant_id,
                actor_key_id=actor_key_id,
                event=event,
                outcome=outcome,
                request_id=request_id,
                subject_id=subject_id,
                attributes=attributes or {},
            )
        )
        await self._session.flush()

    async def recent(
        self, tenant_id: str, *, limit: int = 100, since: datetime | None = None
    ) -> list[AuditRow]:
        """Return recent audit events for a tenant, newest first."""
        statement = select(AuditRow).where(AuditRow.tenant_id == tenant_id)
        if since is not None:
            statement = statement.where(AuditRow.created_at >= since)
        statement = statement.order_by(AuditRow.created_at.desc()).limit(limit)
        return list((await self._session.scalars(statement)).all())


__all__ = [
    "AuditRepository",
    "ChunkRepository",
    "DocumentRepository",
    "SessionRepository",
    "pack_vector",
    "unpack_vector",
]
