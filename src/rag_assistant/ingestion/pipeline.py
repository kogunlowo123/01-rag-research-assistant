"""The ingestion pipeline.

Order of operations:

1. **Validate** size, media type and magic bytes before anything parses the
   payload. A parser is the wrong place to discover that a "PDF" is a ZIP.
2. **Deduplicate** by content digest within the tenant. Re-uploading the same
   file returns the existing document rather than creating a second copy that
   would then compete with itself in retrieval.
3. **Parse** into text plus sanitised metadata.
4. **Scan** the document text for injection signals. The finding is recorded on
   the document at ingestion time, not only at retrieval time, so an operator
   can answer "what did we accept?" without replaying every query.
5. **Chunk**, structure-aware, with overlap.
6. **Embed** in batches.
7. **Index** — chunks, vectors and inverted-index postings in one transaction.

Transaction ownership
---------------------
This pipeline manages its own transactions rather than joining a caller's, and
that is load-bearing. The document's ``status`` column exists so a failure is
*recorded*; written inside the transaction that the failure then rolls back, the
row would vanish and an operator would be left with a document that simply never
arrived. Each stage therefore commits, and the failure path opens a fresh
transaction of its own.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_assistant.config import Settings
from rag_assistant.domain.models import (
    Chunk,
    Document,
    DocumentStatus,
    IngestionReport,
    ParsedDocument,
    SourceKind,
    redact_metadata,
)
from rag_assistant.errors import IngestionError, RagError
from rag_assistant.indexing.text import term_frequencies
from rag_assistant.ingestion.chunking import Chunker
from rag_assistant.ingestion.parsers import parse
from rag_assistant.observability.logging import get_logger
from rag_assistant.observability.tracing import documents_ingested, get_tracer
from rag_assistant.security import injection
from rag_assistant.security.normalization import strip_control_characters
from rag_assistant.security.sources import safe_filename, validate_upload
from rag_assistant.storage.repositories import (
    AuditRepository,
    ChunkRepository,
    DocumentRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from rag_assistant.indexing.vector_store import VectorStore
    from rag_assistant.providers.base import EmbeddingProvider
    from rag_assistant.security.authz import Principal

logger = get_logger(__name__)
tracer = get_tracer()

_MAX_TITLE_CHARS = 200


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    """One document to ingest."""

    payload: bytes
    filename: str
    declared_media_type: str
    principal: Principal
    source_kind: SourceKind = SourceKind.UPLOAD
    source_ref: str = ""
    title: str | None = None
    acl: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _Repositories:
    """The repositories bound to one ingestion transaction."""

    documents: DocumentRepository
    chunks: ChunkRepository
    audit: AuditRepository


class IngestionPipeline:
    """Validates, parses, chunks, embeds and indexes documents."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: EmbeddingProvider,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
    ) -> None:
        """Wire the pipeline's collaborators."""
        self._settings = settings
        self._embedder = embedder
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._chunker = Chunker(
            target_tokens=settings.ingestion.chunk_target_tokens,
            overlap_tokens=settings.ingestion.chunk_overlap_tokens,
            max_chunks=settings.ingestion.max_chunks_per_document,
        )

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[_Repositories]:
        """Open a transaction that commits on success and rolls back on error."""
        session = self._session_factory()
        try:
            yield _Repositories(
                documents=DocumentRepository(session),
                chunks=ChunkRepository(session),
                audit=AuditRepository(session),
            )
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def ingest(self, request: IngestionRequest) -> IngestionReport:
        """Ingest one document and return its report."""
        config = self._settings.ingestion
        tenant = request.principal.tenant_id
        latencies: dict[str, float] = {}

        started = time.perf_counter()
        media_type = validate_upload(
            request.payload,
            declared_media_type=request.declared_media_type,
            allowed_media_types=config.allowed_media_types,
            max_bytes=config.max_document_bytes,
        )
        latencies["validate_ms"] = (time.perf_counter() - started) * 1000.0

        digest = Document.digest(request.payload)
        async with self._transaction() as repositories:
            if existing := await repositories.documents.find_by_digest(tenant, digest):
                await repositories.audit.record(
                    tenant_id=tenant,
                    actor_key_id=request.principal.key_id,
                    event="document.ingest",
                    outcome="duplicate",
                    subject_id=existing.id,
                    attributes={"digest": digest},
                )
                documents_ingested.add(1, {"status": "duplicate"})
                return IngestionReport(
                    document=existing,
                    chunks_created=0,
                    duplicate_of=existing.id,
                    warnings=("a document with identical content already exists",),
                    latency_ms=latencies,
                )

            document = await repositories.documents.add(
                self._new_document(request, media_type=media_type, digest=digest)
            )

        try:
            return await self._run(document, request, media_type, latencies)
        except RagError as exc:
            await self._record_failure(document, request, exc)
            raise

    def _new_document(self, request: IngestionRequest, *, media_type: str, digest: str) -> Document:
        display_name = safe_filename(request.filename)
        source_ref = (
            safe_filename(request.source_ref)
            if request.source_kind is SourceKind.UPLOAD
            else request.source_ref[:2048]
        )
        return Document(
            tenant_id=request.principal.tenant_id,
            title=(request.title or display_name)[:_MAX_TITLE_CHARS],
            source_kind=request.source_kind,
            source_ref=source_ref,
            media_type=media_type,
            content_sha256=digest,
            byte_size=len(request.payload),
            status=DocumentStatus.PENDING,
            metadata=redact_metadata(request.metadata or {}),
            acl=request.acl,
        )

    async def _record_failure(
        self, document: Document, request: IngestionRequest, exc: RagError
    ) -> None:
        """Persist the failure in a transaction of its own."""
        async with self._transaction() as repositories:
            await repositories.documents.set_status(
                document.tenant_id, document.id, DocumentStatus.FAILED, error=exc.message
            )
            await repositories.audit.record(
                tenant_id=document.tenant_id,
                actor_key_id=request.principal.key_id,
                event="document.ingest",
                outcome="failed",
                subject_id=document.id,
                attributes={"error_code": exc.code},
            )
        documents_ingested.add(1, {"status": "failed"})
        logger.warning(
            "ingestion.failed",
            document_id=document.id,
            error_code=exc.code,
            media_type=document.media_type,
        )

    async def _set_status(self, document: Document, status: DocumentStatus, **kwargs: Any) -> None:
        async with self._transaction() as repositories:
            await repositories.documents.set_status(
                document.tenant_id, document.id, status, **kwargs
            )

    async def _run(
        self,
        document: Document,
        request: IngestionRequest,
        media_type: str,
        latencies: dict[str, float],
    ) -> IngestionReport:
        tenant = document.tenant_id
        warnings: list[str] = []

        with tracer.start_as_current_span("ingestion.pipeline") as span:
            span.set_attribute("ingestion.media_type", media_type)
            span.set_attribute("ingestion.bytes", len(request.payload))

            await self._set_status(document, DocumentStatus.PARSING)
            started = time.perf_counter()
            parsed = parse(request.payload, media_type)
            parsed = parsed.model_copy(update={"text": strip_control_characters(parsed.text)})
            latencies["parse_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            scan = injection.scan(parsed.text)
            latencies["scan_ms"] = (time.perf_counter() - started) * 1000.0
            if scan.is_suspicious:
                warnings.append(
                    "this document contains instruction-like text; retrieved passages from it "
                    "will be neutralised or excluded according to the configured policy"
                )
                logger.warning(
                    "security.injection_at_ingest",
                    document_id=document.id,
                    risk=scan.risk,
                    rules=[finding.rule_id for finding in scan.findings],
                )

            await self._set_status(document, DocumentStatus.CHUNKING)
            started = time.perf_counter()
            chunks = self._chunk(parsed, document, warnings)
            latencies["chunk_ms"] = (time.perf_counter() - started) * 1000.0

            await self._set_status(document, DocumentStatus.EMBEDDING)
            started = time.perf_counter()
            vectors = await self._embed([chunk.indexing_text for chunk in chunks])
            latencies["embed_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            async with self._transaction() as repositories:
                await repositories.chunks.add_many(
                    chunks,
                    vectors,
                    provider=self._embedder.name,
                    postings=[term_frequencies(chunk.indexing_text) for chunk in chunks],
                )
                await repositories.documents.set_status(
                    tenant, document.id, DocumentStatus.INDEXED, chunk_count=len(chunks)
                )
                await repositories.audit.record(
                    tenant_id=tenant,
                    actor_key_id=request.principal.key_id,
                    event="document.ingest",
                    outcome="indexed",
                    subject_id=document.id,
                    attributes={
                        "chunks": len(chunks),
                        "media_type": media_type,
                        "injection_risk": scan.risk,
                        "injection_rules": [finding.rule_id for finding in scan.findings],
                    },
                )
            # Invalidated only after the write has committed: doing it earlier
            # would let a concurrent query rebuild the cache from a state that
            # has not landed yet.
            self._vector_store.invalidate(tenant)
            latencies["index_ms"] = (time.perf_counter() - started) * 1000.0

            span.set_attribute("ingestion.chunks", len(chunks))

        documents_ingested.add(1, {"status": "indexed"})
        logger.info(
            "ingestion.completed",
            document_id=document.id,
            chunks=len(chunks),
            media_type=media_type,
        )

        return IngestionReport(
            document=document.model_copy(
                update={"status": DocumentStatus.INDEXED, "chunk_count": len(chunks)}
            ),
            chunks_created=len(chunks),
            warnings=tuple(warnings),
            injection_findings=scan.findings,
            latency_ms={key: round(value, 3) for key, value in latencies.items()},
        )

    def _chunk(
        self, parsed: ParsedDocument, document: Document, warnings: list[str]
    ) -> list[Chunk]:
        chunks = self._chunker.chunk(parsed, document_id=document.id, tenant_id=document.tenant_id)
        if not chunks:
            raise IngestionError("the document produced no indexable chunks")
        if len(chunks) >= self._settings.ingestion.max_chunks_per_document:
            warnings.append(
                "the document hit the per-document chunk limit and was indexed only in part"
            )
        # The document title is carried on every chunk so citation rendering
        # never needs a second query during answer assembly.
        return [
            chunk.model_copy(
                update={"metadata": {**chunk.metadata, "document_title": document.title}}
            )
            for chunk in chunks
        ]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts in provider-sized batches."""
        batch_size = self._settings.embedding.batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self._embedder.embed_documents(texts[start : start + batch_size]))
        if len(vectors) != len(texts):
            raise IngestionError("the embedding provider returned the wrong number of vectors")
        return vectors


__all__ = ["IngestionPipeline", "IngestionRequest"]
