"""HTTP request and response models.

These are separate from the domain models on purpose. The domain models are
free to change shape as the pipeline evolves; the wire contract is not. Keeping
them apart also means an internal field cannot become part of the public API by
accident — ``Chunk.text`` is never serialised to a client, only the quoted span
that supports a citation is.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from rag_assistant.domain.models import (
    Answer,
    Document,
    DocumentStatus,
    IngestionReport,
    RetrievalDiagnostics,
    SourceKind,
)

QueryText = Annotated[str, StringConstraints(min_length=1, max_length=4000, strip_whitespace=True)]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]


class ApiModel(BaseModel):
    """Base for wire models: unknown fields are rejected rather than ignored."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    """Liveness response."""

    status: str = "ok"
    version: str


class ComponentStatus(ApiModel):
    """Readiness of one dependency."""

    name: str
    ready: bool
    detail: str = ""


class ReadinessResponse(ApiModel):
    """Readiness response.

    Returns 503 when any required component is not ready, so an orchestrator can
    hold traffic. Warnings do not fail readiness but are surfaced so a
    deployment running on a development-grade backend is visible.
    """

    status: str
    version: str
    components: list[ComponentStatus]
    warnings: list[str] = Field(default_factory=list)


class IngestUrlRequest(ApiModel):
    """Ingest a document by URL. Refused unless URL ingestion is enabled."""

    url: Annotated[str, StringConstraints(min_length=8, max_length=2048)]
    title: str | None = Field(default=None, max_length=200)
    acl: list[Identifier] = Field(default_factory=list, max_length=32)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentResponse(ApiModel):
    """A document's metadata and lifecycle state."""

    id: str
    title: str
    source_kind: SourceKind
    source_ref: str
    media_type: str
    byte_size: int
    status: DocumentStatus
    chunk_count: int
    metadata: dict[str, str]
    acl: list[str]
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> DocumentResponse:
        """Project a domain document onto the wire model."""
        return cls(
            id=document.id,
            title=document.title,
            source_kind=document.source_kind,
            source_ref=document.source_ref,
            media_type=document.media_type,
            byte_size=document.byte_size,
            status=document.status,
            chunk_count=document.chunk_count,
            metadata=document.metadata,
            acl=list(document.acl),
            error=document.error,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentListResponse(ApiModel):
    """A page of documents."""

    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class InjectionFindingResponse(ApiModel):
    """A prompt-injection signal found in a document."""

    rule_id: str
    description: str
    severity: float


class IngestResponse(ApiModel):
    """Result of ingesting one document."""

    document: DocumentResponse
    chunks_created: int
    duplicate_of: str | None = None
    warnings: list[str] = Field(default_factory=list)
    injection_findings: list[InjectionFindingResponse] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, report: IngestionReport) -> IngestResponse:
        """Project an ingestion report onto the wire model."""
        return cls(
            document=DocumentResponse.from_domain(report.document),
            chunks_created=report.chunks_created,
            duplicate_of=report.duplicate_of,
            warnings=list(report.warnings),
            injection_findings=[
                InjectionFindingResponse(
                    rule_id=finding.rule_id,
                    description=finding.description,
                    severity=finding.severity,
                )
                for finding in report.injection_findings
            ],
            latency_ms=report.latency_ms,
        )


class QueryRequest(ApiModel):
    """Ask a question of the corpus."""

    query: QueryText
    session_id: Identifier | None = Field(
        default=None,
        description="Conversation thread. Prior turns are used to resolve follow-up questions.",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_diagnostics: bool = Field(
        default=False,
        description="Return the retrieval breakdown. Echoes the query back, so it is opt-in.",
    )


class CitationResponse(ApiModel):
    """A citation and the quoted span that supports it."""

    marker: str
    document_id: str
    document_title: str
    chunk_id: str
    locator: str
    quote: str
    support_score: float


class GroundingResponse(ApiModel):
    """Grounding measurement for an answer."""

    score: float
    citation_coverage: float
    total_sentences: int
    supported_sentences: int
    is_grounded: bool
    unsupported_sentences: list[str] = Field(default_factory=list)


class DiagnosticsResponse(ApiModel):
    """Retrieval diagnostics."""

    original_query: str
    rewritten_queries: list[str]
    dense_candidate_count: int
    sparse_candidate_count: int
    fused_candidate_count: int
    reranked: bool
    dropped_by_policy: int
    dropped_by_authorization: int
    neutralised_chunks: int
    latency_ms: dict[str, float]

    @classmethod
    def from_domain(cls, diagnostics: RetrievalDiagnostics) -> DiagnosticsResponse:
        """Project retrieval diagnostics onto the wire model."""
        return cls(
            original_query=diagnostics.original_query,
            rewritten_queries=list(diagnostics.rewritten_queries),
            dense_candidate_count=diagnostics.dense_candidate_count,
            sparse_candidate_count=diagnostics.sparse_candidate_count,
            fused_candidate_count=diagnostics.fused_candidate_count,
            reranked=diagnostics.reranked,
            dropped_by_policy=diagnostics.dropped_by_policy,
            dropped_by_authorization=diagnostics.dropped_by_authorization,
            neutralised_chunks=diagnostics.neutralised_chunks,
            latency_ms=diagnostics.latency_ms,
        )


class QueryResponse(ApiModel):
    """An answer, its evidence and the signals needed to trust it."""

    answer: str
    refused: bool
    refusal_reason: str | None
    confidence: float
    citations: list[CitationResponse]
    grounding: GroundingResponse
    warnings: list[str]
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    session_id: str | None = None
    diagnostics: DiagnosticsResponse | None = None

    @classmethod
    def from_domain(cls, answer: Answer, *, session_id: str | None = None) -> QueryResponse:
        """Project a domain answer onto the wire model."""
        return cls(
            answer=answer.text,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            confidence=answer.confidence,
            citations=[
                CitationResponse(
                    marker=citation.marker,
                    document_id=citation.document_id,
                    document_title=citation.document_title,
                    chunk_id=citation.chunk_id,
                    locator=citation.locator,
                    quote=citation.quote,
                    support_score=citation.support_score,
                )
                for citation in answer.citations
            ],
            grounding=GroundingResponse(
                score=answer.grounding.score,
                citation_coverage=answer.grounding.citation_coverage,
                total_sentences=answer.grounding.total_sentences,
                supported_sentences=answer.grounding.supported_sentences,
                is_grounded=answer.grounding.is_grounded,
                unsupported_sentences=list(answer.grounding.unsupported_sentences),
            ),
            warnings=list(answer.warnings),
            provider=answer.provider,
            model=answer.model,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            latency_ms=answer.latency_ms,
            session_id=session_id,
            diagnostics=(
                DiagnosticsResponse.from_domain(answer.diagnostics)
                if answer.diagnostics is not None
                else None
            ),
        )


class AuditEventResponse(ApiModel):
    """One audit event."""

    event: str
    outcome: str
    actor_key_id: str
    request_id: str | None
    subject_id: str | None
    attributes: dict[str, object]
    created_at: datetime


class AuditListResponse(ApiModel):
    """A page of audit events."""

    events: list[AuditEventResponse]


class ErrorResponse(ApiModel):
    """The single error shape returned by every failing endpoint."""

    code: str
    message: str
    request_id: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "ApiModel",
    "AuditEventResponse",
    "AuditListResponse",
    "CitationResponse",
    "ComponentStatus",
    "DiagnosticsResponse",
    "DocumentListResponse",
    "DocumentResponse",
    "ErrorResponse",
    "GroundingResponse",
    "HealthResponse",
    "IngestResponse",
    "IngestUrlRequest",
    "InjectionFindingResponse",
    "QueryRequest",
    "QueryResponse",
    "ReadinessResponse",
]
