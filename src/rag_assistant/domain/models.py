"""Core domain types.

These models are the contract between layers. They are deliberately free of
persistence, HTTP and provider concerns so that the ingestion, retrieval and
generation packages can be tested and reasoned about independently.

One idea is load-bearing throughout: :class:`TrustLevel`. Every piece of text
that reaches a prompt carries a trust level, and the generation layer refuses
to treat ``UNTRUSTED`` text as instructions. Retrieved document content is
always ``UNTRUSTED``, no matter who uploaded it.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, computed_field

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


def _utc_now() -> datetime:
    """Timezone-aware current time. Naive datetimes are banned by lint rule DTZ."""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Generate a prefixed, URL-safe identifier.

    The prefix makes identifiers self-describing in logs and traces, which
    matters when correlating a chunk id back to its document during an
    incident.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


class TrustLevel(StrEnum):
    """How much authority a piece of text carries when it reaches a prompt.

    ``SYSTEM`` text is authored by this application. ``USER`` text is authored
    by an authenticated caller and may express intent but never policy.
    ``UNTRUSTED`` text originates from a document, a tool result or the wider
    internet; it is evidence only and can never change behaviour.
    """

    SYSTEM = "system"
    USER = "user"
    UNTRUSTED = "untrusted"


class SourceKind(StrEnum):
    """Where a document came from."""

    UPLOAD = "upload"
    URL = "url"
    INLINE = "inline"


class DocumentStatus(StrEnum):
    """Lifecycle state of a document within the ingestion pipeline."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DomainModel(BaseModel):
    """Base for domain models: immutable, strict, and free of extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Document(DomainModel):
    """An ingested source document and its lifecycle state."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    tenant_id: NonEmptyStr
    title: NonEmptyStr
    source_kind: SourceKind
    source_ref: str = Field(
        default="",
        description="Original filename or URL. Sanitised; never used to build a filesystem path.",
    )
    media_type: NonEmptyStr
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = Field(default=0, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)
    acl: tuple[str, ...] = Field(
        default=(),
        description="Group identifiers permitted to retrieve this document. Empty means "
        "tenant-wide visibility.",
    )
    error: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @staticmethod
    def digest(payload: bytes) -> str:
        """Content digest used for deduplication and for change detection."""
        return hashlib.sha256(payload).hexdigest()

    def with_status(self, status: DocumentStatus, *, error: str | None = None) -> Self:
        """Return a copy advanced to a new lifecycle state."""
        return self.model_copy(update={"status": status, "error": error, "updated_at": _utc_now()})


class Chunk(DomainModel):
    """A retrievable span of a document.

    ``ordinal`` preserves reading order so neighbouring chunks can be stitched
    back together to restore context around a hit.
    """

    id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: NonEmptyStr
    tenant_id: NonEmptyStr
    ordinal: int = Field(ge=0)
    text: NonEmptyStr
    token_estimate: int = Field(ge=0)
    section_path: tuple[str, ...] = Field(
        default=(),
        description="Heading breadcrumb, for example ('3. Method', '3.2 Sampling').",
    )
    page: int | None = Field(default=None, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def indexing_text(self) -> str:
        """The text that is embedded and indexed, as opposed to the text that is quoted.

        The heading breadcrumb is prepended here rather than being baked into
        ``text`` because the two have different jobs. Retrieval needs it: an
        embedding of "the limit is 30 days" is far more useful when it also
        carries "Refunds > Eligibility". A citation quote does not: a quote that
        begins with its own breadcrumb reads as though the document said it.
        """
        if not self.section_path:
            return self.text
        return f"{' > '.join(self.section_path)}\n\n{self.text}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def locator(self) -> str:
        """Human-readable position used in citations."""
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        parts.append(f"#{self.ordinal}")
        return " ".join(parts)


class InjectionFinding(DomainModel):
    """A prompt-injection signal detected inside untrusted content."""

    rule_id: NonEmptyStr
    description: NonEmptyStr
    severity: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(
        default="",
        max_length=200,
        description="Redacted evidence span, truncated so audit logs stay bounded.",
    )


class ChunkMatch(DomainModel):
    """A retrieved chunk with its scores and any security annotations."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    trust: TrustLevel = TrustLevel.UNTRUSTED
    injection_findings: tuple[InjectionFinding, ...] = ()
    neutralised: bool = Field(
        default=False,
        description="True when instruction-like spans were stripped before prompting.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def injection_risk(self) -> float:
        """Highest severity among the findings on this chunk."""
        if not self.injection_findings:
            return 0.0
        return max(finding.severity for finding in self.injection_findings)


class RetrievalDiagnostics(DomainModel):
    """Explains why a particular set of chunks was returned.

    Retrieval is the part of a RAG system that most often fails silently, so
    the pipeline records its intermediate state and the API can return it.
    """

    original_query: str
    rewritten_queries: tuple[str, ...] = ()
    dense_candidate_count: int = 0
    sparse_candidate_count: int = 0
    fused_candidate_count: int = 0
    reranked: bool = False
    dropped_by_policy: int = 0
    dropped_by_authorization: int = 0
    neutralised_chunks: int = 0
    latency_ms: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(DomainModel):
    """Everything retrieval produced for one query."""

    matches: tuple[ChunkMatch, ...]
    diagnostics: RetrievalDiagnostics


class Citation(DomainModel):
    """A pointer from a sentence in the answer to the evidence supporting it."""

    marker: NonEmptyStr = Field(description="The in-text marker, for example '[1]'.")
    document_id: NonEmptyStr
    document_title: NonEmptyStr
    chunk_id: NonEmptyStr
    locator: str = ""
    quote: str = Field(default="", max_length=600)
    support_score: float = Field(ge=0.0, le=1.0)


class GroundingReport(DomainModel):
    """Per-answer grounding measurement.

    ``supported_sentences`` counts sentences whose content is traceable to
    retrieved evidence. A low ratio is the signal that an answer is drifting
    away from its sources.
    """

    total_sentences: int = Field(ge=0)
    supported_sentences: int = Field(ge=0)
    unsupported_sentences: tuple[str, ...] = ()
    score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_grounded(self) -> bool:
        """Whether every sentence in the answer traced back to evidence."""
        return self.total_sentences > 0 and self.supported_sentences == self.total_sentences


class Answer(DomainModel):
    """The generated answer plus the evidence and signals needed to trust it."""

    query: str
    text: str
    citations: tuple[Citation, ...] = ()
    grounding: GroundingReport
    refused: bool = False
    refusal_reason: str | None = None
    warnings: tuple[str, ...] = ()
    model: str = ""
    provider: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    diagnostics: RetrievalDiagnostics | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Composite confidence in [0, 1].

        Deliberately simple and explainable: the grounding score attenuated by
        citation coverage. It is a signal for a human reviewer, not a
        calibrated probability, and the README says so.
        """
        if self.refused or not self.citations:
            return 0.0
        return round(self.grounding.score * (0.5 + 0.5 * self.grounding.citation_coverage), 4)


class IngestionReport(DomainModel):
    """Outcome of ingesting one document."""

    document: Document
    chunks_created: int = Field(ge=0)
    duplicate_of: str | None = None
    warnings: tuple[str, ...] = ()
    injection_findings: tuple[InjectionFinding, ...] = ()
    latency_ms: dict[str, float] = Field(default_factory=dict)


class ParsedDocument(DomainModel):
    """Intermediate result of parsing raw bytes into text plus metadata."""

    text: str
    media_type: str
    metadata: dict[str, str] = Field(default_factory=dict)
    page_offsets: tuple[tuple[int, int], ...] = Field(
        default=(),
        description="(character offset, page number) pairs, ascending, for page attribution.",
    )

    def page_for_offset(self, offset: int) -> int | None:
        """Return the 1-based page containing ``offset``, if page data exists."""
        if not self.page_offsets:
            return None
        page: int | None = None
        for start, number in self.page_offsets:
            if start <= offset:
                page = number
            else:
                break
        return page


class QuerySession(DomainModel):
    """A conversation thread. Prior turns are used to disambiguate follow-ups."""

    id: str = Field(default_factory=lambda: new_id("ses"))
    tenant_id: NonEmptyStr
    turns: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="(query, answer) pairs in chronological order.",
    )
    created_at: datetime = Field(default_factory=_utc_now)

    def recent_context(self, limit: int = 3) -> str:
        """Render the last ``limit`` turns as plain text for query rewriting."""
        selected = self.turns[-limit:] if limit > 0 else ()
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in selected)

    def with_turn(self, query: str, answer: str) -> Self:
        """Return a copy with one more turn appended."""
        return self.model_copy(update={"turns": (*self.turns, (query, answer))})


def redact_metadata(
    raw: dict[str, Any], *, max_keys: int = 32, max_len: int = 512
) -> dict[str, str]:
    """Coerce arbitrary parsed metadata into a bounded, string-only mapping.

    Document metadata is attacker-controlled: a PDF's ``/Author`` field can hold
    thousands of characters, control bytes, or text designed to be read as an
    instruction. This normalises keys, drops control characters, truncates
    values and caps the number of entries so metadata can never dominate a
    prompt or a log line.
    """
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        if len(cleaned) >= max_keys:
            break
        if value is None:
            continue
        safe_key = "".join(ch for ch in str(key).strip().lower() if ch.isalnum() or ch in "._-")[
            :64
        ]
        if not safe_key:
            continue
        text = "".join(ch for ch in str(value) if ch.isprintable() or ch == " ").strip()
        if not text:
            continue
        cleaned[safe_key] = text[:max_len]
    return cleaned


__all__ = [
    "Answer",
    "Chunk",
    "ChunkMatch",
    "Citation",
    "Document",
    "DocumentStatus",
    "DomainModel",
    "GroundingReport",
    "IngestionReport",
    "InjectionFinding",
    "NonEmptyStr",
    "ParsedDocument",
    "QuerySession",
    "RetrievalDiagnostics",
    "RetrievalResult",
    "SourceKind",
    "TrustLevel",
    "new_id",
    "redact_metadata",
]
