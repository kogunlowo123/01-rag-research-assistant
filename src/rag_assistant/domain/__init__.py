"""Domain model: the vocabulary shared by every layer of the application."""

from rag_assistant.domain.models import (
    Answer,
    Chunk,
    ChunkMatch,
    Citation,
    Document,
    DocumentStatus,
    GroundingReport,
    IngestionReport,
    InjectionFinding,
    RetrievalDiagnostics,
    RetrievalResult,
    SourceKind,
    TrustLevel,
)

__all__ = [
    "Answer",
    "Chunk",
    "ChunkMatch",
    "Citation",
    "Document",
    "DocumentStatus",
    "GroundingReport",
    "IngestionReport",
    "InjectionFinding",
    "RetrievalDiagnostics",
    "RetrievalResult",
    "SourceKind",
    "TrustLevel",
]
