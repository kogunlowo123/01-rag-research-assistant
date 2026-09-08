"""Persistence: schema, engine lifecycle and repositories."""

from rag_assistant.storage.db import (
    create_engine,
    create_schema,
    create_session_factory,
    session_scope,
)
from rag_assistant.storage.repositories import (
    AuditRepository,
    ChunkRepository,
    DocumentRepository,
    SessionRepository,
    pack_vector,
    unpack_vector,
)

__all__ = [
    "AuditRepository",
    "ChunkRepository",
    "DocumentRepository",
    "SessionRepository",
    "create_engine",
    "create_schema",
    "create_session_factory",
    "pack_vector",
    "session_scope",
    "unpack_vector",
]
