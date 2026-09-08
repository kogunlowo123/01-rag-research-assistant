"""Dependency wiring for the HTTP layer.

Long-lived collaborators — the database engine, the embedding provider, the
chat provider, the vector-store cache — are built once during application
startup and held on :class:`Services`. Per-request objects — the session and the
repositories bound to it — are created per request and torn down with it.

This split is what makes the vector cache useful (it survives requests) and the
transaction correct (it does not).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_assistant.config import Settings
from rag_assistant.generation.answerer import Answerer
from rag_assistant.indexing.vector_store import NumpyVectorStore
from rag_assistant.ingestion.loaders import UrlLoader
from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.providers.base import ChatProvider, EmbeddingProvider
from rag_assistant.retrieval.pipeline import RetrievalPipeline
from rag_assistant.security.authz import ApiKeyAuthenticator, Principal
from rag_assistant.storage.repositories import (
    AuditRepository,
    ChunkRepository,
    DocumentRepository,
    SessionRepository,
)


@dataclass
class Services:
    """Process-lifetime collaborators, built during startup."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    embedder: EmbeddingProvider
    chat: ChatProvider
    authenticator: ApiKeyAuthenticator
    vector_store: NumpyVectorStore
    ingestion: IngestionPipeline
    url_loader: UrlLoader | None

    async def aclose(self) -> None:
        """Release every long-lived resource."""
        await self.embedder.aclose()
        await self.chat.aclose()
        if self.url_loader is not None:
            await self.url_loader.aclose()
        await self.engine.dispose()


@dataclass
class RequestContext:
    """Per-request collaborators bound to one database session."""

    session: AsyncSession
    documents: DocumentRepository
    chunks: ChunkRepository
    sessions: SessionRepository
    audit: AuditRepository
    vector_store: NumpyVectorStore
    ingestion: IngestionPipeline
    retrieval: RetrievalPipeline
    answerer: Answerer


def get_services(request: Request) -> Services:
    """Return the process-lifetime services from application state."""
    services: Services = request.app.state.services
    return services


def get_settings_dep(request: Request) -> Settings:
    """Return the active settings."""
    return get_services(request).settings


async def get_context(request: Request) -> AsyncIterator[RequestContext]:
    """Build the per-request context and commit or roll back around the handler.

    The transaction wraps the whole request rather than each repository call, so
    an ingestion that fails part-way leaves no chunks behind.
    """
    services = get_services(request)
    session = services.session_factory()
    try:
        documents = DocumentRepository(session)
        chunks = ChunkRepository(session)
        # The vector cache lives on application state so it survives requests.
        # It loads through its own short-lived session rather than this one, so
        # a cache refresh is never entangled with the request's transaction.
        vector_store: NumpyVectorStore = services.vector_store

        yield RequestContext(
            session=session,
            documents=documents,
            chunks=chunks,
            sessions=SessionRepository(session),
            audit=AuditRepository(session),
            vector_store=vector_store,
            # Ingestion owns its own transactions, so it is a process-lifetime
            # collaborator rather than one bound to this request's session.
            ingestion=services.ingestion,
            retrieval=RetrievalPipeline(
                settings=services.settings,
                embedder=services.embedder,
                vector_store=vector_store,
                chunks=chunks,
            ),
            answerer=Answerer(settings=services.settings, provider=services.chat),
        )
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_principal(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Authenticate the caller from ``X-API-Key`` or a bearer token."""
    services = get_services(request)
    presented = x_api_key
    if presented is None and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[len("bearer ") :]
    # AuthorizationError propagates to the domain error handler, which renders
    # it as a 403 with the standard error envelope.
    return services.authenticator.authenticate(presented)


ServicesDep = Annotated[Services, Depends(get_services)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ContextDep = Annotated[RequestContext, Depends(get_context)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]

__all__ = [
    "ContextDep",
    "PrincipalDep",
    "RequestContext",
    "Services",
    "ServicesDep",
    "SettingsDep",
    "get_context",
    "get_principal",
    "get_services",
]
