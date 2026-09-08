"""In-process runtime.

The HTTP layer is one way to drive this application; the CLI and the evaluation
runner are others. All three need the same object graph, so it is assembled here
once and the API's dependency wiring, the CLI and the evaluator all use it.

Without this, the evaluation harness would have to go through HTTP to exercise
the pipeline, which would make evaluation depend on a running server and would
measure the network as well as the system under test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_assistant.config import Settings, get_settings
from rag_assistant.generation.answerer import Answerer
from rag_assistant.indexing.vector_store import NumpyVectorStore
from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.observability.logging import configure_logging
from rag_assistant.providers.base import ChatProvider, EmbeddingProvider
from rag_assistant.providers.registry import build_chat_provider, build_embedding_provider
from rag_assistant.retrieval.pipeline import RetrievalPipeline
from rag_assistant.storage.db import create_engine, create_schema, create_session_factory
from rag_assistant.storage.repositories import (
    AuditRepository,
    ChunkRepository,
    DocumentRepository,
    SessionRepository,
)


@dataclass(frozen=True, slots=True)
class UnitOfWork:
    """Repositories bound to one transaction, plus the pipelines that use them."""

    session: AsyncSession
    documents: DocumentRepository
    chunks: ChunkRepository
    sessions: SessionRepository
    audit: AuditRepository
    retrieval: RetrievalPipeline
    answerer: Answerer


class Runtime:
    """Owns the process-lifetime object graph outside the HTTP layer."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Build providers and the database engine from configuration."""
        self.settings: Settings = settings or get_settings()
        self.engine: AsyncEngine = create_engine(self.settings)
        self.session_factory: async_sessionmaker[AsyncSession] = create_session_factory(self.engine)
        self.embedder: EmbeddingProvider = build_embedding_provider(self.settings)
        self.chat: ChatProvider = build_chat_provider(self.settings)
        self.vector_store = NumpyVectorStore(self._load_vectors)
        self.ingestion = IngestionPipeline(
            settings=self.settings,
            embedder=self.embedder,
            session_factory=self.session_factory,
            vector_store=self.vector_store,
        )

    async def _load_vectors(
        self, tenant_id: str, provider: str
    ) -> tuple[list[str], list[list[float]]]:
        async with self.session_factory() as session:
            return await ChunkRepository(session).load_vectors(tenant_id, provider)

    async def start(self) -> None:
        """Create the schema if it does not exist."""
        await create_schema(self.engine)

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator[UnitOfWork]:
        """Open a transaction and yield the repositories and pipelines bound to it."""
        session = self.session_factory()
        try:
            chunks = ChunkRepository(session)
            yield UnitOfWork(
                session=session,
                documents=DocumentRepository(session),
                chunks=chunks,
                sessions=SessionRepository(session),
                audit=AuditRepository(session),
                retrieval=RetrievalPipeline(
                    settings=self.settings,
                    embedder=self.embedder,
                    vector_store=self.vector_store,
                    chunks=chunks,
                ),
                answerer=Answerer(settings=self.settings, provider=self.chat),
            )
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def aclose(self) -> None:
        """Release providers and the database engine."""
        await self.embedder.aclose()
        await self.chat.aclose()
        await self.engine.dispose()

    async def __aenter__(self) -> Self:
        """Start the runtime for use as an async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the runtime on context exit."""
        await self.aclose()


async def build_runtime(settings: Settings | None = None) -> Runtime:
    """Create and start a runtime with logging configured."""
    active = settings or get_settings()
    configure_logging(
        level=active.observability.log_level,
        fmt=active.observability.log_format,
        service_name=active.observability.service_name,
    )
    runtime = Runtime(active)
    await runtime.start()
    return runtime


__all__ = ["Runtime", "UnitOfWork", "build_runtime"]
