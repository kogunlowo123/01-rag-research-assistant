"""Database engine and session lifecycle.

The engine is async so a slow query cannot block the event loop that is also
waiting on model providers. SQLite is the default because it makes the project
runnable from a clean clone with no services; PostgreSQL is a URL change.

Two SQLite-specific pragmas are applied on connect. Without ``foreign_keys=ON``
SQLite silently ignores the cascade deletes the schema declares, which would
orphan chunks and postings when a document is removed. ``journal_mode=WAL``
allows a reader during a write, which matters because ingestion writes for
several seconds while queries continue to arrive.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_assistant.config import Settings
from rag_assistant.observability.logging import get_logger
from rag_assistant.storage.schema import Base

logger = get_logger(__name__)


def _async_url(raw: str) -> str:
    """Translate a synchronous SQLAlchemy URL into its async driver equivalent.

    Accepting either form means operators can paste the URL they already have
    without learning which driver this application uses.
    """
    url = make_url(raw)
    if url.drivername in {"sqlite", "sqlite+pysqlite"}:
        url = url.set(drivername="sqlite+aiosqlite")
    elif url.drivername in {"postgresql", "postgresql+psycopg2", "postgresql+psycopg"}:
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


def _ensure_sqlite_directory(url: str) -> None:
    parsed = make_url(url)
    if not parsed.drivername.startswith("sqlite"):
        return
    database = parsed.database
    if not database or database == ":memory:":
        return
    Path(database).parent.mkdir(parents=True, exist_ok=True)


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine for the configured database."""
    url = _async_url(settings.storage.database_url)
    _ensure_sqlite_directory(url)
    is_sqlite = url.startswith("sqlite")

    kwargs: dict[str, Any] = {"echo": settings.storage.echo_sql, "future": True}
    if not is_sqlite:
        kwargs |= {
            "pool_size": settings.storage.pool_size,
            "pool_timeout": settings.storage.pool_timeout_seconds,
            "pool_pre_ping": True,
        }

    engine = create_async_engine(url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory used by request handlers and background work."""
    return async_sessionmaker(
        engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )


async def create_schema(engine: AsyncEngine) -> None:
    """Create tables that do not yet exist.

    Adequate for a single-service application whose schema changes ship with the
    code. A deployment with more than one writer, or one that needs reversible
    migrations, should adopt Alembic; ``docs/operations.md`` describes the
    migration path and why it is not the default here.
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("storage.schema_ready")


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional session that commits on success and rolls back on error."""
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


__all__ = [
    "create_engine",
    "create_schema",
    "create_session_factory",
    "session_scope",
]
