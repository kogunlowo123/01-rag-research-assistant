"""Application factory.

Builds the FastAPI application, installs middleware in a fixed order, wires the
exception handlers that turn domain errors into the single documented error
shape, and manages the lifecycle of process-lifetime resources.

Startup deliberately fails loudly. Configuration invariants are checked before
the first request, so a deployment that has authentication disabled in
production, or a URL allowlist that could never match, refuses to start rather
than serving in that state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_assistant import __version__
from rag_assistant.api.dependencies import Services
from rag_assistant.api.middleware import (
    AccessLogMiddleware,
    BodyLimitMiddleware,
    CorrelationMiddleware,
    SecurityHeadersMiddleware,
)
from rag_assistant.api.routes import audit, documents, health, query
from rag_assistant.config import Settings, get_settings
from rag_assistant.errors import RagError
from rag_assistant.indexing.vector_store import NumpyVectorStore
from rag_assistant.ingestion.loaders import UrlLoader
from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.observability.logging import configure_logging, get_logger
from rag_assistant.observability.tracing import configure_tracing, shutdown_tracing
from rag_assistant.providers.registry import build_chat_provider, build_embedding_provider
from rag_assistant.security.authz import ApiKeyAuthenticator
from rag_assistant.storage.db import create_engine, create_schema, create_session_factory
from rag_assistant.storage.repositories import ChunkRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)

DESCRIPTION: Final[str] = """\
A retrieval-augmented generation service that answers questions from an ingested
document corpus and reports how well each answer is supported by its sources.

Retrieved document text is treated as untrusted throughout: it never enters a
system prompt, it is scanned for prompt-injection signals before it reaches a
model, and every answer carries a measured grounding score and resolved
citations. An answer that cannot be traced back to retrieved evidence is
refused rather than returned.
"""


def _build_services(settings: Settings) -> Services:
    """Construct process-lifetime collaborators from configuration."""
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    async def load_vectors(tenant_id: str, provider: str) -> tuple[list[str], list[list[float]]]:
        """Load a tenant's vectors in a short-lived session of its own."""
        async with session_factory() as session:
            return await ChunkRepository(session).load_vectors(tenant_id, provider)

    url_loader = (
        UrlLoader(
            allowed_schemes=settings.ingestion.url_allowed_schemes,
            allowed_hosts=settings.ingestion.url_allowed_hosts,
            max_bytes=settings.ingestion.max_document_bytes,
            timeout_seconds=settings.ingestion.url_fetch_timeout_seconds,
            allow_private_network=settings.ingestion.allow_private_network_fetch,
        )
        if settings.ingestion.allow_url_ingestion
        else None
    )

    embedder = build_embedding_provider(settings)
    vector_store = NumpyVectorStore(load_vectors)

    return Services(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        embedder=embedder,
        chat=build_chat_provider(settings),
        authenticator=ApiKeyAuthenticator(
            keys=settings.security.api_keys,
            require_key=settings.security.require_api_key,
        ),
        vector_store=vector_store,
        ingestion=IngestionPipeline(
            settings=settings,
            embedder=embedder,
            session_factory=session_factory,
            vector_store=vector_store,
        ),
        url_loader=url_loader,
    )


def _error_response(
    request: Request, *, status_code: int, code: str, message: str, detail: dict[str, object]
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
            "detail": detail,
        },
    )


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagError)
    async def _domain_error(request: Request, exc: RagError) -> JSONResponse:
        # Client errors are expected traffic and are logged at info; server
        # errors are not, and carry a stack trace.
        log = (
            logger.info if exc.status_code < HTTPStatus.INTERNAL_SERVER_ERROR else logger.exception
        )
        log("api.domain_error", code=exc.code, status=exc.status_code)
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's raw error list can echo the submitted value back, which may
        # be user content, so only the location and the rule are returned.
        fields = [
            {
                "location": ".".join(str(part) for part in error.get("loc", ())),
                "rule": error.get("type", ""),
            }
            for error in exc.errors()[:10]
        ]
        return _error_response(
            request,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="validation_error",
            message="the request did not match the expected schema",
            detail={"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            code="http_error",
            message=str(exc.detail),
            detail={},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The exception is logged in full and the client is told nothing about
        # it: an unhandled error's message is exactly the kind of internal
        # detail that should not cross the boundary.
        logger.exception("api.unhandled_error", error_type=type(exc).__name__)
        return _error_response(
            request,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="an internal error occurred",
            detail={},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings so tests can vary them."""
    active = settings or get_settings()
    configure_logging(
        level=active.observability.log_level,
        fmt=active.observability.log_format,
        service_name=active.observability.service_name,
    )
    active.enforce_environment_invariants()
    configure_tracing(active)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        services = _build_services(active)
        application.state.services = services
        await create_schema(services.engine)
        logger.info(
            "app.started",
            environment=str(active.environment),
            embedding_backend=str(active.embedding.backend),
            chat_backend=str(active.chat.backend),
            version=__version__,
        )
        try:
            yield
        finally:
            await services.aclose()
            shutdown_tracing()
            logger.info("app.stopped")

    app = FastAPI(
        title="RAG Research Assistant",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # Outermost first. Correlation must wrap everything so that a rejection by
    # the body limit is still logged with a request id.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(BodyLimitMiddleware, max_bytes=active.security.max_request_bytes)
    app.add_middleware(CorrelationMiddleware)

    _install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(audit.router)

    if active.observability.tracing_enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")

    return app


def instrumented_session_factory(app: FastAPI) -> async_sessionmaker[AsyncSession]:
    """Expose the session factory. Used by the CLI and by integration tests."""
    services: Services = app.state.services
    return services.session_factory


__all__ = ["create_app", "instrumented_session_factory"]
