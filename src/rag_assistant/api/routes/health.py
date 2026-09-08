"""Liveness and readiness endpoints.

``/healthz`` answers "is the process alive?" and must never touch a dependency:
a health check that fails because the database is slow causes an orchestrator to
restart a healthy process during an incident, turning a degradation into an
outage.

``/readyz`` answers "should this instance receive traffic?" and does check
dependencies. It returns 503 when a required component is unavailable, and lists
warnings — such as a development-grade embedding backend in production — without
failing on them.

Neither endpoint requires authentication, and neither reveals configuration
values, versions of dependencies, or error text from an upstream.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from sqlalchemy import text

from rag_assistant import __version__
from rag_assistant.api.dependencies import ContextDep, ServicesDep
from rag_assistant.api.schemas import ComponentStatus, HealthResponse, ReadinessResponse
from rag_assistant.config import ChatBackend, EmbeddingBackend
from rag_assistant.observability.logging import get_logger
from rag_assistant.providers.hashing import HashingEmbeddingProvider

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    """Report that the process is running. Checks nothing external."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(
    services: ServicesDep, context: ContextDep, response: Response
) -> ReadinessResponse:
    """Report whether this instance can serve traffic."""
    components: list[ComponentStatus] = []
    warnings: list[str] = []

    try:
        await context.session.execute(text("SELECT 1"))
        components.append(ComponentStatus(name="database", ready=True))
    except Exception as exc:
        logger.warning("readiness.database_unavailable", error=type(exc).__name__)
        components.append(
            ComponentStatus(name="database", ready=False, detail="the database is unreachable")
        )

    embedding_backend = services.settings.embedding.backend
    components.append(
        ComponentStatus(
            name="embeddings",
            ready=True,
            detail=f"backend={embedding_backend} model={services.embedder.name}",
        )
    )
    if embedding_backend is EmbeddingBackend.HASHING:
        warnings.append(HashingEmbeddingProvider.quality_notice())

    chat_backend = services.settings.chat.backend
    if chat_backend is ChatBackend.EXTRACTIVE:
        components.append(
            ComponentStatus(name="generation", ready=True, detail="backend=extractive")
        )
        warnings.append(
            "The 'extractive' generation backend answers with verbatim source sentences and "
            "cannot synthesise across passages. Set RAG_CHAT__BACKEND=ollama or openai for "
            "generative answers."
        )
    else:
        healthy = await services.chat.health()
        components.append(
            ComponentStatus(
                name="generation",
                ready=True,
                detail=f"backend={chat_backend} reachable={healthy}",
            )
        )
        if not healthy:
            warnings.append(
                "the configured language model is not reachable; queries will be answered by "
                "the extractive fallback until it recovers"
            )

    ready = all(component.ready for component in components)
    if not ready:
        response.status_code = 503
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        version=__version__,
        components=components,
        warnings=warnings,
    )


__all__ = ["router"]
