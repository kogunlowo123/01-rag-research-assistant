"""Query endpoints: ask a question, and inspect retrieval without generating."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from rag_assistant.api.dependencies import ContextDep, PrincipalDep, ServicesDep
from rag_assistant.api.schemas import (
    DiagnosticsResponse,
    QueryRequest,
    QueryResponse,
)
from rag_assistant.domain.models import QuerySession
from rag_assistant.errors import ValidationError
from rag_assistant.generation.answerer import AnswerRequest
from rag_assistant.observability.logging import get_logger, session_id_var
from rag_assistant.retrieval.pipeline import RetrievalRequest

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["query"])

#: Number of prior turns fed to query rewriting. More context does not improve
#: reference resolution and does make the rewrite noisier.
_HISTORY_TURNS = 3


@router.post("/query", response_model=QueryResponse, summary="Ask a question of the corpus")
async def query(
    body: QueryRequest,
    context: ContextDep,
    services: ServicesDep,
    principal: PrincipalDep,
) -> QueryResponse:
    """Retrieve evidence and produce a grounded, cited answer.

    The response always carries the grounding measurement and any warnings, and
    a refusal is a normal 200 response with ``refused=true`` rather than an
    error — a refusal is a correct outcome, not a failure.
    """
    if len(body.query) > services.settings.security.max_query_chars:
        raise ValidationError("the query exceeds the configured maximum length")

    conversation: QuerySession | None = None
    history = ""
    if body.session_id:
        session_id_var.set(body.session_id)
        conversation = await context.sessions.get(principal.tenant_id, body.session_id)
        if conversation is None:
            conversation = QuerySession(id=body.session_id, tenant_id=principal.tenant_id)
        history = conversation.recent_context(_HISTORY_TURNS)

    retrieval = await context.retrieval.retrieve(
        RetrievalRequest(
            query=body.query,
            principal=principal,
            history=history,
            top_k=body.top_k,
            include_diagnostics=body.include_diagnostics,
        )
    )

    titles = await context.chunks.document_titles(
        principal.tenant_id,
        sorted({match.chunk.document_id for match in retrieval.matches}),
    )

    answer = await context.answerer.answer(
        AnswerRequest(
            query=body.query,
            retrieval=retrieval,
            document_titles=titles,
            include_diagnostics=body.include_diagnostics,
        )
    )

    if conversation is not None:
        summary = answer.text or (answer.refusal_reason or "")
        await context.sessions.upsert(conversation.with_turn(body.query, summary[:1000]))

    await context.audit.record(
        tenant_id=principal.tenant_id,
        actor_key_id=principal.key_id,
        event="query.answer",
        outcome="refused" if answer.refused else "answered",
        subject_id=body.session_id,
        attributes={
            "grounding": answer.grounding.score,
            "citations": len(answer.citations),
            "passages": len(retrieval.matches),
            "dropped_by_policy": retrieval.diagnostics.dropped_by_policy,
            "dropped_by_authorization": retrieval.diagnostics.dropped_by_authorization,
            "neutralised": retrieval.diagnostics.neutralised_chunks,
            "provider": answer.provider,
        },
    )

    return QueryResponse.from_domain(answer, session_id=body.session_id)


@router.get(
    "/retrieve",
    response_model=DiagnosticsResponse,
    summary="Inspect retrieval for a query without generating an answer",
)
async def retrieve(
    context: ContextDep,
    principal: PrincipalDep,
    q: Annotated[str, Query(min_length=1, max_length=4000)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 8,
) -> DiagnosticsResponse:
    """Return the retrieval breakdown for a query.

    Exists because "the answer is wrong" is almost always a retrieval problem,
    and diagnosing it should not require a model call, a token budget or a
    debugger.
    """
    result = await context.retrieval.retrieve(
        RetrievalRequest(
            query=q,
            principal=principal,
            top_k=top_k,
            include_diagnostics=True,
        )
    )
    return DiagnosticsResponse.from_domain(result.diagnostics)


__all__ = ["router"]
