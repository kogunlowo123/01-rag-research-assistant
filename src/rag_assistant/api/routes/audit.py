"""Audit trail endpoint.

The audit log is the record of what the system accepted, refused and flagged.
It is scoped to the calling tenant and contains no document or query text — only
identifiers, rule ids, scores and decisions — so it can be retained and exported
under a longer policy than the content it describes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from rag_assistant.api.dependencies import ContextDep, PrincipalDep
from rag_assistant.api.schemas import AuditEventResponse, AuditListResponse

router = APIRouter(prefix="/v1/audit", tags=["audit"])


@router.get("", response_model=AuditListResponse, summary="Recent audit events")
async def list_audit_events(
    context: ContextDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> AuditListResponse:
    """Return the calling tenant's recent audit events, newest first."""
    rows = await context.audit.recent(principal.tenant_id, limit=limit)
    return AuditListResponse(
        events=[
            AuditEventResponse(
                event=row.event,
                outcome=row.outcome,
                actor_key_id=row.actor_key_id,
                request_id=row.request_id,
                subject_id=row.subject_id,
                attributes=dict(row.attributes or {}),
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


__all__ = ["router"]
