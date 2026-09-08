"""Document lifecycle endpoints: upload, list, inspect, delete."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status

from rag_assistant.api.dependencies import ContextDep, PrincipalDep, ServicesDep
from rag_assistant.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    IngestResponse,
    IngestUrlRequest,
)
from rag_assistant.domain.models import DocumentStatus, SourceKind
from rag_assistant.errors import (
    DocumentTooLargeError,
    NotFoundError,
    PolicyViolationError,
)
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"])

_ACL_SEPARATOR = ","
_MAX_ACL_ENTRIES = 32


def _parse_acl(raw: str | None) -> tuple[str, ...]:
    """Parse the comma-separated ACL form field into a bounded tuple."""
    if not raw:
        return ()
    entries = [part.strip() for part in raw.split(_ACL_SEPARATOR) if part.strip()]
    return tuple(entries[:_MAX_ACL_ENTRIES])


async def _read_upload(upload: UploadFile, *, limit: int) -> bytes:
    """Read an upload, aborting as soon as it exceeds the limit.

    Reading in bounded pieces rather than calling ``read()`` means a client that
    lies about ``Content-Length`` cannot force the process to buffer an
    arbitrarily large body.
    """
    buffer = bytearray()
    while piece := await upload.read(64 * 1024):
        buffer.extend(piece)
        if len(buffer) > limit:
            raise DocumentTooLargeError(
                "the uploaded document exceeds the configured size limit",
                detail={"limit_bytes": limit},
            )
    return bytes(buffer)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
)
async def upload_document(
    request: Request,
    context: ContextDep,
    services: ServicesDep,
    principal: PrincipalDep,
    file: Annotated[UploadFile, File(description="The document to ingest.")],
    title: Annotated[str | None, Form(max_length=200)] = None,
    acl: Annotated[
        str | None,
        Form(description="Comma-separated group identifiers permitted to retrieve this document."),
    ] = None,
) -> IngestResponse:
    """Ingest an uploaded document.

    The document is validated, parsed, chunked, embedded and indexed
    synchronously, and the response reports how many chunks were created and
    whether the content carried prompt-injection signals.
    """
    payload = await _read_upload(file, limit=services.settings.ingestion.max_document_bytes)
    report = await context.ingestion.ingest(
        IngestionRequest(
            payload=payload,
            filename=file.filename or "document",
            declared_media_type=file.content_type or "text/plain",
            principal=principal,
            source_kind=SourceKind.UPLOAD,
            source_ref=file.filename or "",
            title=title,
            acl=_parse_acl(acl),
        )
    )
    logger.info(
        "documents.ingested",
        document_id=report.document.id,
        chunks=report.chunks_created,
        request_id=getattr(request.state, "request_id", None),
    )
    return IngestResponse.from_domain(report)


@router.post(
    "/from-url",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Fetch and index a document by URL",
)
async def ingest_from_url(
    body: IngestUrlRequest,
    context: ContextDep,
    services: ServicesDep,
    principal: PrincipalDep,
) -> IngestResponse:
    """Fetch a document over HTTP and ingest it.

    Disabled by default. When enabled it is restricted to an explicit host
    allowlist and validated after DNS resolution; see ``THREAT-MODEL.md``.
    """
    if services.url_loader is None:
        raise PolicyViolationError(
            "URL ingestion is disabled; enable RAG_INGESTION__ALLOW_URL_INGESTION and "
            "configure RAG_INGESTION__URL_ALLOWED_HOSTS to use this endpoint"
        )

    loaded = await services.url_loader.load(body.url)
    report = await context.ingestion.ingest(
        IngestionRequest(
            payload=loaded.payload,
            filename=body.url.rsplit("/", 1)[-1] or "document",
            declared_media_type=loaded.declared_media_type,
            principal=principal,
            source_kind=SourceKind.URL,
            source_ref=loaded.source_ref,
            title=body.title,
            acl=tuple(body.acl),
            metadata=body.metadata,
        )
    )
    return IngestResponse.from_domain(report)


@router.get("", response_model=DocumentListResponse, summary="List documents")
async def list_documents(
    context: ContextDep,
    principal: PrincipalDep,
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """List the calling tenant's documents, newest first."""
    documents = await context.documents.list(
        principal.tenant_id, status=document_status, limit=limit, offset=offset
    )
    total = await context.documents.count(principal.tenant_id)
    return DocumentListResponse(
        documents=[DocumentResponse.from_domain(document) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get one document")
async def get_document(
    document_id: str,
    context: ContextDep,
    principal: PrincipalDep,
) -> DocumentResponse:
    """Return one document's metadata and ingestion status."""
    document = await context.documents.get(principal.tenant_id, document_id)
    if document is None:
        raise NotFoundError("document not found or not accessible")
    return DocumentResponse.from_domain(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its index entries",
)
async def delete_document(
    document_id: str,
    context: ContextDep,
    principal: PrincipalDep,
) -> Response:
    """Remove a document from the corpus.

    Chunks, vectors and inverted-index postings are deleted immediately and the
    tenant's vector cache is invalidated, so the document stops being
    retrievable within the same request. A tombstone row is retained so audit
    events that reference the document remain resolvable.
    """
    deleted = await context.documents.soft_delete(principal.tenant_id, document_id)
    if not deleted:
        raise NotFoundError("document not found or not accessible")

    context.vector_store.invalidate(principal.tenant_id)
    await context.audit.record(
        tenant_id=principal.tenant_id,
        actor_key_id=principal.key_id,
        event="document.delete",
        outcome="deleted",
        subject_id=document_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
