"""HTTP middleware: correlation, body limits, security headers, access logging.

Order matters and is fixed in :func:`rag_assistant.api.app.create_app`. The body
limit runs before anything reads the body, and correlation runs outermost so
every log line — including one emitted while rejecting an oversized body —
carries a request id.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Final

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from rag_assistant.observability.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)

logger = get_logger(__name__)

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

#: Applied to every response. The API returns JSON and never renders HTML, so a
#: restrictive policy costs nothing and removes a class of browser-side attacks
#: against any tool that displays a response inline.
SECURITY_HEADERS: Final[dict[str, str]] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

_UUID_HEX_LENGTH: Final[int] = 64


def _sanitise_request_id(raw: str | None) -> str:
    """Accept a client-supplied correlation id only if it is safe to log.

    A caller-controlled value ends up in log records and response headers, so it
    is bounded and restricted to characters that cannot forge a log line or a
    header. Anything else is replaced rather than rejected: correlation is a
    convenience, not a reason to fail a request.
    """
    if not raw:
        return uuid.uuid4().hex
    candidate = raw.strip()[:_UUID_HEX_LENGTH]
    if candidate and all(char.isalnum() or char in "-_" for char in candidate):
        return candidate
    return uuid.uuid4().hex


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, binds it to the log context and echoes it back."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Bind correlation state for the duration of the request."""
        request_id = _sanitise_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        bind_request_context(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Rejects oversized requests before the body is read.

    ``Content-Length`` is checked first because it is free. It is only a hint,
    so the ingestion path independently enforces the document size limit while
    reading; this middleware is the cheap outer bound that keeps an obviously
    oversized request from reaching application code at all.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        """Configure the maximum accepted body size."""
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Reject requests whose declared length exceeds the limit."""
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "validation_error",
                        "message": "invalid Content-Length header",
                    },
                )
            if length > self._max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "code": "document_too_large",
                        "message": "the request body exceeds the configured limit",
                        "detail": {"limit_bytes": self._max_bytes},
                    },
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds hardening headers to every response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach the static security header set."""
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emits one structured record per request.

    The record carries the route template rather than the concrete path, so
    document identifiers do not end up in log aggregation, and never carries the
    query string or body.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Time the request and log its outcome."""
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http.request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            raise

        route = request.scope.get("route")
        logger.info(
            "http.request",
            method=request.method,
            route=getattr(route, "path", request.url.path),
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return response


__all__ = [
    "REQUEST_ID_HEADER",
    "SECURITY_HEADERS",
    "AccessLogMiddleware",
    "BodyLimitMiddleware",
    "CorrelationMiddleware",
    "SecurityHeadersMiddleware",
]
