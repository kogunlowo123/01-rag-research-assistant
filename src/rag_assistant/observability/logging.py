"""Structured logging with automatic secret redaction.

Every log record is a JSON object carrying the request id, tenant and trace
context, so a single request can be reconstructed from a log aggregator without
string parsing.

A redaction processor runs last in the chain. It rewrites values whose key
looks sensitive and masks credential-shaped substrings anywhere in the event.
Redaction sits in the logging pipeline rather than at each call site because a
control that depends on every future call site remembering it is not a control.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any, Final

import structlog
from structlog.types import EventDict, Processor

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)

#: Keys whose values are replaced wholesale, matched case-insensitively as substrings.
SENSITIVE_KEY_PARTS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "cookie",
        "credential",
        "password",
        "passwd",
        "private_key",
        "secret",
        "session_key",
        "set-cookie",
        "token",
        "x-api-key",
    }
)

REDACTED: Final[str] = "[redacted]"
_MAX_VALUE_CHARS: Final[int] = 2000
#: Recursion limit when walking nested structures during redaction. Deeper
#: values are emitted unchanged rather than risking a pathological traversal.
_MAX_REDACT_DEPTH: Final[int] = 4

#: Credential-shaped substrings that must never reach a log sink even when they
#: appear inside an otherwise innocuous message such as an exception string.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI-style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*", re.IGNORECASE),
    re.compile(r"\bey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),  # JWT
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _mask_text(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _redact_value(key: str, value: Any, depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return REDACTED
    if depth > _MAX_REDACT_DEPTH:
        return value
    if isinstance(value, str):
        masked = _mask_text(value)
        return (
            masked
            if len(masked) <= _MAX_VALUE_CHARS
            else masked[:_MAX_VALUE_CHARS] + "...[truncated]"
        )
    if isinstance(value, MutableMapping):
        return {str(k): _redact_value(str(k), v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_redact_value(key, item, depth + 1) for item in value)
    return value


def redact_processor(_logger: object, _name: str, event_dict: EventDict) -> EventDict:
    """Structlog processor that removes credentials from every emitted record."""
    return {str(key): _redact_value(str(key), value) for key, value in event_dict.items()}


def request_context_processor(_logger: object, _name: str, event_dict: EventDict) -> EventDict:
    """Attach the ambient request identifiers to every record."""
    if (request_id := request_id_var.get()) is not None:
        event_dict.setdefault("request_id", request_id)
    if (tenant_id := tenant_id_var.get()) is not None:
        event_dict.setdefault("tenant_id", tenant_id)
    if (session_id := session_id_var.get()) is not None:
        event_dict.setdefault("session_id", session_id)
    return event_dict


def trace_context_processor(_logger: object, _name: str, event_dict: EventDict) -> EventDict:
    """Attach the active OpenTelemetry trace and span ids when tracing is on.

    This is what makes a log line clickable from a trace view. The import is
    local so the logging module stays usable if the OTel SDK is absent.
    """
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - OTel is a hard dependency here
        return event_dict

    span = trace.get_current_span()
    context = span.get_span_context()
    if context.is_valid:
        event_dict.setdefault("trace_id", format(context.trace_id, "032x"))
        event_dict.setdefault("span_id", format(context.span_id, "016x"))
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "json",
    service_name: str = "rag-research-assistant",
) -> None:
    """Install the structlog pipeline and route stdlib logging through it.

    Safe to call more than once; later calls replace the configuration, which
    is what tests want when they toggle formats.
    """
    threshold = logging.getLevelNamesMapping()[level.upper()]

    # The shared chain runs for records from this application and, via
    # ``foreign_pre_chain``, for records emitted by third-party libraries
    # through the standard library. Both therefore carry the same fields and the
    # same redaction — a deployment with two log formats is a deployment whose
    # aggregation queries only find half the story.
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        request_context_processor,
        trace_context_processor,
        structlog.processors.StackInfoRenderer(),
        redact_processor,
    ]

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            # Redaction runs again after the traceback has been rendered: an
            # exception string can carry a credential that the structured
            # fields did not.
            redact_processor,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(threshold)

    for noisy in ("uvicorn.access", "uvicorn.error", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, threshold))

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_request_context(
    *, request_id: str, tenant_id: str | None = None, session_id: str | None = None
) -> None:
    """Bind identifiers for the duration of the current request."""
    request_id_var.set(request_id)
    tenant_id_var.set(tenant_id)
    session_id_var.set(session_id)


def clear_request_context() -> None:
    """Clear the request identifiers once the request has completed."""
    request_id_var.set(None)
    tenant_id_var.set(None)
    session_id_var.set(None)


__all__ = [
    "REDACTED",
    "SENSITIVE_KEY_PARTS",
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "redact_processor",
    "request_id_var",
    "session_id_var",
    "tenant_id_var",
]
