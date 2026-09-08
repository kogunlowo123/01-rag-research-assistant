"""Domain exception hierarchy.

Every error raised deliberately by the application derives from
:class:`RagError`. Each carries an HTTP status and a stable machine-readable
``code`` so the API layer can translate exceptions without knowing about
individual failure modes, and so clients can branch on ``code`` rather than on
human-readable prose.

Messages are safe to return to callers: they never embed filesystem paths,
provider URLs, credentials or raw document content.
"""

from __future__ import annotations

from http import HTTPStatus


class RagError(Exception):
    """Base class for every deliberate application error."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: dict[str, object] | None = None) -> None:
        """Record a client-safe message and optional structured detail."""
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConfigurationError(RagError):
    """The application is configured in a way that cannot work."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "configuration_error"


class ValidationError(RagError):
    """Caller-supplied input failed validation."""

    status_code = HTTPStatus.BAD_REQUEST
    code = "validation_error"


class DocumentTooLargeError(ValidationError):
    """An uploaded or fetched document exceeded the configured size limit."""

    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    code = "document_too_large"


class UnsupportedMediaTypeError(ValidationError):
    """The document's media type is not in the ingestion allowlist."""

    status_code = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class UnsafeSourceError(ValidationError):
    """A source URL or path was rejected by the SSRF or traversal guard."""

    status_code = HTTPStatus.BAD_REQUEST
    code = "unsafe_source"


class NotFoundError(RagError):
    """The requested resource does not exist, or is not visible to the caller."""

    status_code = HTTPStatus.NOT_FOUND
    code = "not_found"


class AuthorizationError(RagError):
    """The caller is not permitted to access the resource."""

    status_code = HTTPStatus.FORBIDDEN
    code = "forbidden"


class ProviderError(RagError):
    """An upstream model provider failed."""

    status_code = HTTPStatus.BAD_GATEWAY
    code = "provider_error"


class ProviderTimeoutError(ProviderError):
    """An upstream model provider did not respond within the configured budget."""

    status_code = HTTPStatus.GATEWAY_TIMEOUT
    code = "provider_timeout"


class ProviderUnavailableError(ProviderError):
    """An upstream model provider is not reachable or not configured."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "provider_unavailable"


class IngestionError(RagError):
    """A document could not be parsed or indexed."""

    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "ingestion_failed"


class RetrievalError(RagError):
    """Retrieval could not be completed."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "retrieval_failed"


class PolicyViolationError(RagError):
    """A request was refused by a security policy."""

    status_code = HTTPStatus.FORBIDDEN
    code = "policy_violation"


__all__ = [
    "AuthorizationError",
    "ConfigurationError",
    "DocumentTooLargeError",
    "IngestionError",
    "NotFoundError",
    "PolicyViolationError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RagError",
    "RetrievalError",
    "UnsafeSourceError",
    "UnsupportedMediaTypeError",
    "ValidationError",
]
