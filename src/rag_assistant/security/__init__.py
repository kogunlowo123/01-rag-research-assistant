"""Security controls: authentication, authorisation, source validation and injection defence."""

from rag_assistant.security.authz import (
    ApiKeyAuthenticator,
    Principal,
    can_read_document,
    require_read,
)
from rag_assistant.security.injection import (
    NEUTRALISED_MARKER,
    ScanResult,
    aggregate_risk,
    context_dilution_score,
    neutralise,
    scan,
)
from rag_assistant.security.normalization import normalize, strip_control_characters
from rag_assistant.security.sources import (
    safe_filename,
    sniff_media_type,
    validate_upload,
    validate_url,
)

__all__ = [
    "NEUTRALISED_MARKER",
    "ApiKeyAuthenticator",
    "Principal",
    "ScanResult",
    "aggregate_risk",
    "can_read_document",
    "context_dilution_score",
    "neutralise",
    "normalize",
    "require_read",
    "safe_filename",
    "scan",
    "sniff_media_type",
    "strip_control_characters",
    "validate_upload",
    "validate_url",
]
