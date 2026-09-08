"""Validation of everything that enters the ingestion pipeline.

Three attack surfaces are closed here.

**SSRF.** URL ingestion is opt-in, restricted to an explicit host allowlist, and
— critically — validated after DNS resolution rather than on the hostname
string. A hostname allowlist alone is defeated by a DNS record that resolves to
``169.254.169.254``; :func:`validate_url` therefore resolves the host and
rejects every address in private, loopback, link-local, multicast or reserved
space unless a test explicitly opts in. Redirects are not followed, because a
permitted host can redirect to a forbidden one.

**Path traversal.** Uploaded filenames are treated as untrusted labels, never
as paths. :func:`safe_filename` strips directory components, drive letters,
NTFS alternate data streams and reserved Windows device names, and the result is
used only for display; stored objects are named by content digest.

**Content confusion.** A declared media type is a claim by the uploader.
:func:`sniff_media_type` reads the leading bytes and refuses a document whose
signature contradicts its declared type, so a PDF parser is never handed a ZIP.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final
from urllib.parse import urlsplit

from rag_assistant.errors import (
    DocumentTooLargeError,
    UnsafeSourceError,
    UnsupportedMediaTypeError,
)

#: Magic-byte signatures for the media types this application accepts.
_SIGNATURES: Final[tuple[tuple[bytes, str], ...]] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"PK\x03\x04", "application/zip"),
    (b"\x1f\x8b", "application/gzip"),
    (b"\x7fELF", "application/x-executable"),
    (b"MZ", "application/x-msdownload"),
    (b"\xca\xfe\xba\xbe", "application/java-vm"),
)

#: Signatures that are never acceptable regardless of the configured allowlist:
#: archives and executables have no place in a text corpus and are a classic
#: route to parser exploitation and zip-bomb denial of service.
_ALWAYS_REJECTED: Final[frozenset[str]] = frozenset(
    {
        "application/zip",
        "application/gzip",
        "application/x-executable",
        "application/x-msdownload",
        "application/java-vm",
    }
)

_RESERVED_WINDOWS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)

_UNSAFE_FILENAME_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._ -]")
_MAX_FILENAME_LENGTH: Final[int] = 120
_SNIFF_BYTES: Final[int] = 16


def safe_filename(raw: str, *, fallback: str = "document") -> str:
    """Reduce an uploaded filename to a safe display label.

    The result is never used to build a filesystem path — stored content is
    addressed by digest — but it is rendered in citations and API responses, so
    it must not carry traversal sequences, control characters or bidirectional
    overrides that could misrepresent the extension to a human reviewer.
    """
    candidate = unicodedata.normalize("NFKC", raw).strip()
    # Take the final component under both path grammars: a POSIX server must not
    # be fooled by a Windows-style path and vice versa.
    candidate = PureWindowsPath(PurePosixPath(candidate).name).name
    # NTFS alternate data stream suffix.
    candidate = candidate.split(":")[0]
    candidate = _UNSAFE_FILENAME_CHARS.sub("_", candidate).strip(" .")

    if not candidate:
        return fallback
    stem, dot, suffix = candidate.rpartition(".")
    if dot and stem.lower() in _RESERVED_WINDOWS_NAMES:
        candidate = f"{stem}_{dot}{suffix}"
    elif not dot and candidate.lower() in _RESERVED_WINDOWS_NAMES:
        candidate = f"{candidate}_"

    if len(candidate) > _MAX_FILENAME_LENGTH:
        stem, dot, suffix = candidate.rpartition(".")
        keep = _MAX_FILENAME_LENGTH - len(suffix) - 1 if dot else _MAX_FILENAME_LENGTH
        candidate = f"{stem[:keep]}{dot}{suffix}" if dot else candidate[:_MAX_FILENAME_LENGTH]
    return candidate or fallback


def sniff_media_type(payload: bytes, declared: str) -> str:
    """Resolve the effective media type, refusing declarations the bytes contradict.

    Returns the declared type when the payload has no recognised signature —
    plain text and Markdown legitimately have none. Raises when the signature
    identifies a forbidden format, or contradicts a declared binary format.
    """
    head = payload[:_SNIFF_BYTES]
    detected: str | None = None
    for signature, media_type in _SIGNATURES:
        if head.startswith(signature):
            detected = media_type
            break

    if detected in _ALWAYS_REJECTED:
        raise UnsupportedMediaTypeError(
            "the uploaded content is an archive or executable, which cannot be ingested",
            detail={"detected": detected},
        )

    if detected is None:
        if declared == "application/pdf":
            raise UnsupportedMediaTypeError(
                "content declared as application/pdf does not begin with a PDF signature"
            )
        return declared

    if declared != detected and declared not in {"application/octet-stream", ""}:
        raise UnsupportedMediaTypeError(
            "the declared media type does not match the content signature",
            detail={"declared": declared, "detected": detected},
        )
    return detected


def validate_upload(
    payload: bytes,
    *,
    declared_media_type: str,
    allowed_media_types: frozenset[str],
    max_bytes: int,
) -> str:
    """Validate an uploaded document and return its effective media type."""
    if not payload:
        raise UnsupportedMediaTypeError("the uploaded document is empty")
    if len(payload) > max_bytes:
        raise DocumentTooLargeError(
            "the uploaded document exceeds the configured size limit",
            detail={"limit_bytes": max_bytes, "received_bytes": len(payload)},
        )

    normalised = declared_media_type.split(";", maxsplit=1)[0].strip().lower() or "text/plain"
    effective = sniff_media_type(payload, normalised)
    if effective not in allowed_media_types:
        raise UnsupportedMediaTypeError(
            "this media type is not in the ingestion allowlist",
            detail={"media_type": effective, "allowed": sorted(allowed_media_types)},
        )
    return effective


def _is_forbidden_address(address: str) -> bool:
    """Whether an IP literal falls in address space the fetcher must never reach."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return bool(
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def resolve_host(host: str) -> list[str]:
    """Resolve a hostname to every address it maps to.

    Every address is checked, not just the first: a hostname with both a public
    and a private A record would otherwise pass validation and then be connected
    to on the private one.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeSourceError("the source hostname could not be resolved") from exc
    # sockaddr is (host, port) for IPv4 and (host, port, flowinfo, scope) for
    # IPv6; the first element is the address string in both cases.
    return sorted({str(info[4][0]) for info in infos})


def validate_url(
    url: str,
    *,
    allowed_schemes: frozenset[str],
    allowed_hosts: frozenset[str],
    allow_private_network: bool = False,
) -> str:
    """Validate a source URL against the SSRF policy and return its canonical form.

    Validation order matters: scheme and host allowlists are cheap and are
    checked first, so a rejected request never triggers a DNS lookup that an
    attacker could use as an out-of-band signal.
    """
    parsed = urlsplit(url.strip())

    if parsed.scheme.lower() not in allowed_schemes:
        raise UnsafeSourceError(
            "the URL scheme is not permitted",
            detail={"allowed_schemes": sorted(allowed_schemes)},
        )
    if parsed.username or parsed.password:
        raise UnsafeSourceError("credentials embedded in the URL are not permitted")

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeSourceError("the URL does not contain a host")
    if not allowed_hosts or host not in allowed_hosts:
        raise UnsafeSourceError(
            "the URL host is not in the ingestion allowlist",
            detail={"host": host},
        )
    if parsed.port is not None and parsed.port not in {80, 443}:
        raise UnsafeSourceError("only the standard HTTP and HTTPS ports may be fetched")

    if not allow_private_network:
        addresses = resolve_host(host)
        if not addresses:
            raise UnsafeSourceError("the source hostname resolved to no addresses")
        forbidden = [address for address in addresses if _is_forbidden_address(address)]
        if forbidden:
            # The offending address is not returned: it would confirm internal
            # topology to the caller who supplied the hostname.
            raise UnsafeSourceError("the URL resolves into a network range that cannot be fetched")

    return parsed.geturl()


__all__ = [
    "resolve_host",
    "safe_filename",
    "sniff_media_type",
    "validate_upload",
    "validate_url",
]
