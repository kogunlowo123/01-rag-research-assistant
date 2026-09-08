"""Loading raw document bytes from the supported sources.

Two sources exist: bytes supplied directly by an authenticated caller, and
bytes fetched from a URL. The second is disabled by default because it turns
the service into an HTTP client that an attacker partially controls.

The fetcher enforces its limits during streaming rather than after. Checking
``Content-Length`` alone is insufficient: it is a hint, it can be absent, and it
can understate the body. The download is aborted the moment the accumulated
size exceeds the budget, so a decompression bomb or an endless stream cannot
exhaust memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from rag_assistant.errors import (
    DocumentTooLargeError,
    ProviderTimeoutError,
    UnsafeSourceError,
)
from rag_assistant.security.sources import validate_url

_HTTP_ERROR_FLOOR = 400


@dataclass(frozen=True, slots=True)
class LoadedBytes:
    """Raw document bytes plus the media type the source claimed."""

    payload: bytes
    declared_media_type: str
    source_ref: str


class UrlLoader:
    """Fetches documents over HTTP under an explicit SSRF policy."""

    def __init__(
        self,
        *,
        allowed_schemes: frozenset[str],
        allowed_hosts: frozenset[str],
        max_bytes: int,
        timeout_seconds: float,
        allow_private_network: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the fetch policy and optionally inject a client for tests."""
        self._allowed_schemes = allowed_schemes
        self._allowed_hosts = allowed_hosts
        self._max_bytes = max_bytes
        self._allow_private_network = allow_private_network
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
            # Redirects are refused rather than followed: an allowlisted host
            # that redirects to an internal address would otherwise defeat the
            # entire SSRF policy.
            follow_redirects=False,
            headers={"user-agent": "rag-research-assistant/1.0 (+document-ingestion)"},
        )

    async def load(self, url: str) -> LoadedBytes:
        """Fetch a document, enforcing the size budget while streaming."""
        canonical = validate_url(
            url,
            allowed_schemes=self._allowed_schemes,
            allowed_hosts=self._allowed_hosts,
            allow_private_network=self._allow_private_network,
        )

        buffer = bytearray()
        try:
            async with self._client.stream("GET", canonical) as response:
                if response.is_redirect:
                    raise UnsafeSourceError("the source URL redirected; redirects are not followed")
                if response.status_code >= _HTTP_ERROR_FLOOR:
                    raise UnsafeSourceError(
                        "the source URL returned an error status",
                        detail={"status": response.status_code},
                    )

                declared = response.headers.get("content-type", "application/octet-stream")
                async for piece in response.aiter_bytes():
                    buffer.extend(piece)
                    if len(buffer) > self._max_bytes:
                        raise DocumentTooLargeError(
                            "the fetched document exceeds the configured size limit",
                            detail={"limit_bytes": self._max_bytes},
                        )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("the source URL did not respond in time") from exc
        except httpx.HTTPError as exc:
            raise UnsafeSourceError("the source URL could not be fetched") from exc

        return LoadedBytes(
            payload=bytes(buffer),
            declared_media_type=declared.split(";")[0].strip().lower(),
            source_ref=canonical,
        )

    async def aclose(self) -> None:
        """Close the underlying client if this loader created it."""
        if self._owns_client:
            await self._client.aclose()


__all__ = ["LoadedBytes", "UrlLoader"]
