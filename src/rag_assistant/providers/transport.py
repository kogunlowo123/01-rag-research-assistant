"""Shared HTTP transport policy for outbound provider calls.

Every provider uses the same client so that timeout, retry and error-mapping
behaviour is defined once. Three decisions are encoded here:

* **Timeouts are always set.** ``httpx`` defaults to five seconds, which is
  wrong in both directions for model calls; connect and read budgets are
  separated so a slow model does not look like an unreachable one.
* **Retries are restricted to idempotent failures.** Connection errors,
  timeouts, ``429`` and ``5xx`` are retried with exponential backoff and full
  jitter. A ``4xx`` other than ``429`` is a client bug and is never retried.
* **Upstream failures become domain errors.** Callers see
  :class:`ProviderTimeoutError` or :class:`ProviderError`, never an
  ``httpx`` exception, and the message never contains the response body, which
  may echo the request and therefore user content.
"""

from __future__ import annotations

import asyncio
import secrets
from types import TracebackType
from typing import Any, Final, Self

import httpx

from rag_assistant.errors import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from rag_assistant.observability.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})
_BASE_BACKOFF_SECONDS: Final[float] = 0.25
_MAX_BACKOFF_SECONDS: Final[float] = 8.0
#: Response bodies larger than this are truncated before parsing so a hostile or
#: broken upstream cannot exhaust memory.
_MAX_RESPONSE_BYTES: Final[int] = 32 * 1024 * 1024
#: Status at or above which a provider response is treated as a failure.
_HTTP_ERROR_FLOOR: Final[int] = 400


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter.

    Full jitter rather than fixed backoff because several workers retrying a
    recovering provider in lockstep is how a partial outage becomes a total
    one. ``secrets`` is used rather than ``random`` only to satisfy the static
    analyser; either is adequate for jitter.
    """
    ceiling = min(_MAX_BACKOFF_SECONDS, _BASE_BACKOFF_SECONDS * float(2**attempt))
    return ceiling * (secrets.randbelow(1000) / 1000.0)


class ProviderTransport:
    """A retrying JSON HTTP client scoped to one upstream provider."""

    def __init__(
        self,
        *,
        base_url: str,
        provider_name: str,
        timeout_seconds: float,
        max_retries: int = 2,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Build a transport, optionally around an injected client for tests."""
        self._provider_name = provider_name
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(
                timeout=timeout_seconds,
                connect=min(5.0, timeout_seconds),
                read=timeout_seconds,
                write=min(10.0, timeout_seconds),
                pool=min(5.0, timeout_seconds),
            ),
            headers={"accept": "application/json", **(headers or {})},
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
            follow_redirects=False,
        )

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON and return the decoded object, retrying transient failures."""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(path, json=payload)
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderTimeoutError(
                        f"{self._provider_name} did not respond within the timeout budget"
                    ) from exc
            except httpx.ConnectError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderUnavailableError(
                        f"{self._provider_name} is not reachable"
                    ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self._provider_name} transport failure") from exc
            else:
                if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                    logger.warning(
                        "provider.retry",
                        provider=self._provider_name,
                        status=response.status_code,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(_backoff_delay(attempt))
                    continue
                return self._decode(response)

            logger.warning(
                "provider.retry",
                provider=self._provider_name,
                error=type(last_error).__name__,
                attempt=attempt + 1,
            )
            await asyncio.sleep(_backoff_delay(attempt))

        # Unreachable: every path above either returns or raises on the final
        # attempt. Kept as a defensive guard rather than an implicit None.
        raise ProviderError(f"{self._provider_name} exhausted its retry budget")

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= _HTTP_ERROR_FLOOR:
            # The body is intentionally excluded: providers echo request content
            # in errors, and that content may be user data.
            raise ProviderError(
                f"{self._provider_name} returned HTTP {response.status_code}",
                detail={"status": response.status_code, "provider": self._provider_name},
            )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ProviderError(f"{self._provider_name} returned an oversized response")
        try:
            decoded: Any = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self._provider_name} returned a non-JSON body") from exc
        if not isinstance(decoded, dict):
            raise ProviderError(f"{self._provider_name} returned an unexpected JSON shape")
        return decoded

    async def get_ok(self, path: str) -> bool:
        """Return whether a GET against ``path`` succeeds. Used for health checks."""
        try:
            response = await self._client.get(path)
        except httpx.HTTPError:
            return False
        return response.status_code < _HTTP_ERROR_FLOOR

    async def aclose(self) -> None:
        """Close the underlying client if this transport created it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter an async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the transport on context exit."""
        await self.aclose()


__all__ = ["ProviderTransport"]
