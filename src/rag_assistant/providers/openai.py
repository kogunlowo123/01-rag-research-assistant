"""OpenAI-compatible provider.

Written against the OpenAI REST surface rather than the vendor SDK so that any
compatible endpoint — Azure OpenAI, vLLM, LiteLLM, Together, a local
llama.cpp server — works by changing ``RAG_PROVIDERS__OPENAI_BASE_URL``. The
SDK would add a dependency and a vendor coupling for functionality this module
already needs from :mod:`rag_assistant.providers.transport`.

The API key is held as a :class:`~pydantic.SecretStr` in configuration and is
only unwrapped when the ``Authorization`` header is built, so it cannot be
printed by an accidental ``repr`` of the settings object. The header name is on
the logging redaction list.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import SecretStr

from rag_assistant.errors import ConfigurationError, ProviderError
from rag_assistant.providers.base import GenerationRequest, GenerationResponse
from rag_assistant.providers.transport import ProviderTransport

PROVIDER_NAME = "openai"


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    if api_key is None:
        raise ConfigurationError(
            "the openai backend is selected but RAG_PROVIDERS__OPENAI_API_KEY is not set"
        )
    return {"authorization": f"Bearer {api_key.get_secret_value()}"}


class OpenAIEmbeddingProvider:
    """Embeddings from an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        dimensions: int,
        batch_size: int = 32,
        timeout_seconds: float = 30.0,
        transport: ProviderTransport | None = None,
    ) -> None:
        """Configure the endpoint, credentials and batching."""
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._transport = transport or ProviderTransport(
            base_url=base_url,
            provider_name=f"{PROVIDER_NAME}-embeddings",
            timeout_seconds=timeout_seconds,
            headers=_auth_headers(api_key),
        )

    @property
    def name(self) -> str:
        """Provider identifier stored with every vector."""
        return f"{PROVIDER_NAME}:{self._model}"

    @property
    def dimensions(self) -> int:
        """Configured vector width."""
        return self._dimensions

    async def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        payload = await self._transport.post_json(
            "/embeddings",
            {"model": self._model, "input": inputs, "dimensions": self._dimensions},
        )
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            raise ProviderError("openai returned an unexpected number of embeddings")

        # The API documents that results may arrive out of order, so sort by the
        # returned index rather than trusting position.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors: list[list[float]] = []
        for item in ordered:
            raw = item.get("embedding")
            if not isinstance(raw, list):
                raise ProviderError("openai returned a malformed embedding vector")
            vector = [float(value) for value in raw]
            if len(vector) != self._dimensions:
                raise ProviderError(
                    "openai embedding width does not match configuration",
                    detail={"expected": self._dimensions, "received": len(vector)},
                )
            vectors.append(vector)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing, respecting the configured batch size."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._transport.aclose()


class OpenAIChatProvider:
    """Answer generation from an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        transport: ProviderTransport | None = None,
    ) -> None:
        """Configure the endpoint, credentials and model."""
        self._model = model
        self._transport = transport or ProviderTransport(
            base_url=base_url,
            provider_name=PROVIDER_NAME,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            headers=_auth_headers(api_key),
        )

    @property
    def name(self) -> str:
        """Provider identifier recorded on every answer."""
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        """Model identifier recorded on every answer."""
        return self._model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate an answer, mapping trust levels onto chat roles."""
        messages: list[dict[str, str]] = []
        if system := request.system_text():
            messages.append({"role": "system", "content": system})

        user_parts: list[str] = []
        if evidence := request.render_untrusted():
            user_parts.append(evidence)
        if user := request.user_text():
            user_parts.append(user)
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_completion_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.stop:
            body["stop"] = list(request.stop)

        started = time.perf_counter()
        payload = await self._transport.post_json("/chat/completions", body)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("openai returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("openai returned a choice without message content")

        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return GenerationResponse(
            text=message["content"],
            model=str(payload.get("model", self._model)),
            provider=PROVIDER_NAME,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=elapsed_ms,
            finish_reason=str(choices[0].get("finish_reason", "stop")),
        )

    async def health(self) -> bool:
        """Whether the endpoint responds to a model listing."""
        return await self._transport.get_ok("/models")

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._transport.aclose()


__all__ = ["PROVIDER_NAME", "OpenAIChatProvider", "OpenAIEmbeddingProvider"]
