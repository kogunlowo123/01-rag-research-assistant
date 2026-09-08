"""Ollama provider: local models, no credentials, no per-token cost.

Ollama is the default *real* backend for this project because it is the only
way to demonstrate the full pipeline — including generation quality and
grounding behaviour — without a paid API key.

Trust levels are mapped onto the chat API deliberately. System-authored text
becomes the ``system`` message. Retrieved passages never do: they are placed in
the ``user`` message inside explicitly labelled fences, because a model that
receives untrusted text in the system role has been handed the application's
authority. See :mod:`rag_assistant.generation.prompt` for the fencing format.
"""

from __future__ import annotations

import time
from typing import Any

from rag_assistant.errors import ProviderError
from rag_assistant.providers.base import GenerationRequest, GenerationResponse
from rag_assistant.providers.transport import ProviderTransport

PROVIDER_NAME = "ollama"


class OllamaEmbeddingProvider:
    """Embeddings from a locally served Ollama model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 30.0,
        transport: ProviderTransport | None = None,
    ) -> None:
        """Configure the endpoint and expected vector width."""
        self._model = model
        self._dimensions = dimensions
        self._transport = transport or ProviderTransport(
            base_url=base_url,
            provider_name=f"{PROVIDER_NAME}-embeddings",
            timeout_seconds=timeout_seconds,
        )

    @property
    def name(self) -> str:
        """Provider identifier stored with every vector."""
        return f"{PROVIDER_NAME}:{self._model}"

    @property
    def dimensions(self) -> int:
        """Configured vector width."""
        return self._dimensions

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        payload = await self._transport.post_json(
            "/api/embed", {"model": self._model, "input": inputs}
        )
        raw = payload.get("embeddings")
        if not isinstance(raw, list) or len(raw) != len(inputs):
            raise ProviderError("ollama returned an unexpected number of embeddings")

        vectors: list[list[float]] = []
        for item in raw:
            if not isinstance(item, list):
                raise ProviderError("ollama returned a malformed embedding vector")
            vector = [float(value) for value in item]
            if len(vector) != self._dimensions:
                # Dimension drift silently corrupts an index: every vector after
                # a model change becomes incomparable with those before it.
                raise ProviderError(
                    "ollama embedding width does not match configuration",
                    detail={"expected": self._dimensions, "received": len(vector)},
                )
            vectors.append(vector)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        vectors = await self._embed([text])
        return vectors[0]

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._transport.aclose()


class OllamaChatProvider:
    """Answer generation from a locally served Ollama model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        transport: ProviderTransport | None = None,
    ) -> None:
        """Configure the endpoint and model."""
        self._model = model
        self._transport = transport or ProviderTransport(
            base_url=base_url,
            provider_name=PROVIDER_NAME,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
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

        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_output_tokens,
        }
        if request.stop:
            options["stop"] = list(request.stop)

        started = time.perf_counter()
        payload = await self._transport.post_json(
            "/api/chat",
            {"model": self._model, "messages": messages, "stream": False, "options": options},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("ollama returned a response without message content")

        return GenerationResponse(
            text=message["content"],
            model=str(payload.get("model", self._model)),
            provider=PROVIDER_NAME,
            prompt_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(payload.get("eval_count", 0) or 0),
            latency_ms=elapsed_ms,
            finish_reason=str(payload.get("done_reason", "stop")),
        )

    async def health(self) -> bool:
        """Whether the Ollama server responds to a model listing."""
        return await self._transport.get_ok("/api/tags")

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._transport.aclose()


__all__ = ["PROVIDER_NAME", "OllamaChatProvider", "OllamaEmbeddingProvider"]
