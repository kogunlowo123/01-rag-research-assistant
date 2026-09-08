"""Provider construction and the degraded-mode fallback wrapper.

The rest of the application never imports a concrete provider. It asks this
module for something satisfying :class:`~rag_assistant.providers.base.ChatProvider`
or :class:`~rag_assistant.providers.base.EmbeddingProvider`, and configuration
decides what it gets.

:class:`FallbackChatProvider` implements the answer to "what happens when the
model is unavailable?". Rather than returning a 502 to the caller, it degrades
to extractive answering over the same retrieved evidence and marks the response
so the degradation is visible in the API payload and in telemetry. Degrading
silently would be worse than failing.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rag_assistant.config import ChatBackend, EmbeddingBackend, Settings
from rag_assistant.errors import ProviderError
from rag_assistant.observability.logging import get_logger
from rag_assistant.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    GenerationRequest,
    GenerationResponse,
)
from rag_assistant.providers.extractive import ExtractiveChatProvider
from rag_assistant.providers.hashing import HashingEmbeddingProvider
from rag_assistant.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from rag_assistant.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the configured embedding provider."""
    config = settings.embedding
    match config.backend:
        case EmbeddingBackend.HASHING:
            return HashingEmbeddingProvider(dimensions=config.dimensions)
        case EmbeddingBackend.FASTEMBED:
            from rag_assistant.providers.fastembed_provider import FastEmbedProvider

            return FastEmbedProvider(
                model=config.model,
                dimensions=config.dimensions,
                cache_dir=config.cache_dir,
                batch_size=config.batch_size,
            )
        case EmbeddingBackend.OLLAMA:
            return OllamaEmbeddingProvider(
                base_url=settings.providers.ollama_base_url,
                model=config.model,
                dimensions=config.dimensions,
                timeout_seconds=config.timeout_seconds,
            )
        case EmbeddingBackend.OPENAI:
            return OpenAIEmbeddingProvider(
                base_url=settings.providers.openai_base_url,
                api_key=settings.providers.openai_api_key,
                model=config.model,
                dimensions=config.dimensions,
                batch_size=config.batch_size,
                timeout_seconds=config.timeout_seconds,
            )


def build_chat_provider(settings: Settings) -> ChatProvider:
    """Construct the configured chat provider, wrapped in the extractive fallback."""
    config = settings.chat
    extractive = ExtractiveChatProvider(max_sentences=settings.generation.max_sentences)

    match config.backend:
        case ChatBackend.EXTRACTIVE:
            return extractive
        case ChatBackend.OLLAMA:
            primary: ChatProvider = OllamaChatProvider(
                base_url=settings.providers.ollama_base_url,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        case ChatBackend.OPENAI:
            primary = OpenAIChatProvider(
                base_url=settings.providers.openai_base_url,
                api_key=settings.providers.openai_api_key,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            )

    return FallbackChatProvider(primary=primary, fallback=extractive)


class FallbackChatProvider:
    """Wraps a primary provider and degrades to extractive answering on failure.

    The wrapper is not a retry loop: the transport already retries transient
    failures with backoff. This handles the case where retries were exhausted,
    the provider is misconfigured, or the model is simply not installed —
    situations where another attempt would not help but a lower-quality answer
    still would.
    """

    def __init__(self, *, primary: ChatProvider, fallback: ChatProvider) -> None:
        """Wire the primary provider and its degraded-mode replacement."""
        self._primary = primary
        self._fallback = fallback

    @property
    def name(self) -> str:
        """Identifier of the primary provider."""
        return self._primary.name

    @property
    def model(self) -> str:
        """Identifier of the primary model."""
        return self._primary.model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate with the primary provider, falling back on provider failure."""
        try:
            return await self._primary.generate(request)
        except ProviderError as exc:
            logger.warning(
                "provider.degraded",
                primary=self._primary.name,
                fallback=self._fallback.name,
                error_code=exc.code,
                reason=exc.message,
            )
            started = time.perf_counter()
            response = await self._fallback.generate(request)
            return GenerationResponse(
                text=response.text,
                model=response.model,
                provider=response.provider,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                finish_reason="degraded_extractive",
                metadata={
                    "degraded": "true",
                    "degraded_from": self._primary.name,
                    "degraded_reason": exc.code,
                },
            )

    async def health(self) -> bool:
        """Health of the primary provider. The fallback is always available."""
        return await self._primary.health()

    async def aclose(self) -> None:
        """Close both providers."""
        await self._primary.aclose()
        await self._fallback.aclose()


async def close_all(providers: Sequence[object]) -> None:
    """Close every provider that exposes ``aclose``. Used during shutdown."""
    for provider in providers:
        closer = getattr(provider, "aclose", None)
        if callable(closer):
            await closer()


__all__ = [
    "FallbackChatProvider",
    "build_chat_provider",
    "build_embedding_provider",
    "close_all",
]
