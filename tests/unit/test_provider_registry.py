"""Provider selection and the degraded-mode fallback."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from rag_assistant.config import (
    ChatBackend,
    ChatSettings,
    EmbeddingBackend,
    EmbeddingSettings,
    ProviderEndpoints,
    Settings,
)
from rag_assistant.domain.models import TrustLevel
from rag_assistant.errors import ConfigurationError, ProviderUnavailableError
from rag_assistant.providers.base import (
    GenerationRequest,
    GenerationResponse,
    PromptSegment,
)
from rag_assistant.providers.extractive import ExtractiveChatProvider
from rag_assistant.providers.hashing import HashingEmbeddingProvider
from rag_assistant.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from rag_assistant.providers.openai import OpenAIEmbeddingProvider
from rag_assistant.providers.registry import (
    FallbackChatProvider,
    build_chat_provider,
    build_embedding_provider,
    close_all,
)

pytestmark = pytest.mark.unit


def request_with_evidence() -> GenerationRequest:
    return GenerationRequest(
        segments=(
            PromptSegment(trust=TrustLevel.SYSTEM, content="Policy.", label="policy"),
            PromptSegment(
                trust=TrustLevel.UNTRUSTED,
                content="Refunds are available for thirty days after purchase.",
                label="1",
                header="marker=1",
                context="policy.md",
            ),
            PromptSegment(
                trust=TrustLevel.USER, content="Question: refund window", label="question"
            ),
        )
    )


class BrokenProvider:
    """A chat provider that always fails, standing in for an unreachable model."""

    name = "broken"
    model = "broken-model"

    def __init__(self) -> None:
        """Track whether the provider was closed."""
        self.closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise ProviderUnavailableError("the model is not installed")

    async def health(self) -> bool:
        return False

    async def aclose(self) -> None:
        self.closed = True


class TestEmbeddingProviderSelection:
    def _settings(self, backend: EmbeddingBackend, **extra: object) -> Settings:
        return Settings(
            embedding=EmbeddingSettings(backend=backend, dimensions=8),
            providers=ProviderEndpoints(**extra),
        )

    def test_hashing_is_selected(self) -> None:
        provider = build_embedding_provider(self._settings(EmbeddingBackend.HASHING))
        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.dimensions == 8

    def test_ollama_is_selected(self) -> None:
        provider = build_embedding_provider(self._settings(EmbeddingBackend.OLLAMA))
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_openai_is_selected_when_a_key_is_present(self) -> None:
        provider = build_embedding_provider(
            self._settings(EmbeddingBackend.OPENAI, openai_api_key=SecretStr("k"))
        )
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_openai_without_a_key_fails_at_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            build_embedding_provider(self._settings(EmbeddingBackend.OPENAI))

    def test_fastembed_reports_a_clear_error_when_the_extra_is_absent(self) -> None:
        """Selecting it must never fail with a bare ImportError at query time."""
        provider = build_embedding_provider(self._settings(EmbeddingBackend.FASTEMBED))
        assert provider.name.startswith("fastembed:")


class TestChatProviderSelection:
    def test_extractive_is_returned_unwrapped(self) -> None:
        settings = Settings(chat=ChatSettings(backend=ChatBackend.EXTRACTIVE))
        assert isinstance(build_chat_provider(settings), ExtractiveChatProvider)

    def test_ollama_is_wrapped_in_the_fallback(self) -> None:
        settings = Settings(chat=ChatSettings(backend=ChatBackend.OLLAMA))
        provider = build_chat_provider(settings)
        assert isinstance(provider, FallbackChatProvider)
        assert provider.name == "ollama"

    def test_openai_is_wrapped_in_the_fallback(self) -> None:
        settings = Settings(
            chat=ChatSettings(backend=ChatBackend.OPENAI),
            providers=ProviderEndpoints(openai_api_key=SecretStr("k")),
        )
        assert isinstance(build_chat_provider(settings), FallbackChatProvider)

    def test_the_wrapped_primary_is_the_configured_one(self) -> None:
        settings = Settings(chat=ChatSettings(backend=ChatBackend.OLLAMA, model="llama3.2:3b"))
        provider = build_chat_provider(settings)
        assert provider.model == "llama3.2:3b"
        assert isinstance(provider, FallbackChatProvider)
        assert isinstance(provider._primary, OllamaChatProvider)

    def test_openai_chat_without_a_key_fails_at_construction(self) -> None:
        settings = Settings(chat=ChatSettings(backend=ChatBackend.OPENAI))
        with pytest.raises(ConfigurationError):
            build_chat_provider(settings)


class TestFallbackBehaviour:
    async def test_an_unavailable_model_degrades_to_extractive(self) -> None:
        provider = FallbackChatProvider(
            primary=BrokenProvider(), fallback=ExtractiveChatProvider(max_sentences=2)
        )
        response = await provider.generate(request_with_evidence())
        assert "thirty days" in response.text
        assert response.finish_reason == "degraded_extractive"

    async def test_the_degradation_is_visible_rather_than_silent(self) -> None:
        """A quietly degraded answer is indistinguishable from a good one."""
        provider = FallbackChatProvider(primary=BrokenProvider(), fallback=ExtractiveChatProvider())
        response = await provider.generate(request_with_evidence())
        assert response.metadata["degraded"] == "true"
        assert response.metadata["degraded_from"] == "broken"
        assert response.metadata["degraded_reason"] == "provider_unavailable"

    async def test_health_reflects_the_primary_not_the_fallback(self) -> None:
        provider = FallbackChatProvider(primary=BrokenProvider(), fallback=ExtractiveChatProvider())
        assert await provider.health() is False

    async def test_a_working_primary_is_used_unchanged(self) -> None:
        class Working:
            name = "working"
            model = "m"

            async def generate(self, request: GenerationRequest) -> GenerationResponse:
                return GenerationResponse(text="primary answer", model="m", provider="working")

            async def health(self) -> bool:
                return True

            async def aclose(self) -> None:
                return None

        provider = FallbackChatProvider(primary=Working(), fallback=ExtractiveChatProvider())
        response = await provider.generate(request_with_evidence())
        assert response.text == "primary answer"
        assert response.metadata == {}

    async def test_closing_closes_both_providers(self) -> None:
        primary = BrokenProvider()
        provider = FallbackChatProvider(primary=primary, fallback=ExtractiveChatProvider())
        await provider.aclose()
        assert primary.closed is True


class TestCloseAll:
    async def test_providers_exposing_aclose_are_closed(self) -> None:
        primary = BrokenProvider()
        await close_all([primary, object()])
        assert primary.closed is True


class TestHashingProvider:
    async def test_embeddings_are_deterministic_across_instances(self) -> None:
        first = await HashingEmbeddingProvider(dimensions=64).embed_query("refund window")
        second = await HashingEmbeddingProvider(dimensions=64).embed_query("refund window")
        assert first == second

    async def test_lexically_similar_text_scores_higher_than_unrelated_text(self) -> None:
        provider = HashingEmbeddingProvider(dimensions=256)
        query = await provider.embed_query("refund window thirty days")
        related, unrelated = await provider.embed_documents(
            [
                "Customers may request a refund within thirty days.",
                "The mitochondrion is the powerhouse of the cell.",
            ]
        )
        dot = sum(a * b for a, b in zip(query, related, strict=True))
        other = sum(a * b for a, b in zip(query, unrelated, strict=True))
        assert dot > other

    async def test_empty_text_yields_a_zero_vector(self) -> None:
        vector = await HashingEmbeddingProvider(dimensions=32).embed_query("")
        assert vector == [0.0] * 32

    def test_a_tiny_dimension_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            HashingEmbeddingProvider(dimensions=2)

    def test_the_quality_notice_names_the_alternative_backends(self) -> None:
        notice = HashingEmbeddingProvider.quality_notice()
        assert "fastembed" in notice
        assert "tests" in notice

    async def test_closing_is_a_no_op(self) -> None:
        await HashingEmbeddingProvider().aclose()
