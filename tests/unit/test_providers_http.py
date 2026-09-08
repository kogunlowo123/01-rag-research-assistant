"""Outbound provider transport, retries, error mapping and wire formats."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from rag_assistant.domain.models import TrustLevel
from rag_assistant.errors import (
    ConfigurationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from rag_assistant.providers.base import GenerationRequest, PromptSegment
from rag_assistant.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from rag_assistant.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from rag_assistant.providers.transport import ProviderTransport

pytestmark = pytest.mark.unit


def transport_with(handler: Any, **kwargs: Any) -> ProviderTransport:
    return ProviderTransport(
        base_url="http://provider.invalid",
        provider_name="test-provider",
        timeout_seconds=1.0,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://provider.invalid"
        ),
        **kwargs,
    )


def prompt(question: str = "How long is the refund window?") -> GenerationRequest:
    return GenerationRequest(
        segments=(
            PromptSegment(trust=TrustLevel.SYSTEM, content="Policy text.", label="policy"),
            PromptSegment(
                trust=TrustLevel.UNTRUSTED,
                content="Refunds are available for 30 days.",
                label="1",
                header="marker=1 | document=policy.md",
                context="policy.md Refunds",
            ),
            PromptSegment(trust=TrustLevel.USER, content=f"Question: {question}", label="question"),
        ),
        fence_open="<<<EV-abc",
        fence_close="EV-abc>>>",
    )


class TestTransportRetries:
    async def test_a_retryable_status_is_retried_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        transport = transport_with(flaky, max_retries=3)
        assert await transport.post_json("/x", {}) == {"ok": True}
        assert calls["n"] == 3
        await transport.aclose()

    async def test_a_client_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def bad_request(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad"})

        transport = transport_with(bad_request, max_retries=3)
        with pytest.raises(ProviderError):
            await transport.post_json("/x", {})
        assert calls["n"] == 1
        await transport.aclose()

    async def test_the_upstream_body_is_never_included_in_the_error(self) -> None:
        """Providers echo the request in errors, and the request holds user data."""

        def leaky(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "your prompt was: SECRET-USER-CONTENT"})

        transport = transport_with(leaky, max_retries=0)
        with pytest.raises(ProviderError) as raised:
            await transport.post_json("/x", {})
        assert "SECRET-USER-CONTENT" not in str(raised.value)
        assert "SECRET-USER-CONTENT" not in repr(raised.value.detail)
        await transport.aclose()

    async def test_a_timeout_becomes_a_domain_timeout_error(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        transport = transport_with(timeout, max_retries=1)
        with pytest.raises(ProviderTimeoutError):
            await transport.post_json("/x", {})
        await transport.aclose()

    async def test_a_connection_failure_becomes_provider_unavailable(self) -> None:
        def refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        transport = transport_with(refused, max_retries=1)
        with pytest.raises(ProviderUnavailableError):
            await transport.post_json("/x", {})
        await transport.aclose()

    async def test_a_non_json_body_is_rejected(self) -> None:
        transport = transport_with(lambda r: httpx.Response(200, content=b"<html>not json"))
        with pytest.raises(ProviderError, match="non-JSON"):
            await transport.post_json("/x", {})
        await transport.aclose()

    async def test_a_json_array_is_rejected(self) -> None:
        transport = transport_with(lambda r: httpx.Response(200, json=[1, 2, 3]))
        with pytest.raises(ProviderError, match="unexpected JSON shape"):
            await transport.post_json("/x", {})
        await transport.aclose()

    async def test_retries_are_exhausted_and_then_reported(self) -> None:
        transport = transport_with(lambda r: httpx.Response(503), max_retries=2)
        with pytest.raises(ProviderError):
            await transport.post_json("/x", {})
        await transport.aclose()

    async def test_health_check_reports_reachability(self) -> None:
        healthy = transport_with(lambda r: httpx.Response(200, json={}))
        assert await healthy.get_ok("/health") is True
        await healthy.aclose()

        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        unhealthy = transport_with(down)
        assert await unhealthy.get_ok("/health") is False
        await unhealthy.aclose()

    async def test_the_context_manager_closes_the_client(self) -> None:
        async with transport_with(lambda r: httpx.Response(200, json={})) as transport:
            assert await transport.post_json("/x", {}) == {}


class TestOllamaEmbeddings:
    def _provider(self, handler: Any, dimensions: int = 4) -> OllamaEmbeddingProvider:
        return OllamaEmbeddingProvider(
            base_url="http://ollama.invalid",
            model="nomic-embed-text",
            dimensions=dimensions,
            transport=transport_with(handler),
        )

    async def test_documents_and_queries_are_embedded(self) -> None:
        def embed(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})

        provider = self._provider(embed)
        assert provider.dimensions == 4
        assert provider.name == "ollama:nomic-embed-text"
        assert await provider.embed_query("q") == pytest.approx([0.1, 0.2, 0.3, 0.4])
        await provider.aclose()

    async def test_an_empty_batch_makes_no_request(self) -> None:
        calls: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"embeddings": []})

        provider = self._provider(record)
        assert await provider.embed_documents([]) == []
        assert calls == []
        await provider.aclose()

    async def test_a_dimension_mismatch_is_refused_rather_than_indexed(self) -> None:
        """Mixing widths silently corrupts an index, so it must fail loudly."""

        def wrong_width(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

        provider = self._provider(wrong_width, dimensions=4)
        with pytest.raises(ProviderError, match="width"):
            await provider.embed_query("q")
        await provider.aclose()

    async def test_a_short_response_is_refused(self) -> None:
        def short(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": []})

        provider = self._provider(short)
        with pytest.raises(ProviderError, match="unexpected number"):
            await provider.embed_documents(["a", "b"])
        await provider.aclose()


class TestOllamaChat:
    async def test_the_request_maps_trust_levels_onto_roles(self) -> None:
        captured: dict[str, Any] = {}

        def chat(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2:3b",
                    "message": {"content": "Refunds last 30 days. [1]"},
                    "prompt_eval_count": 120,
                    "eval_count": 9,
                    "done_reason": "stop",
                },
            )

        provider = OllamaChatProvider(
            base_url="http://ollama.invalid", model="llama3.2:3b", transport=transport_with(chat)
        )
        response = await provider.generate(prompt())

        roles = [message["role"] for message in captured["messages"]]
        assert roles == ["system", "user"]
        system = captured["messages"][0]["content"]
        user = captured["messages"][1]["content"]
        assert "Refunds are available for 30 days." not in system
        assert "Refunds are available for 30 days." in user
        assert response.text.startswith("Refunds last 30 days.")
        assert response.prompt_tokens == 120
        assert response.completion_tokens == 9
        assert response.provider == "ollama"
        await provider.aclose()

    async def test_the_evidence_fence_is_applied_exactly_once(self) -> None:
        captured: dict[str, Any] = {}

        def chat(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}})

        provider = OllamaChatProvider(
            base_url="http://o.invalid", model="m", transport=transport_with(chat)
        )
        await provider.generate(prompt())
        user = captured["messages"][1]["content"]
        assert user.count("<<<EV-abc") == 1
        assert user.count("EV-abc>>>") == 1
        await provider.aclose()

    async def test_a_response_without_content_is_refused(self) -> None:
        provider = OllamaChatProvider(
            base_url="http://o.invalid",
            model="m",
            transport=transport_with(lambda r: httpx.Response(200, json={"done": True})),
        )
        with pytest.raises(ProviderError, match="without message content"):
            await provider.generate(prompt())
        await provider.aclose()

    async def test_health_is_delegated_to_the_tag_listing(self) -> None:
        provider = OllamaChatProvider(
            base_url="http://o.invalid",
            model="m",
            transport=transport_with(lambda r: httpx.Response(200, json={"models": []})),
        )
        assert await provider.health() is True
        await provider.aclose()


class TestOpenAiProvider:
    def test_a_missing_key_is_a_configuration_error_not_a_runtime_one(self) -> None:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            OpenAIChatProvider(base_url="http://x", api_key=None, model="gpt")

    async def test_embeddings_are_reordered_by_the_returned_index(self) -> None:
        """The API documents that results may arrive out of order."""

        def embed(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [1.0, 1.0]},
                        {"index": 0, "embedding": [0.0, 0.0]},
                    ]
                },
            )

        provider = OpenAIEmbeddingProvider(
            base_url="http://api.invalid",
            api_key=SecretStr("test-key"),
            model="text-embedding-3-small",
            dimensions=2,
            transport=transport_with(embed),
        )
        vectors = await provider.embed_documents(["first", "second"])
        assert vectors[0] == [0.0, 0.0]
        assert vectors[1] == [1.0, 1.0]
        await provider.aclose()

    async def test_batching_respects_the_configured_size(self) -> None:
        batches: list[int] = []

        def embed(request: httpx.Request) -> httpx.Response:
            import json

            inputs = json.loads(request.content)["input"]
            batches.append(len(inputs))
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [0.5]} for i in range(len(inputs))]},
            )

        provider = OpenAIEmbeddingProvider(
            base_url="http://api.invalid",
            api_key=SecretStr("test-key"),
            model="m",
            dimensions=1,
            batch_size=2,
            transport=transport_with(embed),
        )
        await provider.embed_documents(["a", "b", "c", "d", "e"])
        assert batches == [2, 2, 1]
        await provider.aclose()

    async def test_chat_completion_is_parsed_with_usage(self) -> None:
        def chat(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-test",
                    "choices": [
                        {
                            "message": {"content": "The window is 30 days. [1]"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 12},
                },
            )

        provider = OpenAIChatProvider(
            base_url="http://api.invalid",
            api_key=SecretStr("test-key"),
            model="gpt-test",
            transport=transport_with(chat),
        )
        response = await provider.generate(prompt())
        assert response.prompt_tokens == 200
        assert response.completion_tokens == 12
        assert response.finish_reason == "stop"
        await provider.aclose()

    async def test_a_response_with_no_choices_is_refused(self) -> None:
        provider = OpenAIChatProvider(
            base_url="http://api.invalid",
            api_key=SecretStr("test-key"),
            model="m",
            transport=transport_with(lambda r: httpx.Response(200, json={"choices": []})),
        )
        with pytest.raises(ProviderError, match="no choices"):
            await provider.generate(prompt())
        await provider.aclose()

    async def test_a_missing_usage_block_defaults_to_zero(self) -> None:
        def chat(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

        provider = OpenAIChatProvider(
            base_url="http://api.invalid",
            api_key=SecretStr("test-key"),
            model="m",
            transport=transport_with(chat),
        )
        response = await provider.generate(prompt())
        assert response.prompt_tokens == 0
        await provider.aclose()

    async def test_the_api_key_is_sent_as_a_bearer_header(self) -> None:
        seen: dict[str, str] = {}

        def capture(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

        provider = OpenAIEmbeddingProvider(
            base_url="http://api.invalid",
            api_key=SecretStr("test-key"),
            model="m",
            dimensions=1,
            transport=ProviderTransport(
                base_url="http://api.invalid",
                provider_name="openai-embeddings",
                timeout_seconds=1.0,
                headers={"authorization": "Bearer test-key"},
                client=httpx.AsyncClient(
                    transport=httpx.MockTransport(capture),
                    base_url="http://api.invalid",
                    headers={"authorization": "Bearer test-key"},
                ),
            ),
        )
        await provider.embed_query("q")
        assert seen["authorization"] == "Bearer test-key"
        await provider.aclose()
