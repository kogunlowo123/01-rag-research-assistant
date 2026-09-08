"""Boundary controls: authentication, isolation, SSRF, uploads and log hygiene."""

from __future__ import annotations

import io
import logging
from typing import Any

import httpx
import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from rag_assistant.api.app import create_app
from rag_assistant.config import SecuritySettings, Settings
from rag_assistant.errors import ConfigurationError, DocumentTooLargeError, UnsafeSourceError
from rag_assistant.ingestion.loaders import UrlLoader
from rag_assistant.observability.logging import REDACTED, configure_logging, get_logger
from rag_assistant.security import sources

pytestmark = pytest.mark.security

MARKDOWN = b"# Title\n\nA sentence of perfectly ordinary body text for indexing.\n"
BIDI_OVERRIDE = chr(0x202E)


def with_keys(settings: Settings, *keys: str) -> Settings:
    """Rebuild the security block through validation so keys become SecretStr."""
    return settings.model_copy(
        update={
            "security": SecuritySettings(
                require_api_key=True,
                api_keys=keys,
                injection_action=settings.security.injection_action,
            )
        }
    )


def production(settings: Settings, **overrides: Any) -> Settings:
    """A production-shaped configuration backed by a real database URL."""
    return settings.model_copy(
        update={
            "environment": "production",
            "storage": settings.storage.model_copy(
                update={"database_url": "postgresql+psycopg://user:pass@host/db"}
            ),
            **overrides,
        }
    )


def problems(settings: Settings) -> list[str]:
    """Return the invariant violations the settings raise, as plain strings."""
    with pytest.raises(ConfigurationError) as raised:
        settings.enforce_environment_invariants()
    reported = raised.value.detail["problems"]
    assert isinstance(reported, list)
    return [str(problem) for problem in reported]


class TestTenantIsolationOverHttp:
    async def test_a_second_tenants_key_cannot_see_the_first_tenants_documents(
        self, settings: Settings
    ) -> None:
        app = create_app(with_keys(settings, "acme:acme-key-value", "globex:globex-key-value"))
        async with (
            AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client,
            app.router.lifespan_context(app),
        ):
            upload = await client.post(
                "/v1/documents",
                files={"file": ("a.md", MARKDOWN, "text/markdown")},
                headers={"X-API-Key": "acme-key-value"},
            )
            assert upload.status_code == 201
            document_id = upload.json()["document"]["id"]

            listing = await client.get("/v1/documents", headers={"X-API-Key": "globex-key-value"})
            direct = await client.get(
                f"/v1/documents/{document_id}", headers={"X-API-Key": "globex-key-value"}
            )
            query = await client.post(
                "/v1/query",
                json={"query": "ordinary body text for indexing"},
                headers={"X-API-Key": "globex-key-value"},
            )
            audit = await client.get("/v1/audit", headers={"X-API-Key": "globex-key-value"})

        assert listing.json()["total"] == 0
        assert direct.status_code == 404
        assert query.json()["refused"] is True

        # The second tenant sees its own query event and nothing referencing the
        # first tenant's document.
        events = audit.json()["events"]
        assert {event["event"] for event in events} == {"query.answer"}
        assert all(event["subject_id"] != document_id for event in events)

    async def test_a_missing_document_and_a_forbidden_one_are_indistinguishable(
        self, settings: Settings
    ) -> None:
        """Otherwise the API is an oracle for enumerating another tenant's ids."""
        app = create_app(with_keys(settings, "acme:acme-key-value", "globex:globex-key-value"))
        async with (
            AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client,
            app.router.lifespan_context(app),
        ):
            upload = await client.post(
                "/v1/documents",
                files={"file": ("a.md", MARKDOWN, "text/markdown")},
                headers={"X-API-Key": "acme-key-value"},
            )
            existing = upload.json()["document"]["id"]
            forbidden = await client.get(
                f"/v1/documents/{existing}", headers={"X-API-Key": "globex-key-value"}
            )
            missing = await client.get(
                "/v1/documents/doc_nonexistent", headers={"X-API-Key": "globex-key-value"}
            )

        assert forbidden.status_code == missing.status_code == 404
        assert forbidden.json()["message"] == missing.json()["message"]


class TestUploadValidation:
    @pytest.mark.parametrize(
        "filename",
        [
            "../../etc/passwd",
            "..\\..\\windows\\win.ini",
            f"normal.md{BIDI_OVERRIDE}gpj.exe",
        ],
    )
    async def test_a_hostile_filename_never_reaches_the_stored_title(
        self, client: AsyncClient, filename: str
    ) -> None:
        response = await client.post(
            "/v1/documents", files={"file": (filename, MARKDOWN, "text/markdown")}
        )
        assert response.status_code == 201
        title = response.json()["document"]["title"]
        assert "/" not in title
        assert "\\" not in title
        assert BIDI_OVERRIDE not in title

    @pytest.mark.parametrize(
        ("payload", "declared"),
        [
            (b"PK\x03\x04zip-bomb", "text/plain"),
            (b"\x1f\x8b\x08gzip", "text/plain"),
            (b"\x7fELF\x02\x01", "text/plain"),
            (b"MZ\x90\x00exe", "text/plain"),
            (b"not really a pdf", "application/pdf"),
        ],
    )
    async def test_dangerous_or_mislabelled_content_is_refused(
        self, client: AsyncClient, payload: bytes, declared: str
    ) -> None:
        response = await client.post(
            "/v1/documents", files={"file": ("payload", payload, declared)}
        )
        assert response.status_code == 415

    async def test_an_empty_upload_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/v1/documents", files={"file": ("empty.md", b"", "text/markdown")}
        )
        assert response.status_code == 415


class TestSsrfPolicy:
    @staticmethod
    def _loader(handler: Any, *, max_bytes: int = 10_000, private: bool = False) -> UrlLoader:
        return UrlLoader(
            allowed_schemes=frozenset({"https"}),
            allowed_hosts=frozenset({"docs.example.com"}),
            max_bytes=max_bytes,
            timeout_seconds=1.0,
            allow_private_network=private,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    async def test_a_host_outside_the_allowlist_is_refused_without_a_request(self) -> None:
        sent: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, content=b"body")

        loader = self._loader(record, private=True)
        with pytest.raises(UnsafeSourceError):
            await loader.load("https://evil.example.com/doc")
        assert sent == []
        await loader.aclose()

    async def test_a_redirect_is_refused_rather_than_followed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["93.184.216.34"])

        def redirect(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})

        loader = self._loader(redirect)
        with pytest.raises(UnsafeSourceError):
            await loader.load("https://docs.example.com/doc")
        await loader.aclose()

    async def test_a_body_larger_than_the_budget_aborts_mid_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content-Length is a hint; enforcement must happen while reading."""
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["93.184.216.34"])

        def oversized(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x" * 10_000)

        loader = self._loader(oversized, max_bytes=100)
        with pytest.raises(DocumentTooLargeError):
            await loader.load("https://docs.example.com/doc")
        await loader.aclose()

    async def test_an_error_status_from_the_source_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["93.184.216.34"])
        loader = self._loader(lambda request: httpx.Response(500))
        with pytest.raises(UnsafeSourceError):
            await loader.load("https://docs.example.com/doc")
        await loader.aclose()

    async def test_an_allowlisted_public_host_is_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["93.184.216.34"])

        def ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"# Doc\n\nBody.", headers={"content-type": "text/markdown"}
            )

        loader = self._loader(ok)
        loaded = await loader.load("https://docs.example.com/doc.md")
        assert loaded.payload.startswith(b"# Doc")
        assert loaded.declared_media_type == "text/markdown"
        await loader.aclose()


class TestConfigurationInvariants:
    def test_production_refuses_sqlite(self, settings: Settings) -> None:
        candidate = settings.model_copy(update={"environment": "production"})
        assert any("PostgreSQL" in problem for problem in problems(candidate))

    def test_production_refuses_disabled_authentication(self, settings: Settings) -> None:
        candidate = production(
            settings, security=SecuritySettings(require_api_key=False, api_keys=())
        )
        assert any("authentication cannot be disabled" in p for p in problems(candidate))

    def test_production_refuses_private_network_fetching(self, settings: Settings) -> None:
        candidate = production(
            settings,
            ingestion=settings.ingestion.model_copy(update={"allow_private_network_fetch": True}),
        )
        assert any("allow_private_network_fetch" in p for p in problems(candidate))

    def test_production_refuses_document_content_logging(self, settings: Settings) -> None:
        candidate = production(
            settings,
            observability=settings.observability.model_copy(update={"log_document_content": True}),
        )
        assert any("log_document_content" in p for p in problems(candidate))

    def test_production_refuses_sql_echo(self, settings: Settings) -> None:
        candidate = production(settings)
        candidate = candidate.model_copy(
            update={"storage": candidate.storage.model_copy(update={"echo_sql": True})}
        )
        assert any("echo_sql" in p for p in problems(candidate))

    def test_authentication_without_any_key_is_refused_everywhere(self, settings: Settings) -> None:
        broken = settings.model_copy(
            update={"security": SecuritySettings(require_api_key=True, api_keys=())}
        )
        assert any("no caller could ever authenticate" in p for p in problems(broken))

    def test_url_ingestion_without_an_allowlist_is_refused(self, settings: Settings) -> None:
        broken = settings.model_copy(
            update={
                "ingestion": settings.ingestion.model_copy(
                    update={"allow_url_ingestion": True, "url_allowed_hosts": frozenset()}
                )
            }
        )
        assert any("url_allowed_hosts" in p for p in problems(broken))

    def test_a_valid_test_configuration_passes(self, settings: Settings) -> None:
        settings.enforce_environment_invariants()

    def test_a_valid_production_configuration_passes(self, settings: Settings) -> None:
        production(settings).enforce_environment_invariants()

    def test_an_api_key_never_appears_in_a_settings_repr(self, settings: Settings) -> None:
        from tests.conftest import TEST_API_KEY

        assert TEST_API_KEY not in repr(settings)
        assert TEST_API_KEY not in str(settings)


class TestLogRedaction:
    @staticmethod
    def _capture(**fields: Any) -> str:
        """Emit one record through the real logging pipeline and return the output."""
        configure_logging(level="INFO", fmt="json")
        stream = io.StringIO()
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        original = handler.stream
        handler.setStream(stream)
        try:
            get_logger("tests.security").info("event", **fields)
        finally:
            handler.setStream(original)
        return stream.getvalue()

    @pytest.mark.parametrize(
        "field",
        ["api_key", "apikey", "authorization", "password", "token", "secret", "session_key"],
    )
    def test_sensitive_keys_are_replaced(self, field: str) -> None:
        emitted = self._capture(**{field: "super-secret-value"})
        assert "super-secret-value" not in emitted
        assert REDACTED in emitted

    @pytest.mark.parametrize(
        "value",
        [
            "sk-abcdefghijklmnopqrstuvwxyz0123",
            "ghp_abcdefghijklmnopqrstuvwxyz012345",
            "Bearer abcdefghijklmnopqrstuvwxyz012345",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijkl",
            "AKIAIOSFODNN7EXAMPLE",
            "-----BEGIN RSA PRIVATE KEY-----",
        ],
    )
    def test_credential_shaped_values_are_masked_anywhere_in_the_record(self, value: str) -> None:
        """A secret inside an innocuous field must not survive either."""
        emitted = self._capture(detail=f"upstream said: {value}")
        assert value not in emitted
        assert REDACTED in emitted

    def test_nested_structures_are_redacted(self) -> None:
        emitted = self._capture(context={"headers": {"authorization": "Bearer abc123def456"}})
        assert "abc123def456" not in emitted

    def test_ordinary_values_survive(self) -> None:
        emitted = self._capture(document_id="doc_123", chunks=7)
        assert "doc_123" in emitted
        assert "7" in emitted

    def test_very_long_values_are_truncated(self) -> None:
        emitted = self._capture(body="a" * 10_000)
        assert "truncated" in emitted
        assert len(emitted) < 6000

    def test_redaction_runs_after_every_other_processor(self) -> None:
        """Redaction last in the chain is what makes it impossible to bypass."""
        from rag_assistant.observability.logging import redact_processor

        configure_logging(level="INFO", fmt="json")
        processors = structlog.get_config()["processors"]
        assert processors[-2] is redact_processor
