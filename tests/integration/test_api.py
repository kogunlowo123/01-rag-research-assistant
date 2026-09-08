"""HTTP contract: status codes, payload shapes and headers."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from rag_assistant.api.middleware import REQUEST_ID_HEADER, SECURITY_HEADERS

pytestmark = pytest.mark.integration

MARKDOWN = (
    b"# Service Levels\n\n## Availability\n\n"
    b"The platform targets 99.9 percent availability measured monthly. "
    b"Unplanned downtime is credited at five percent of the monthly fee per hour.\n\n"
    b"## Support response\n\n"
    b"Severity one incidents receive a response within 15 minutes at any hour.\n"
)


async def upload(
    client: AsyncClient, payload: bytes = MARKDOWN, name: str = "sla.md"
) -> dict[str, Any]:
    response = await client.post(
        "/v1/documents",
        files={"file": (name, payload, "text/markdown")},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


class TestHealthEndpoints:
    async def test_liveness_needs_no_credential(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_readiness_reports_components_and_backend_warnings(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert {component["name"] for component in body["components"]} >= {
            "database",
            "embeddings",
            "generation",
        }
        # The deterministic backends must announce themselves rather than
        # passing silently as production-grade.
        assert any("hashing" in warning for warning in body["warnings"])
        assert any("extractive" in warning for warning in body["warnings"])


class TestMiddleware:
    async def test_every_response_carries_the_security_headers(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get("/healthz")
        for header, value in SECURITY_HEADERS.items():
            assert response.headers[header] == value

    async def test_a_request_id_is_generated_and_echoed(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get("/healthz")
        assert response.headers[REQUEST_ID_HEADER]

    async def test_a_supplied_request_id_is_preserved(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.get(
            "/healthz", headers={REQUEST_ID_HEADER: "trace-abc-123"}
        )
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"

    async def test_a_hostile_request_id_is_replaced(self, anonymous_client: AsyncClient) -> None:
        """A caller-supplied value reaches the logs, so it must be constrained."""
        response = await anonymous_client.get(
            "/healthz", headers={REQUEST_ID_HEADER: "abc def\ninjected=1"}
        )
        echoed = response.headers[REQUEST_ID_HEADER]
        assert "\n" not in echoed
        assert " " not in echoed

    async def test_an_oversized_body_is_rejected_before_it_is_read(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/v1/query",
            content=b"x" * (3 * 1024 * 1024),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "document_too_large"


class TestAuthentication:
    async def test_a_protected_endpoint_refuses_an_anonymous_caller(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get("/v1/documents")
        assert response.status_code == 403
        assert response.json()["code"] == "forbidden"

    async def test_a_wrong_key_is_refused(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.get("/v1/documents", headers={"X-API-Key": "not-the-key"})
        assert response.status_code == 403

    async def test_a_bearer_token_is_accepted(self, anonymous_client: AsyncClient) -> None:
        from tests.conftest import TEST_API_KEY

        response = await anonymous_client.get(
            "/v1/documents", headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )
        assert response.status_code == 200

    async def test_the_error_body_never_echoes_the_presented_key(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get(
            "/v1/documents", headers={"X-API-Key": "leaked-secret-value"}
        )
        assert "leaked-secret-value" not in response.text


class TestDocumentEndpoints:
    async def test_upload_returns_the_indexed_document(self, client: AsyncClient) -> None:
        body = await upload(client)
        assert body["document"]["status"] == "indexed"
        assert body["chunks_created"] > 0
        assert body["document"]["media_type"] == "text/markdown"

    async def test_listing_returns_the_uploaded_document(self, client: AsyncClient) -> None:
        await upload(client)
        response = await client.get("/v1/documents")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["documents"][0]["title"] == "sla.md"

    async def test_fetching_one_document_returns_its_status(self, client: AsyncClient) -> None:
        document_id = (await upload(client))["document"]["id"]
        response = await client.get(f"/v1/documents/{document_id}")
        assert response.status_code == 200
        assert response.json()["id"] == document_id

    async def test_an_unknown_document_is_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/v1/documents/doc_does_not_exist")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    async def test_deletion_removes_the_document_from_listings(self, client: AsyncClient) -> None:
        document_id = (await upload(client))["document"]["id"]
        assert (await client.delete(f"/v1/documents/{document_id}")).status_code == 204
        assert (await client.get("/v1/documents")).json()["total"] == 0

    async def test_deleting_twice_is_not_found_the_second_time(self, client: AsyncClient) -> None:
        document_id = (await upload(client))["document"]["id"]
        await client.delete(f"/v1/documents/{document_id}")
        assert (await client.delete(f"/v1/documents/{document_id}")).status_code == 404

    async def test_an_archive_upload_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/v1/documents",
            files={"file": ("payload.txt", b"PK\x03\x04zip", "text/plain")},
        )
        assert response.status_code == 415
        assert response.json()["code"] == "unsupported_media_type"

    async def test_a_duplicate_upload_reports_the_original(self, client: AsyncClient) -> None:
        first = await upload(client)
        second = await upload(client, name="copy.md")
        assert second["duplicate_of"] == first["document"]["id"]
        assert second["chunks_created"] == 0

    async def test_url_ingestion_is_refused_while_disabled(self, client: AsyncClient) -> None:
        response = await client.post(
            "/v1/documents/from-url", json={"url": "https://docs.example.com/a.pdf"}
        )
        assert response.status_code == 403
        assert "URL ingestion is disabled" in response.json()["message"]

    async def test_an_unknown_field_is_rejected_rather_than_ignored(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/v1/documents/from-url",
            json={"url": "https://docs.example.com/a.pdf", "unexpected": True},
        )
        assert response.status_code == 422


class TestQueryEndpoint:
    async def test_a_grounded_answer_carries_citations_and_a_score(
        self, client: AsyncClient
    ) -> None:
        await upload(client)
        response = await client.post(
            "/v1/query",
            json={"query": "How quickly is a severity one incident answered?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is False
        assert body["citations"]
        assert body["grounding"]["score"] > 0
        assert body["confidence"] > 0
        assert "15 minutes" in body["answer"]

    async def test_a_citation_resolves_to_a_real_document(self, client: AsyncClient) -> None:
        document_id = (await upload(client))["document"]["id"]
        body = (await client.post("/v1/query", json={"query": "severity one response time"})).json()
        assert body["citations"][0]["document_id"] == document_id
        assert body["citations"][0]["quote"]

    async def test_an_unanswerable_question_is_refused_with_a_reason(
        self, client: AsyncClient
    ) -> None:
        await upload(client)
        body = (
            await client.post(
                "/v1/query",
                json={"query": "What is the company's cryptocurrency treasury policy?"},
            )
        ).json()
        assert body["refused"] is True
        assert body["refusal_reason"]
        assert body["citations"] == []
        assert body["confidence"] == 0.0

    async def test_a_refusal_is_a_200_not_an_error(self, client: AsyncClient) -> None:
        """A refusal is a correct outcome, so it must not look like a failure."""
        await upload(client)
        response = await client.post("/v1/query", json={"query": "unrelated cryptocurrency"})
        assert response.status_code == 200

    async def test_diagnostics_are_returned_on_request(self, client: AsyncClient) -> None:
        await upload(client)
        body = (
            await client.post(
                "/v1/query",
                json={"query": "severity one response", "include_diagnostics": True},
            )
        ).json()
        assert body["diagnostics"]["dense_candidate_count"] > 0
        assert body["diagnostics"]["latency_ms"]

    async def test_diagnostics_are_absent_by_default(self, client: AsyncClient) -> None:
        await upload(client)
        body = (await client.post("/v1/query", json={"query": "severity one"})).json()
        assert body["diagnostics"] is None

    async def test_a_session_carries_context_between_turns(self, client: AsyncClient) -> None:
        await upload(client)
        first = await client.post(
            "/v1/query",
            json={"query": "What is the severity one response target?", "session_id": "s-1"},
        )
        assert first.json()["session_id"] == "s-1"
        second = await client.post(
            "/v1/query", json={"query": "and what about availability?", "session_id": "s-1"}
        )
        assert second.status_code == 200

    async def test_an_empty_query_is_rejected(self, client: AsyncClient) -> None:
        assert (await client.post("/v1/query", json={"query": "   "})).status_code == 422

    async def test_an_oversized_top_k_is_rejected(self, client: AsyncClient) -> None:
        response = await client.post("/v1/query", json={"query": "anything", "top_k": 5000})
        assert response.status_code == 422

    async def test_the_retrieve_endpoint_reports_without_generating(
        self, client: AsyncClient
    ) -> None:
        await upload(client)
        response = await client.get("/v1/retrieve", params={"q": "severity one response"})
        assert response.status_code == 200
        body = response.json()
        assert body["original_query"] == "severity one response"
        assert body["fused_candidate_count"] > 0


class TestAuditEndpoint:
    async def test_ingestion_and_query_are_both_audited(self, client: AsyncClient) -> None:
        await upload(client)
        await client.post("/v1/query", json={"query": "severity one response"})
        events = (await client.get("/v1/audit")).json()["events"]
        assert {event["event"] for event in events} >= {"document.ingest", "query.answer"}

    async def test_audit_entries_contain_no_document_or_query_text(
        self, client: AsyncClient
    ) -> None:
        await upload(client)
        await client.post("/v1/query", json={"query": "severity one response target"})
        payload = (await client.get("/v1/audit")).text
        assert "severity one response target" not in payload
        assert "99.9 percent" not in payload


class TestErrorContract:
    async def test_every_error_uses_the_same_envelope(self, client: AsyncClient) -> None:
        response = await client.get("/v1/documents/doc_missing")
        body = response.json()
        assert set(body) == {"code", "message", "request_id", "detail"}
        assert body["request_id"]

    async def test_validation_errors_do_not_echo_the_submitted_value(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/v1/query", json={"query": "x" * 100000, "session_id": "!!invalid!!"}
        )
        assert response.status_code == 422
        assert "x" * 1000 not in response.text
        assert "!!invalid!!" not in response.text

    async def test_the_openapi_document_is_served(self, client: AsyncClient) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/v1/query" in paths
        assert "/v1/documents" in paths
