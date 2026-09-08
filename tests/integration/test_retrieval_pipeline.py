"""Hybrid retrieval against a real index."""

from __future__ import annotations

import pytest

from rag_assistant.domain.models import SourceKind
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.retrieval.pipeline import RetrievalRequest
from rag_assistant.runtime import Runtime
from rag_assistant.security.authz import Principal

pytestmark = pytest.mark.integration

POLICY = b"""# Support Policy

## Severity definitions

Severity one means a total loss of service for all users. Severity two means a
partial loss of service or a loss affecting a single tenant.

## Response targets

Severity one incidents receive a response within 15 minutes. Severity two
incidents receive a response within 4 business hours.

## Error codes

INC_9032 is raised when the incident queue rejects a submission because the
severity field is absent.
"""

SHIPPING = b"""# Shipping

## Domestic

Standard domestic shipping takes 3 to 7 business days.

## International

International shipping takes 7 to 21 business days.
"""


async def ingest(
    runtime: Runtime, principal: Principal, payload: bytes, name: str, **kw: object
) -> str:
    report = await runtime.ingestion.ingest(
        IngestionRequest(
            payload=payload,
            filename=name,
            declared_media_type="text/markdown",
            principal=principal,
            source_kind=SourceKind.UPLOAD,
            source_ref=name,
            title=name,
            **kw,  # type: ignore[arg-type]
        )
    )
    return report.document.id


async def retrieve(runtime: Runtime, principal: Principal, query: str, **kw: object) -> object:
    async with runtime.unit_of_work() as unit:
        return await unit.retrieval.retrieve(
            RetrievalRequest(query=query, principal=principal, **kw)  # type: ignore[arg-type]
        )


class TestHybridRetrieval:
    async def test_a_semantic_style_query_finds_the_right_section(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        await ingest(runtime, principal, POLICY, "support.md")
        await ingest(runtime, principal, SHIPPING, "shipping.md")
        result = await retrieve(
            runtime, principal, "how quickly is a severity one incident answered"
        )
        texts = " ".join(match.chunk.text for match in result.matches)  # type: ignore[attr-defined]
        assert "15 minutes" in texts

    async def test_an_exact_identifier_is_found_by_the_sparse_retriever(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        await ingest(runtime, principal, POLICY, "support.md")
        await ingest(runtime, principal, SHIPPING, "shipping.md")
        result = await retrieve(runtime, principal, "INC_9032")
        top = result.matches[0]  # type: ignore[attr-defined]
        assert "INC_9032" in top.chunk.text
        assert top.sparse_rank is not None

    async def test_diagnostics_report_both_retrievers(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        await ingest(runtime, principal, POLICY, "support.md")
        result = await retrieve(
            runtime, principal, "severity one response target", include_diagnostics=True
        )
        diagnostics = result.diagnostics  # type: ignore[attr-defined]
        assert diagnostics.dense_candidate_count > 0
        assert diagnostics.sparse_candidate_count > 0
        assert diagnostics.fused_candidate_count > 0
        assert set(diagnostics.latency_ms) >= {"rewrite_ms", "search_ms", "fuse_ms", "rerank_ms"}

    async def test_diagnostics_omit_the_query_unless_requested(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        """The query is user data; echoing it back is opt-in."""
        await ingest(runtime, principal, POLICY, "support.md")
        result = await retrieve(runtime, principal, "severity one")
        assert result.diagnostics.original_query == ""  # type: ignore[attr-defined]

    async def test_top_k_is_respected(self, runtime: Runtime, principal: Principal) -> None:
        await ingest(runtime, principal, POLICY, "support.md")
        await ingest(runtime, principal, SHIPPING, "shipping.md")
        result = await retrieve(runtime, principal, "shipping days", top_k=2)
        assert len(result.matches) <= 2  # type: ignore[attr-defined]

    async def test_an_empty_index_returns_nothing_rather_than_failing(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        result = await retrieve(runtime, principal, "anything at all")
        assert result.matches == ()  # type: ignore[attr-defined]


class TestDeletionVisibility:
    async def test_a_deleted_document_stops_being_retrievable_immediately(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        document_id = await ingest(runtime, principal, POLICY, "support.md")
        before = await retrieve(runtime, principal, "severity one incident response")
        assert before.matches  # type: ignore[attr-defined]

        async with runtime.unit_of_work() as unit:
            assert await unit.documents.soft_delete(principal.tenant_id, document_id)
        runtime.vector_store.invalidate(principal.tenant_id)

        after = await retrieve(runtime, principal, "severity one incident response")
        assert after.matches == ()  # type: ignore[attr-defined]

    async def test_deleting_removes_the_sparse_postings_too(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        document_id = await ingest(runtime, principal, POLICY, "support.md")
        async with runtime.unit_of_work() as unit:
            await unit.documents.soft_delete(principal.tenant_id, document_id)
        runtime.vector_store.invalidate(principal.tenant_id)

        async with runtime.unit_of_work() as unit:
            postings = await unit.chunks.term_statistics(principal.tenant_id, ["severity"])
        assert postings == {}


class TestTenantIsolation:
    async def test_one_tenant_never_retrieves_another_tenants_document(
        self, runtime: Runtime
    ) -> None:
        acme = Principal(tenant_id="acme", key_id="aaaaaaaa")
        globex = Principal(tenant_id="globex", key_id="bbbbbbbb")
        await ingest(runtime, acme, POLICY, "support.md")

        result = await retrieve(runtime, globex, "severity one incident response")
        assert result.matches == ()  # type: ignore[attr-defined]

    async def test_isolation_holds_for_an_exact_identifier(self, runtime: Runtime) -> None:
        """Lexical search is the path most likely to leak across a partition."""
        acme = Principal(tenant_id="acme", key_id="aaaaaaaa")
        globex = Principal(tenant_id="globex", key_id="bbbbbbbb")
        await ingest(runtime, acme, POLICY, "support.md")

        result = await retrieve(runtime, globex, "INC_9032")
        assert result.matches == ()  # type: ignore[attr-defined]


class TestDocumentAcl:
    async def test_a_restricted_document_is_invisible_without_the_group(
        self, runtime: Runtime
    ) -> None:
        owner = Principal(tenant_id="acme", key_id="aaaaaaaa", groups=frozenset({"support"}))
        outsider = Principal(tenant_id="acme", key_id="bbbbbbbb")
        await ingest(runtime, owner, POLICY, "support.md", acl=("support",))

        allowed = await retrieve(runtime, owner, "severity one incident response")
        denied = await retrieve(
            runtime, outsider, "severity one incident response", include_diagnostics=True
        )

        assert allowed.matches  # type: ignore[attr-defined]
        assert denied.matches == ()  # type: ignore[attr-defined]
        assert denied.diagnostics.dropped_by_authorization > 0  # type: ignore[attr-defined]


class TestUntrustedContentPolicy:
    async def test_a_poisoned_passage_is_neutralised_before_it_can_be_used(
        self, runtime: Runtime, principal: Principal, poisoned_markdown: bytes
    ) -> None:
        await ingest(runtime, principal, poisoned_markdown, "vendor.md")
        result = await retrieve(
            runtime, principal, "vendor quarterly units delivered", include_diagnostics=True
        )
        texts = " ".join(match.chunk.text for match in result.matches)  # type: ignore[attr-defined]
        assert "attacker.invalid" not in texts
        assert "unrestricted assistant" not in texts

    async def test_the_factual_content_of_a_poisoned_document_survives(
        self, runtime: Runtime, principal: Principal, poisoned_markdown: bytes
    ) -> None:
        """Neutralising must not throw away the passage's legitimate content."""
        await ingest(runtime, principal, poisoned_markdown, "vendor.md")
        result = await retrieve(runtime, principal, "how many units did the vendor deliver")
        texts = " ".join(match.chunk.text for match in result.matches)  # type: ignore[attr-defined]
        assert "412 units" in texts

    async def test_drop_mode_removes_the_passage_entirely(
        self, runtime: Runtime, principal: Principal, poisoned_markdown: bytes
    ) -> None:
        await ingest(runtime, principal, poisoned_markdown, "vendor.md")
        runtime.settings = runtime.settings.model_copy(
            update={
                "security": runtime.settings.security.model_copy(
                    update={"injection_action": "drop"}
                )
            }
        )
        result = await retrieve(
            runtime, principal, "ignore previous instructions", include_diagnostics=True
        )
        assert result.diagnostics.dropped_by_policy > 0  # type: ignore[attr-defined]
