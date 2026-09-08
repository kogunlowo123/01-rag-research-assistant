"""Ingestion against real storage."""

from __future__ import annotations

import pytest

from rag_assistant.domain.models import DocumentStatus, IngestionReport, SourceKind
from rag_assistant.errors import DocumentTooLargeError, IngestionError, UnsupportedMediaTypeError
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.runtime import Runtime
from rag_assistant.security.authz import Principal

pytestmark = pytest.mark.integration


async def ingest(
    runtime: Runtime, principal: Principal, payload: bytes, **kwargs: object
) -> IngestionReport:
    return await runtime.ingestion.ingest(
        IngestionRequest(
            payload=payload,
            filename=str(kwargs.pop("filename", "handbook.md")),
            declared_media_type=str(kwargs.pop("media_type", "text/markdown")),
            principal=principal,
            source_kind=SourceKind.UPLOAD,
            **kwargs,  # type: ignore[arg-type]
        )
    )


class TestHappyPath:
    async def test_a_document_reaches_the_indexed_state_with_chunks(
        self, runtime: Runtime, principal: Principal, sample_markdown: bytes
    ) -> None:
        report = await ingest(runtime, principal, sample_markdown)
        assert report.document.status is DocumentStatus.INDEXED
        assert report.chunks_created > 0

    async def test_chunks_vectors_and_postings_are_all_written(
        self, runtime: Runtime, principal: Principal, sample_markdown: bytes
    ) -> None:
        report = await ingest(runtime, principal, sample_markdown)
        async with runtime.unit_of_work() as unit:
            ids, vectors = await unit.chunks.load_vectors(
                principal.tenant_id, runtime.embedder.name
            )
            postings = await unit.chunks.term_statistics(principal.tenant_id, ["availability"])
            count, average_length = await unit.chunks.corpus_statistics(principal.tenant_id)

        assert len(ids) == report.chunks_created
        assert all(len(vector) == runtime.settings.embedding.dimensions for vector in vectors)
        assert postings
        assert count == len(ids)
        assert average_length > 0

    async def test_the_document_title_is_carried_on_every_chunk(
        self, runtime: Runtime, principal: Principal, sample_markdown: bytes
    ) -> None:
        report = await ingest(runtime, principal, sample_markdown, title="Service Levels")
        async with runtime.unit_of_work() as unit:
            ids, _ = await unit.chunks.load_vectors(principal.tenant_id, runtime.embedder.name)
            chunks = await unit.chunks.get_many(principal.tenant_id, ids)
        assert chunks
        assert all(
            chunk.metadata["document_title"] == "Service Levels" for chunk in chunks.values()
        )
        assert report.document.title == "Service Levels"

    async def test_an_audit_event_is_recorded(
        self, runtime: Runtime, principal: Principal, sample_markdown: bytes
    ) -> None:
        await ingest(runtime, principal, sample_markdown)
        async with runtime.unit_of_work() as unit:
            events = await unit.audit.recent(principal.tenant_id)
        assert any(event.event == "document.ingest" for event in events)
        assert all(event.outcome != "failed" for event in events)


class TestDeduplication:
    async def test_identical_content_is_not_indexed_twice(
        self, runtime: Runtime, principal: Principal, sample_markdown: bytes
    ) -> None:
        first = await ingest(runtime, principal, sample_markdown)
        second = await ingest(runtime, principal, sample_markdown, filename="copy.md")
        assert second.duplicate_of == first.document.id
        assert second.chunks_created == 0

    async def test_the_same_content_in_two_tenants_produces_two_documents(
        self, runtime: Runtime, sample_markdown: bytes
    ) -> None:
        acme = Principal(tenant_id="acme", key_id="aaaaaaaa")
        globex = Principal(tenant_id="globex", key_id="bbbbbbbb")
        first = await ingest(runtime, acme, sample_markdown)
        second = await ingest(runtime, globex, sample_markdown)
        assert second.duplicate_of is None
        assert first.document.id != second.document.id


class TestInjectionAtIngest:
    async def test_a_poisoned_document_is_indexed_but_flagged(
        self, runtime: Runtime, principal: Principal, poisoned_markdown: bytes
    ) -> None:
        report = await ingest(runtime, principal, poisoned_markdown, filename="vendor.md")
        assert report.document.status is DocumentStatus.INDEXED
        assert report.injection_findings
        assert any("instruction-like" in warning for warning in report.warnings)

    async def test_the_finding_is_audited_with_its_rule_ids(
        self, runtime: Runtime, principal: Principal, poisoned_markdown: bytes
    ) -> None:
        await ingest(runtime, principal, poisoned_markdown, filename="vendor.md")
        async with runtime.unit_of_work() as unit:
            events = await unit.audit.recent(principal.tenant_id)
        ingest_event = next(event for event in events if event.event == "document.ingest")
        assert ingest_event.attributes["injection_rules"]
        assert float(str(ingest_event.attributes["injection_risk"])) > 0.5


class TestFailurePaths:
    async def test_an_oversized_document_is_refused(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        oversized = b"x" * (runtime.settings.ingestion.max_document_bytes + 1)
        with pytest.raises(DocumentTooLargeError):
            await ingest(runtime, principal, oversized, media_type="text/plain")

    async def test_an_archive_is_refused_whatever_it_claims_to_be(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            await ingest(runtime, principal, b"PK\x03\x04payload", media_type="text/plain")

    async def test_an_unparseable_document_is_recorded_as_failed(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        with pytest.raises(IngestionError):
            await ingest(runtime, principal, b"%PDF-1.4\nbroken", media_type="application/pdf")

        async with runtime.unit_of_work() as unit:
            documents = await unit.documents.list(principal.tenant_id)
            events = await unit.audit.recent(principal.tenant_id)

        assert any(document.status is DocumentStatus.FAILED for document in documents)
        assert any(event.outcome == "failed" for event in events)

    async def test_a_failed_document_records_a_client_safe_reason(
        self, runtime: Runtime, principal: Principal
    ) -> None:
        with pytest.raises(IngestionError):
            await ingest(runtime, principal, b"%PDF-1.4\nbroken", media_type="application/pdf")
        async with runtime.unit_of_work() as unit:
            documents = await unit.documents.list(principal.tenant_id)
        failed = next(d for d in documents if d.status is DocumentStatus.FAILED)
        assert failed.error
        assert "Traceback" not in failed.error
