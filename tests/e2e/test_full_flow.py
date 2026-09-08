"""End-to-end paths: the CLI, the HTTP service, and the evaluation gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from rag_assistant.cli import main
from rag_assistant.config import Settings
from rag_assistant.evaluation.metrics import Thresholds
from rag_assistant.evaluation.runner import run_dataset

pytestmark = pytest.mark.e2e

DATASET = Path(__file__).resolve().parents[2] / "data" / "regression" / "dataset.jsonl"


class TestHttpEndToEnd:
    async def test_upload_query_cite_and_delete(self, client: AsyncClient) -> None:
        """The whole user-visible path, in one test, against the real application."""
        upload = await client.post(
            "/v1/documents",
            files={
                "file": (
                    "refunds.md",
                    b"# Refund Policy\n\n## Eligibility\n\n"
                    b"Customers may request a refund within 30 days of purchase. "
                    b"Proof of purchase is required for every request.\n\n"
                    b"## Processing\n\n"
                    b"Approved refunds are processed within 5 business days.\n",
                    "text/markdown",
                )
            },
        )
        assert upload.status_code == 201
        document_id = upload.json()["document"]["id"]

        answer = await client.post(
            "/v1/query",
            json={
                "query": "How many days do customers have to request a refund?",
                "include_diagnostics": True,
            },
        )
        assert answer.status_code == 200
        body = answer.json()

        assert body["refused"] is False
        assert "30 days" in body["answer"]
        assert body["citations"], body
        assert body["citations"][0]["document_id"] == document_id
        assert body["citations"][0]["quote"]
        assert body["grounding"]["score"] > 0.5
        assert body["confidence"] > 0.0
        assert body["diagnostics"]["fused_candidate_count"] > 0

        audit = await client.get("/v1/audit")
        assert {event["event"] for event in audit.json()["events"]} == {
            "document.ingest",
            "query.answer",
        }

        assert (await client.delete(f"/v1/documents/{document_id}")).status_code == 204
        after = await client.post("/v1/query", json={"query": "refund window"})
        assert after.json()["refused"] is True

    async def test_a_multi_turn_session_resolves_a_follow_up(self, client: AsyncClient) -> None:
        await client.post(
            "/v1/documents",
            files={
                "file": (
                    "policy.md",
                    b"# Policy\n\n## Refunds\n\n"
                    b"Customers may request a refund within 30 days of purchase.\n\n"
                    b"## Digital goods\n\n"
                    b"Digital goods are refundable within 14 days if never downloaded.\n",
                    "text/markdown",
                )
            },
        )
        first = await client.post(
            "/v1/query",
            json={"query": "What is the refund window?", "session_id": "thread-1"},
        )
        assert first.json()["refused"] is False

        follow_up = await client.post(
            "/v1/query",
            json={"query": "and what about digital goods?", "session_id": "thread-1"},
        )
        assert follow_up.status_code == 200
        assert follow_up.json()["session_id"] == "thread-1"


class TestEvaluationGate:
    async def test_the_shipped_regression_suite_passes_its_thresholds(
        self, open_settings: Settings
    ) -> None:
        """The gate that guards CI, executed here so it cannot silently rot."""
        report = await run_dataset(DATASET, settings=open_settings, thresholds=Thresholds())

        assert report.total > 0
        assert report.passed_gate, report.threshold_failures()
        assert report.mean_recall >= 0.8
        assert report.by_category()["adversarial"].pass_rate == 1.0
        assert report.embedding_provider
        assert report.chat_provider

    async def test_the_gate_actually_fails_when_thresholds_are_impossible(
        self, open_settings: Settings
    ) -> None:
        """A gate that cannot fail is decoration."""
        report = await run_dataset(
            DATASET,
            settings=open_settings,
            thresholds=Thresholds(min_pass_rate=1.01, min_recall_at_k=1.01),
        )
        assert not report.passed_gate
        assert report.threshold_failures()


class TestCliEndToEnd:
    def test_ingest_then_query_through_the_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "refunds.md").write_text(
            "# Refund Policy\n\n## Eligibility\n\n"
            "Customers may request a refund within 30 days of purchase.\n",
            encoding="utf-8",
        )

        monkeypatch.setenv(
            "RAG_STORAGE__DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}"
        )
        monkeypatch.setenv("RAG_SECURITY__REQUIRE_API_KEY", "false")
        monkeypatch.setenv("RAG_OBSERVABILITY__LOG_LEVEL", "ERROR")
        monkeypatch.setenv("RAG_EMBEDDING__DIMENSIONS", "256")

        from rag_assistant.config import reset_settings_cache

        reset_settings_cache()
        try:
            assert main(["ingest", str(corpus)]) == 0
            assert "indexed" in capsys.readouterr().out

            assert main(["query", "How long is the refund window?"]) == 0
            output = capsys.readouterr().out
            assert "30 days" in output
            assert "refunds.md" in output
        finally:
            reset_settings_cache()

    def test_the_cli_reports_a_failed_ingest_without_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4\nnot really a pdf")

        monkeypatch.setenv("RAG_STORAGE__DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'x.db'}")
        monkeypatch.setenv("RAG_SECURITY__REQUIRE_API_KEY", "false")
        monkeypatch.setenv("RAG_OBSERVABILITY__LOG_LEVEL", "ERROR")

        from rag_assistant.config import reset_settings_cache

        reset_settings_cache()
        try:
            assert main(["ingest", str(broken)]) == 1
            assert "FAILED" in capsys.readouterr().err
        finally:
            reset_settings_cache()

    def test_no_matching_files_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("RAG_STORAGE__DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'y.db'}")
        monkeypatch.setenv("RAG_SECURITY__REQUIRE_API_KEY", "false")

        from rag_assistant.config import reset_settings_cache

        reset_settings_cache()
        try:
            assert main(["ingest", str(empty)]) == 1
            assert "no files matched" in capsys.readouterr().err
        finally:
            reset_settings_cache()

    def test_the_parser_rejects_an_unknown_command(self) -> None:
        with pytest.raises(SystemExit):
            main(["not-a-command"])
