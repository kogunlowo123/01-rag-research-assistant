"""Evaluation runner.

Ingests a dataset's corpus into a scratch index, executes every case through the
real retrieval and generation pipeline, scores the outcomes, and writes a
machine-readable report. A run that violates a threshold exits non-zero, which
is how the CI job fails on a quality regression.

The index is built fresh for every run into a temporary database. That costs a
few seconds and buys determinism: a stale index from a previous run cannot make
a broken retrieval look healthy.
"""

from __future__ import annotations

import json
import mimetypes
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rag_assistant.config import Settings
from rag_assistant.domain.models import SourceKind
from rag_assistant.evaluation.dataset import EvalCase, EvalDataset
from rag_assistant.evaluation.metrics import (
    CaseResult,
    EvaluationReport,
    Thresholds,
    citation_precision,
    recall_at_k,
    reciprocal_rank,
)
from rag_assistant.generation.answerer import AnswerRequest
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.observability.logging import get_logger
from rag_assistant.retrieval.pipeline import RetrievalRequest
from rag_assistant.runtime import Runtime
from rag_assistant.security.authz import Principal

if TYPE_CHECKING:
    from rag_assistant.domain.models import Answer, RetrievalResult

logger = get_logger(__name__)

EVAL_TENANT = "evaluation"
_PREVIEW_CHARS = 240


def _evaluation_settings(base: Settings, database_url: str) -> Settings:
    """Derive settings for an evaluation run against a scratch database."""
    return base.model_copy(
        update={
            "storage": base.storage.model_copy(update={"database_url": database_url}),
            "security": base.security.model_copy(update={"require_api_key": False, "api_keys": ()}),
        }
    )


async def ingest_corpus(runtime: Runtime, corpus_dir: Path, principal: Principal) -> int:
    """Ingest every file in the corpus directory. Returns the document count."""
    if not corpus_dir.is_dir():
        msg = f"corpus directory {corpus_dir} does not exist"
        raise FileNotFoundError(msg)

    ingested = 0
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        media_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        if path.suffix.lower() == ".md":
            media_type = "text/markdown"
        await runtime.ingestion.ingest(
            IngestionRequest(
                payload=path.read_bytes(),
                filename=path.name,
                declared_media_type=media_type,
                principal=principal,
                source_kind=SourceKind.UPLOAD,
                source_ref=path.name,
                title=path.name,
            )
        )
        ingested += 1
    return ingested


def score_case(
    case: EvalCase,
    answer: Answer,
    retrieval: RetrievalResult,
    titles: dict[str, str],
    *,
    top_k: int,
) -> CaseResult:
    """Compare one case's expectations against what the system produced."""
    retrieved_titles = [
        titles.get(match.chunk.document_id, match.chunk.document_id) for match in retrieval.matches
    ]
    # A document may contribute several chunks; rank by first appearance.
    ordered_documents: list[str] = []
    for title in retrieved_titles:
        if title not in ordered_documents:
            ordered_documents.append(title)

    cited_documents = [citation.document_title for citation in answer.citations]
    lowered_answer = answer.text.lower()
    failures: list[str] = []

    if case.must_refuse and not answer.refused:
        failures.append("expected_refusal_but_answered")
    if not case.must_refuse and answer.refused:
        failures.append("unexpected_refusal")

    if not case.must_refuse:
        for needle in case.must_contain:
            if needle.lower() not in lowered_answer:
                failures.append("missing_expected_content")
                break
        if case.require_citation and not answer.citations:
            failures.append("no_resolvable_citation")
        if answer.grounding.score < case.min_grounding:
            failures.append("grounding_below_case_minimum")

    for needle in case.must_not_contain:
        if needle.lower() in lowered_answer:
            failures.append("contains_forbidden_content")
            break

    recall = recall_at_k(ordered_documents, case.expected_documents, top_k)
    if case.expected_documents and recall < 1.0:
        failures.append("retrieval_missed_expected_document")

    return CaseResult(
        case_id=case.id,
        category=case.category,
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank(ordered_documents, case.expected_documents),
        citation_precision=citation_precision(cited_documents, case.expected_documents),
        grounding=answer.grounding.score,
        refused=answer.refused,
        citations=len(answer.citations),
        latency_ms=answer.latency_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        provider=answer.provider,
        answer_preview=(answer.text or answer.refusal_reason or "")[:_PREVIEW_CHARS],
    )


async def run_dataset(
    dataset_path: Path,
    *,
    settings: Settings,
    thresholds: Thresholds | None = None,
    top_k: int = 8,
) -> EvaluationReport:
    """Run every case in a dataset and return the report."""
    dataset = EvalDataset.load(dataset_path)
    corpus_dir = dataset.resolve_corpus(dataset_path)
    principal = Principal(tenant_id=EVAL_TENANT, key_id="evaluation")

    with tempfile.TemporaryDirectory(prefix="rag-eval-") as scratch:
        database = Path(scratch) / "eval.db"
        runtime = Runtime(_evaluation_settings(settings, f"sqlite+aiosqlite:///{database}"))
        await runtime.start()
        try:
            documents = await ingest_corpus(runtime, corpus_dir, principal)
            logger.info("evaluation.corpus_ready", dataset=dataset.name, documents=documents)

            results: list[CaseResult] = []
            for case in dataset.cases:
                async with runtime.unit_of_work() as unit:
                    retrieval = await unit.retrieval.retrieve(
                        RetrievalRequest(query=case.query, principal=principal, top_k=top_k)
                    )
                    titles = await unit.chunks.document_titles(
                        principal.tenant_id,
                        sorted({match.chunk.document_id for match in retrieval.matches}),
                    )
                    answer = await unit.answerer.answer(
                        AnswerRequest(query=case.query, retrieval=retrieval, document_titles=titles)
                    )
                results.append(score_case(case, answer, retrieval, titles, top_k=top_k))

            return EvaluationReport(
                dataset=dataset.name,
                results=results,
                thresholds=thresholds or Thresholds(),
                embedding_provider=runtime.embedder.name,
                chat_provider=f"{runtime.chat.name}:{runtime.chat.model}",
            )
        finally:
            await runtime.aclose()


def write_report(report: EvaluationReport, destination: Path) -> None:
    """Write the machine-readable report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_summary(report: EvaluationReport) -> str:
    """Render a human-readable summary for a terminal or a CI log."""
    lines = [
        f"dataset          {report.dataset}",
        f"embeddings       {report.embedding_provider}",
        f"generation       {report.chat_provider}",
        f"cases            {report.passed}/{report.total} passed ({report.pass_rate:.1%})",
        f"recall@k         {report.mean_recall:.3f}",
        f"MRR              {report.mean_reciprocal_rank:.3f}",
        f"grounding        {report.mean_grounding:.3f}",
        f"citation prec.   {report.mean_citation_precision:.3f}",
        f"p95 latency      {report.p95_latency_ms:.0f} ms",
        f"tokens           {report.total_tokens}",
        "",
        "by category:",
    ]
    for name, summary in sorted(report.by_category().items()):
        lines.append(
            f"  {name:<14} {summary.passed}/{summary.total} "
            f"recall={summary.mean_recall:.3f} grounding={summary.mean_grounding:.3f}"
        )

    if reasons := report.failure_reasons():
        lines.extend(["", "failure reasons:"])
        lines.extend(f"  {reason:<38} {count}" for reason, count in reasons.most_common())

    lines.append("")
    if report.passed_gate:
        lines.append("QUALITY GATE: PASSED")
    else:
        lines.append("QUALITY GATE: FAILED")
        lines.extend(f"  - {failure}" for failure in report.threshold_failures())
    return "\n".join(lines)


__all__ = [
    "EVAL_TENANT",
    "ingest_corpus",
    "render_summary",
    "run_dataset",
    "score_case",
    "write_report",
]
