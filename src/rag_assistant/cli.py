"""Command-line interface.

Exists so the pipeline can be driven without an HTTP server: ingest a directory,
ask a question, run the evaluation gate. The CLI uses the same
:class:`~rag_assistant.runtime.Runtime` as the API, so what it exercises is the
real system rather than a parallel implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import sys
from pathlib import Path

from rag_assistant.config import get_settings
from rag_assistant.domain.models import SourceKind
from rag_assistant.errors import RagError
from rag_assistant.evaluation.metrics import Thresholds
from rag_assistant.evaluation.runner import render_summary, run_dataset, write_report
from rag_assistant.generation.answerer import AnswerRequest
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.retrieval.pipeline import RetrievalRequest
from rag_assistant.runtime import build_runtime
from rag_assistant.security.authz import Principal

DEFAULT_TENANT = "default"


def _principal(tenant: str) -> Principal:
    return Principal(tenant_id=tenant, key_id="cli")


def _media_type(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return "text/markdown"
    return mimetypes.guess_type(path.name)[0] or "text/plain"


async def _ingest(paths: list[Path], tenant: str) -> int:
    runtime = await build_runtime()
    principal = _principal(tenant)
    failures = 0
    try:
        files = [
            candidate
            for path in paths
            for candidate in (sorted(path.rglob("*")) if path.is_dir() else [path])
            if candidate.is_file()
        ]
        if not files:
            print("no files matched", file=sys.stderr)
            return 1

        for file in files:
            try:
                report = await runtime.ingestion.ingest(
                    IngestionRequest(
                        payload=file.read_bytes(),
                        filename=file.name,
                        declared_media_type=_media_type(file),
                        principal=principal,
                        source_kind=SourceKind.UPLOAD,
                        source_ref=file.name,
                    )
                )
            except RagError as exc:
                failures += 1
                print(f"FAILED  {file.name}: {exc.message}", file=sys.stderr)
                continue

            status = "duplicate" if report.duplicate_of else "indexed"
            print(f"{status:<10} {file.name}  chunks={report.chunks_created}")
            for warning in report.warnings:
                print(f"           warning: {warning}")
    finally:
        await runtime.aclose()
    return 1 if failures else 0


async def _query(question: str, tenant: str, top_k: int, diagnostics: bool) -> int:
    runtime = await build_runtime()
    principal = _principal(tenant)
    try:
        async with runtime.unit_of_work() as unit:
            retrieval = await unit.retrieval.retrieve(
                RetrievalRequest(
                    query=question,
                    principal=principal,
                    top_k=top_k,
                    include_diagnostics=diagnostics,
                )
            )
            titles = await unit.chunks.document_titles(
                principal.tenant_id,
                sorted({match.chunk.document_id for match in retrieval.matches}),
            )
            answer = await unit.answerer.answer(
                AnswerRequest(
                    query=question,
                    retrieval=retrieval,
                    document_titles=titles,
                    include_diagnostics=diagnostics,
                )
            )
    finally:
        await runtime.aclose()

    if answer.refused:
        print(f"REFUSED: {answer.refusal_reason}")
    else:
        print(answer.text)
        print()
        print(
            f"confidence {answer.confidence:.2f}  grounding {answer.grounding.score:.2f}  "
            f"provider {answer.provider}"
        )
        for citation in answer.citations:
            print(f"  {citation.marker} {citation.document_title} ({citation.locator})")
            print(f'      "{citation.quote[:160]}"')

    for warning in answer.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if diagnostics and answer.diagnostics is not None:
        print()
        print("diagnostics:")
        print(f"  rewrites          {list(answer.diagnostics.rewritten_queries)}")
        print(f"  dense candidates  {answer.diagnostics.dense_candidate_count}")
        print(f"  sparse candidates {answer.diagnostics.sparse_candidate_count}")
        print(f"  dropped (policy)  {answer.diagnostics.dropped_by_policy}")
        print(f"  neutralised       {answer.diagnostics.neutralised_chunks}")
        print(f"  latency           {answer.diagnostics.latency_ms}")
    return 0 if not answer.refused else 2


async def _evaluate(dataset: Path, report_path: Path | None, top_k: int) -> int:
    report = await run_dataset(
        dataset, settings=get_settings(), thresholds=Thresholds(), top_k=top_k
    )
    print(render_summary(report))
    if report_path is not None:
        write_report(report, report_path)
        print(f"\nreport written to {report_path}")
    return 0 if report.passed_gate else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="rag-assistant",
        description="Ingest documents, ask grounded questions, and run the evaluation gate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="reload on source changes")

    ingest = subparsers.add_parser("ingest", help="ingest files or directories")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.add_argument("--tenant", default=DEFAULT_TENANT)

    ask = subparsers.add_parser("query", help="ask a question of the corpus")
    ask.add_argument("question")
    ask.add_argument("--tenant", default=DEFAULT_TENANT)
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--diagnostics", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run an evaluation dataset")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    evaluate.add_argument("--top-k", type=int, default=8)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "rag_assistant.api.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
            access_log=False,
            server_header=False,
        )
        return 0

    if args.command == "ingest":
        return asyncio.run(_ingest(args.paths, args.tenant))
    if args.command == "query":
        return asyncio.run(_query(args.question, args.tenant, args.top_k, args.diagnostics))
    if args.command == "evaluate":
        return asyncio.run(_evaluate(args.dataset, args.report, args.top_k))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
