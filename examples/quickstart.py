#!/usr/bin/env python3
"""Ingest a corpus and ask a question, entirely in process.

Runs against a temporary index with the zero-setup backends, so it needs no
services, no credentials and no model download.

    uv run python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from rag_assistant.config import (
    ChatBackend,
    ChatSettings,
    EmbeddingBackend,
    EmbeddingSettings,
    Environment,
    SecuritySettings,
    Settings,
    StorageSettings,
)
from rag_assistant.domain.models import SourceKind
from rag_assistant.generation.answerer import AnswerRequest
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.observability.logging import configure_logging
from rag_assistant.retrieval.pipeline import RetrievalRequest
from rag_assistant.runtime import Runtime
from rag_assistant.security.authz import Principal

CORPUS = Path(__file__).resolve().parents[1] / "data" / "regression" / "corpus"

QUESTIONS = [
    "How many days do customers have to request a refund?",
    "What does ERR_4417 mean?",
    "How long does international shipping take?",
    "What is the company's cryptocurrency treasury policy?",
]


def build_settings(database: Path) -> Settings:
    """Configuration for a throwaway, credential-free index."""
    return Settings(
        environment=Environment.LOCAL,
        storage=StorageSettings(database_url=f"sqlite+aiosqlite:///{database}"),
        embedding=EmbeddingSettings(backend=EmbeddingBackend.HASHING, dimensions=384),
        chat=ChatSettings(backend=ChatBackend.EXTRACTIVE),
        security=SecuritySettings(require_api_key=False, api_keys=()),
    )


async def main() -> None:
    """Ingest the sample corpus and answer a few questions against it."""
    with tempfile.TemporaryDirectory(prefix="rag-quickstart-") as scratch:
        runtime = Runtime(build_settings(Path(scratch) / "quickstart.db"))
        await runtime.start()
        principal = Principal(tenant_id="demo", key_id="example")

        try:
            print("Ingesting", CORPUS)
            for path in sorted(CORPUS.glob("*.md")):
                report = await runtime.ingestion.ingest(
                    IngestionRequest(
                        payload=path.read_bytes(),
                        filename=path.name,
                        declared_media_type="text/markdown",
                        principal=principal,
                        source_kind=SourceKind.UPLOAD,
                        source_ref=path.name,
                        title=path.name,
                    )
                )
                print(f"  {path.name:<32} {report.chunks_created} chunks")

            for question in QUESTIONS:
                print()
                print("=" * 78)
                print("Q:", question)
                async with runtime.unit_of_work() as unit:
                    retrieval = await unit.retrieval.retrieve(
                        RetrievalRequest(query=question, principal=principal, top_k=6)
                    )
                    titles = await unit.chunks.document_titles(
                        principal.tenant_id,
                        sorted({match.chunk.document_id for match in retrieval.matches}),
                    )
                    answer = await unit.answerer.answer(
                        AnswerRequest(query=question, retrieval=retrieval, document_titles=titles)
                    )

                if answer.refused:
                    print("REFUSED:", answer.refusal_reason)
                    continue

                print("A:", answer.text)
                print(
                    f"   confidence={answer.confidence:.2f} "
                    f"grounding={answer.grounding.score:.2f} "
                    f"provider={answer.provider}"
                )
                for citation in answer.citations:
                    print(f"   {citation.marker} {citation.document_title} ({citation.locator})")
                    print(f'       "{citation.quote[:120]}"')
                for warning in answer.warnings:
                    print("   warning:", warning)
        finally:
            await runtime.aclose()


if __name__ == "__main__":
    # A library does not configure logging for its host, so an example that runs
    # the runtime directly has to do it.
    configure_logging(level="ERROR", fmt="console")
    asyncio.run(main())
