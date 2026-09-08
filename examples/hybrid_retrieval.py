#!/usr/bin/env python3
"""Show what each retriever contributes, and why both are run.

Dense and sparse retrieval fail on different queries. This script runs each
retriever alone and then the fused pipeline over the same corpus, and prints
the ranking each produced, so the difference is visible rather than asserted.

    uv run python examples/hybrid_retrieval.py
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
    RetrievalSettings,
    SecuritySettings,
    Settings,
    StorageSettings,
)
from rag_assistant.domain.models import SourceKind
from rag_assistant.indexing.bm25 import BM25Index
from rag_assistant.ingestion.pipeline import IngestionRequest
from rag_assistant.observability.logging import configure_logging
from rag_assistant.retrieval.fusion import reciprocal_rank_fusion
from rag_assistant.retrieval.pipeline import RetrievalRequest
from rag_assistant.runtime import Runtime
from rag_assistant.security.authz import Principal

CORPUS = Path(__file__).resolve().parents[1] / "data" / "regression" / "corpus"

QUERIES = [
    "ERR_4417",
    "the document I uploaded was too big to accept",
    "how long do I have to send something back",
]


def build_settings(database: Path) -> Settings:
    """Configuration for a throwaway index with reranking left on."""
    return Settings(
        storage=StorageSettings(database_url=f"sqlite+aiosqlite:///{database}"),
        embedding=EmbeddingSettings(backend=EmbeddingBackend.HASHING, dimensions=384),
        chat=ChatSettings(backend=ChatBackend.EXTRACTIVE),
        retrieval=RetrievalSettings(top_k=4, dense_candidates=20, sparse_candidates=20),
        security=SecuritySettings(require_api_key=False, api_keys=()),
    )


def describe(chunk_ids: list[str], titles: dict[str, str], limit: int = 4) -> str:
    """Render a ranking as a short, readable list."""
    if not chunk_ids:
        return "      (nothing)"
    return "\n".join(
        f"      {rank}. {titles.get(chunk_id, chunk_id)}"
        for rank, chunk_id in enumerate(chunk_ids[:limit], start=1)
    )


async def main() -> None:
    """Compare dense, sparse and fused rankings for several queries."""
    with tempfile.TemporaryDirectory(prefix="rag-hybrid-") as scratch:
        runtime = Runtime(build_settings(Path(scratch) / "hybrid.db"))
        await runtime.start()
        principal = Principal(tenant_id="demo", key_id="example")

        try:
            for path in sorted(CORPUS.glob("*.md")):
                await runtime.ingestion.ingest(
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

            for query in QUERIES:
                print()
                print("=" * 78)
                print("query:", query)

                async with runtime.unit_of_work() as unit:
                    vector = await runtime.embedder.embed_query(query)
                    dense = await runtime.vector_store.search(
                        principal.tenant_id, runtime.embedder.name, vector, limit=20
                    )
                    sparse = await BM25Index(unit.chunks).search(
                        principal.tenant_id, query, limit=20
                    )
                    fused = reciprocal_rank_fusion(
                        [(hit.chunk_id, hit.score) for hit in dense],
                        [(hit.chunk_id, hit.score) for hit in sparse],
                    )
                    pipeline = await unit.retrieval.retrieve(
                        RetrievalRequest(query=query, principal=principal, top_k=4)
                    )

                    every_id = (
                        [hit.chunk_id for hit in dense]
                        + [hit.chunk_id for hit in sparse]
                        + [hit.chunk_id for hit in fused]
                    )
                    chunks = await unit.chunks.get_many(principal.tenant_id, every_id)

                titles = {
                    chunk_id: f"{chunk.metadata.get('document_title', '?')}  {chunk.locator}"
                    for chunk_id, chunk in chunks.items()
                }

                print("   dense only (cosine over embeddings):")
                print(describe([hit.chunk_id for hit in dense], titles))
                print("   sparse only (BM25 over the inverted index):")
                print(describe([hit.chunk_id for hit in sparse], titles))
                print("   fused (reciprocal rank fusion):")
                print(describe([hit.chunk_id for hit in fused], titles))
                print("   after the full pipeline (authorisation, policy, MMR, stitching):")
                print(
                    describe(
                        [match.chunk.id for match in pipeline.matches],
                        {
                            match.chunk.id: titles.get(match.chunk.id, "?")
                            for match in pipeline.matches
                        },
                    )
                )

            print()
            print("=" * 78)
            print(
                "The default embedding backend here is 'hashing', which matches on\n"
                "lexical overlap only. Set RAG_EMBEDDING__BACKEND=fastembed to see the\n"
                "dense side handle paraphrase, which is where it earns its place."
            )
        finally:
            await runtime.aclose()


if __name__ == "__main__":
    # A library does not configure logging for its host, so an example that runs
    # the runtime directly has to do it.
    configure_logging(level="ERROR", fmt="console")
    asyncio.run(main())
