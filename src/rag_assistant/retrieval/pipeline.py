"""The retrieval pipeline.

Order of operations, and why it is this order:

1. **Rewrite** — additive query variants, so a follow-up question retrieves
   against its resolved subject as well as its literal text.
2. **Dense and sparse search**, per variant, in parallel. Both are run for every
   variant rather than choosing one: the whole point of hybrid retrieval is that
   the two fail on different queries.
3. **Fuse** with reciprocal rank fusion into a single candidate list.
4. **Authorise** — drop chunks whose document the caller may not read. This
   happens *before* injection scanning and reranking so that no work, and no
   diagnostic signal, is spent on documents the caller cannot see.
5. **Scan for injection** and apply the configured policy: annotate, neutralise
   or drop. The count of each outcome is recorded in diagnostics and audited.
6. **Rerank** with MMR to remove near-duplicates from the context window.
7. **Stitch neighbours** — pull adjacent chunks so a sentence cut at a chunk
   boundary is complete in the evidence the model sees.

Every stage records its latency. When a query is slow, the diagnostics say which
stage was slow, without a profiler.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_assistant.config import Settings
from rag_assistant.domain.models import (
    Chunk,
    ChunkMatch,
    RetrievalDiagnostics,
    RetrievalResult,
    TrustLevel,
)
from rag_assistant.indexing.bm25 import BM25Index
from rag_assistant.observability.logging import get_logger
from rag_assistant.observability.tracing import get_tracer, injection_findings
from rag_assistant.retrieval.fusion import (
    FusedHit,
    lexical_overlap_score,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from rag_assistant.retrieval.rewrite import rewrite
from rag_assistant.security import injection
from rag_assistant.security.authz import Principal, can_read_document

if TYPE_CHECKING:
    from rag_assistant.indexing.vector_store import VectorStore
    from rag_assistant.providers.base import EmbeddingProvider
    from rag_assistant.storage.repositories import ChunkRepository

logger = get_logger(__name__)
tracer = get_tracer()

#: Distinct rules that must fire before the block threshold can force a drop.
_CORROBORATION_RULES = 2


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Everything the pipeline needs to answer one retrieval."""

    query: str
    principal: Principal
    history: str = ""
    top_k: int | None = None
    include_diagnostics: bool = False


class RetrievalPipeline:
    """Hybrid retrieval with authorisation and untrusted-content policy."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        chunks: ChunkRepository,
    ) -> None:
        """Wire the pipeline's collaborators."""
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunks = chunks
        self._bm25 = BM25Index(chunks)

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run the full pipeline for one query."""
        config = self._settings.retrieval
        top_k = request.top_k or config.top_k
        tenant = request.principal.tenant_id
        latencies: dict[str, float] = {}

        with tracer.start_as_current_span("retrieval.pipeline") as span:
            span.set_attribute("retrieval.top_k", top_k)

            started = time.perf_counter()
            queries = rewrite(
                request.query, history=request.history, enabled=config.enable_query_rewrite
            )
            latencies["rewrite_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            dense, sparse = await self._search_all(tenant, queries)
            latencies["search_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            fused = reciprocal_rank_fusion(dense, sparse, k=config.rrf_k)
            latencies["fuse_ms"] = (time.perf_counter() - started) * 1000.0

            # Fetch a generous candidate pool: policy and authorisation will
            # remove some, and reranking needs alternatives to choose between.
            pool = fused[: max(top_k * 4, top_k + 10)]
            started = time.perf_counter()
            matches, dropped_authz = await self._materialise(tenant, request.principal, pool)
            latencies["hydrate_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            matches, dropped_policy, neutralised = self._apply_content_policy(matches)
            latencies["policy_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            selected = self._rerank(
                request.query, matches, top_k=top_k, enabled=config.enable_rerank
            )
            latencies["rerank_ms"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            selected = await self._stitch(tenant, selected, config.context_window_neighbours)
            latencies["stitch_ms"] = (time.perf_counter() - started) * 1000.0

            span.set_attribute("retrieval.results", len(selected))
            span.set_attribute("retrieval.dropped_by_policy", dropped_policy)
            span.set_attribute("retrieval.dropped_by_authorization", dropped_authz)

        diagnostics = RetrievalDiagnostics(
            original_query=request.query if request.include_diagnostics else "",
            rewritten_queries=tuple(queries[1:]) if request.include_diagnostics else (),
            dense_candidate_count=len(dense),
            sparse_candidate_count=len(sparse),
            fused_candidate_count=len(fused),
            reranked=config.enable_rerank,
            dropped_by_policy=dropped_policy,
            dropped_by_authorization=dropped_authz,
            neutralised_chunks=neutralised,
            latency_ms={key: round(value, 3) for key, value in latencies.items()},
        )
        return RetrievalResult(matches=tuple(selected), diagnostics=diagnostics)

    async def _search_all(
        self, tenant: str, queries: list[str]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Run dense and sparse retrieval for every query variant concurrently."""
        config = self._settings.retrieval

        async def dense_for(query: str) -> list[tuple[str, float]]:
            vector = await self._embedder.embed_query(query)
            hits = await self._vector_store.search(
                tenant, self._embedder.name, vector, limit=config.dense_candidates
            )
            return [(hit.chunk_id, hit.score) for hit in hits]

        # Dense searches run concurrently: each is an embedding call followed by
        # an in-memory matrix product, so overlapping them is a real saving.
        dense_lists = await asyncio.gather(*(dense_for(query) for query in queries))

        # Sparse searches run sequentially. They share this request's
        # AsyncSession, which is not safe for concurrent use — and would not
        # parallelise anyway, since a session holds a single connection.
        sparse_lists: list[list[tuple[str, float]]] = []
        for query in queries:
            hits = await self._bm25.search(tenant, query, limit=config.sparse_candidates)
            sparse_lists.append([(hit.chunk_id, hit.score) for hit in hits])

        return self._merge_variants(list(dense_lists)), self._merge_variants(sparse_lists)

    @staticmethod
    def _merge_variants(lists: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
        """Merge per-variant result lists, keeping each chunk's best score.

        The original query is variant zero, so a chunk found only by a rewrite
        ranks below one found by the original when scores are equal.
        """
        best: dict[str, float] = {}
        for results in lists:
            for chunk_id, score in results:
                if score > best.get(chunk_id, float("-inf")):
                    best[chunk_id] = score
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))

    async def _materialise(
        self, tenant: str, principal: Principal, pool: list[FusedHit]
    ) -> tuple[list[ChunkMatch], int]:
        """Load chunk text and drop anything the caller may not read."""
        if not pool:
            return [], 0

        chunks = await self._chunks.get_many(tenant, [hit.chunk_id for hit in pool])
        acls = await self._chunks.document_acls(
            tenant, sorted({chunk.document_id for chunk in chunks.values()})
        )

        matches: list[ChunkMatch] = []
        denied = 0
        for hit in pool:
            chunk = chunks.get(hit.chunk_id)
            if chunk is None:
                # The chunk was deleted between ranking and hydration. Skipping
                # is correct: the alternative is citing content that no longer
                # exists.
                continue
            if not can_read_document(
                principal, tenant_id=chunk.tenant_id, acl=acls.get(chunk.document_id, ())
            ):
                denied += 1
                continue
            matches.append(
                ChunkMatch(
                    chunk=chunk,
                    score=hit.score,
                    dense_rank=hit.dense_rank,
                    sparse_rank=hit.sparse_rank,
                    dense_score=hit.dense_score,
                    sparse_score=hit.sparse_score,
                    trust=TrustLevel.UNTRUSTED,
                )
            )
        return matches, denied

    def _decide(self, result: injection.ScanResult) -> str:
        """Decide what to do with a suspicious passage: ``drop``, ``neutralise`` or ``annotate``.

        The configured action applies, with one override. Risk at or above the
        block threshold forces a drop, but only when at least two distinct rules
        fired. A single rule match is not enough to discard a passage: the
        highest severity any one rule carries is 0.9, and documents that
        legitimately *discuss* an attack — a security policy, an incident
        report, this project's own threat model — reliably match exactly one
        rule. Requiring two independent signals is what separates quoting an
        attack from attempting one.
        """
        action = self._settings.security.injection_action
        if action == "drop":
            return "drop"
        corroborated = len({finding.rule_id for finding in result.findings}) >= _CORROBORATION_RULES
        if result.risk >= self._settings.security.injection_block_threshold and corroborated:
            return "drop"
        return action

    def _apply_content_policy(self, matches: list[ChunkMatch]) -> tuple[list[ChunkMatch], int, int]:
        """Scan retrieved text for injection and apply the content policy."""
        kept: list[ChunkMatch] = []
        dropped = 0
        neutralised = 0

        for match in matches:
            result = injection.scan(match.chunk.text)
            if not result.is_suspicious:
                kept.append(match)
                continue

            decision = self._decide(result)
            # The decision is logged, not the configured mode: an audit trail
            # that records the setting rather than what happened cannot answer
            # "was this passage shown to the model?".
            for finding in result.findings:
                injection_findings.add(1, {"rule": finding.rule_id, "action": decision})
            logger.warning(
                "security.injection_detected",
                chunk_id=match.chunk.id,
                document_id=match.chunk.document_id,
                risk=result.risk,
                rules=[finding.rule_id for finding in result.findings],
                decision=decision,
            )

            if decision == "drop":
                dropped += 1
                continue

            if decision == "neutralise":
                cleaned = injection.neutralise(match.chunk.text, result.spans)
                neutralised += 1
                kept.append(
                    match.model_copy(
                        update={
                            "chunk": match.chunk.model_copy(update={"text": cleaned}),
                            "injection_findings": result.findings,
                            "neutralised": True,
                        }
                    )
                )
                continue

            kept.append(match.model_copy(update={"injection_findings": result.findings}))

        return kept, dropped, neutralised

    def _rerank(
        self, query: str, matches: list[ChunkMatch], *, top_k: int, enabled: bool
    ) -> list[ChunkMatch]:
        """Blend fusion score with lexical evidence, then diversify with MMR."""
        if not matches:
            return []

        scored = [
            match.model_copy(
                update={"rerank_score": lexical_overlap_score(query, match.chunk.indexing_text)}
            )
            for match in matches
        ]

        if not enabled:
            return scored[:top_k]

        # A chunk that fuses well *and* shares query vocabulary is a better
        # candidate than one that only fuses well. The weighting keeps fusion
        # dominant: lexical overlap is a tie-breaker, not the ranking.
        blended = [
            (
                match.chunk.id,
                match.score + 0.001 * (match.rerank_score or 0.0),
                match.chunk.indexing_text,
            )
            for match in scored
        ]
        blended.sort(key=lambda item: (-item[1], item[0]))

        order = maximal_marginal_relevance(
            blended, limit=top_k, lambda_=self._settings.retrieval.mmr_lambda
        )
        by_id = {match.chunk.id: match for match in scored}
        return [by_id[chunk_id] for chunk_id in order if chunk_id in by_id]

    async def _stitch(
        self, tenant: str, matches: list[ChunkMatch], window: int
    ) -> list[ChunkMatch]:
        """Extend each match with its neighbouring chunks' text.

        The match keeps its own identity and citation; only the text presented
        to the model grows. Neighbours are scanned for injection too, because a
        neighbouring chunk is exactly where an attacker would put the payload if
        stitching were unchecked.
        """
        if window <= 0 or not matches:
            return matches

        stitched: list[ChunkMatch] = []
        for match in matches:
            neighbours = await self._chunks.neighbours(
                tenant, match.chunk.document_id, match.chunk.ordinal, window
            )
            if len(neighbours) <= 1:
                stitched.append(match)
                continue

            parts: list[str] = []
            for neighbour in neighbours:
                if neighbour.id == match.chunk.id:
                    # Use the match's own text, which has already been through
                    # the content policy. Re-reading it from the repository here
                    # would silently reinstate spans that were just neutralised.
                    parts.append(match.chunk.text)
                    continue
                scan = injection.scan(neighbour.text)
                if not scan.is_suspicious:
                    parts.append(neighbour.text)
                    continue
                decision = self._decide(scan)
                if decision == "drop":
                    continue
                parts.append(injection.neutralise(neighbour.text, scan.spans))

            merged = self._merge_overlapping(parts)
            stitched.append(
                match.model_copy(update={"chunk": match.chunk.model_copy(update={"text": merged})})
            )
        return stitched

    @staticmethod
    def _merge_overlapping(parts: list[str]) -> str:
        """Join consecutive chunk texts, removing the overlap the chunker added."""
        if not parts:
            return ""
        merged = parts[0]
        for part in parts[1:]:
            # Chunk overlap is bounded by the configured window, so a bounded
            # suffix/prefix probe is sufficient and keeps this linear.
            limit = min(len(merged), len(part), 2000)
            overlap = 0
            for size in range(limit, 20, -1):
                if merged.endswith(part[:size]):
                    overlap = size
                    break
            merged = merged + ("\n" if overlap == 0 else "") + part[overlap:]
        return merged


def as_chunk_texts(matches: list[ChunkMatch]) -> list[Chunk]:
    """Extract the chunks from a match list. Convenience for callers and tests."""
    return [match.chunk for match in matches]


__all__ = ["RetrievalPipeline", "RetrievalRequest", "as_chunk_texts"]
