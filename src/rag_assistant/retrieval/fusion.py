"""Rank fusion and reranking.

Reciprocal rank fusion
----------------------
Dense and sparse retrievers produce scores on incomparable scales: a cosine
similarity of 0.82 and a BM25 score of 14.3 cannot be added or averaged without
inventing a normalisation that changes behaviour whenever the corpus changes.
RRF sidesteps the problem by combining *ranks* rather than scores, which is why
it is the default fusion method in most production hybrid systems:

    score(d) = sum over retrievers of 1 / (k + rank(d))

``k`` (default 60) damps the influence of the very top ranks so a single
retriever's first result cannot dominate the fused list.

Maximal marginal relevance
--------------------------
Fusion optimises relevance per chunk, which reliably produces a context window
of eight near-identical passages when a corpus contains duplicated boilerplate.
MMR re-selects greedily, trading a controlled amount of relevance for coverage:

    select argmax [ lambda * relevance(d) - (1 - lambda) * max similarity(d, S) ]

That matters for grounding as much as for quality — an answer supported by five
copies of one paragraph is not five times better supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag_assistant.indexing.text import terms

#: Lexical similarity at or above which a candidate is treated as a restatement
#: of an already-selected passage and skipped outright.
_NEAR_DUPLICATE_CEILING = 0.9


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A candidate after fusion, retaining each retriever's contribution."""

    chunk_id: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


def reciprocal_rank_fusion(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    k: int = 60,
) -> list[FusedHit]:
    """Fuse two ranked lists by reciprocal rank."""
    dense_ranks = {chunk_id: index for index, (chunk_id, _) in enumerate(dense, start=1)}
    sparse_ranks = {chunk_id: index for index, (chunk_id, _) in enumerate(sparse, start=1)}
    dense_scores = dict(dense)
    sparse_scores = dict(sparse)

    fused: list[FusedHit] = []
    for chunk_id in dense_ranks.keys() | sparse_ranks.keys():
        score = 0.0
        if (rank := dense_ranks.get(chunk_id)) is not None:
            score += 1.0 / (k + rank)
        if (rank := sparse_ranks.get(chunk_id)) is not None:
            score += 1.0 / (k + rank)
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                score=score,
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                dense_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
            )
        )
    # Ties break on chunk id so ordering is deterministic across runs, which is
    # what makes retrieval regression tests meaningful.
    fused.sort(key=lambda hit: (-hit.score, hit.chunk_id))
    return fused


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def maximal_marginal_relevance(
    candidates: Sequence[tuple[str, float, str]],
    *,
    limit: int,
    lambda_: float = 0.7,
) -> list[str]:
    """Select a relevant but non-redundant subset.

    ``candidates`` are ``(chunk_id, relevance, text)`` triples. Similarity
    between candidates is lexical (Jaccard over index terms) rather than
    embedding-based: it needs no extra model call, is deterministic, and is more
    than sufficient for detecting the near-duplicate passages MMR exists to
    suppress.
    """
    if limit <= 0 or not candidates:
        return []

    term_sets = {chunk_id: set(terms(text)) for chunk_id, _, text in candidates}
    relevance = {chunk_id: score for chunk_id, score, _ in candidates}

    remaining = [chunk_id for chunk_id, _, _ in candidates]
    if not remaining:
        return []

    # Relevance arrives on the RRF scale; rescale to [0, 1] so lambda_ means the
    # same thing regardless of how many retrievers contributed.
    values = np.asarray([relevance[chunk_id] for chunk_id in remaining], dtype=np.float64)
    spread = float(values.max() - values.min())
    scaled = (
        dict.fromkeys(remaining, 1.0)
        if spread == 0.0
        else {
            chunk_id: float((relevance[chunk_id] - values.min()) / spread) for chunk_id in remaining
        }
    )

    selected: list[str] = [remaining.pop(0)]
    while remaining and len(selected) < limit:
        best_id: str | None = None
        best_score = float("-inf")
        for chunk_id in remaining:
            redundancy = max(
                _jaccard(term_sets[chunk_id], term_sets[chosen]) for chosen in selected
            )
            # A passage that is a near-exact restatement of one already chosen
            # adds no information at any lambda, and corpora are full of them:
            # a policy quoted in three documents, a boilerplate footer, an
            # overlapping chunk boundary. Spending a context slot on it is
            # strictly worse than spending it on anything else, so the ceiling
            # applies independently of the relevance/diversity trade-off.
            if redundancy >= _NEAR_DUPLICATE_CEILING:
                continue
            score = lambda_ * scaled[chunk_id] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score, best_id = score, chunk_id
        if best_id is None:
            # Everything left is a near-duplicate of something already selected.
            # Returning fewer, distinct passages is the correct outcome.
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def lexical_overlap_score(query: str, text: str) -> float:
    """Fraction of the query's distinct terms present in ``text``.

    Used as a cheap, explainable reranking signal and as an input to grounding.
    It is not a semantic measure and is never used alone to decide relevance.
    """
    query_terms = set(terms(query))
    if not query_terms:
        return 0.0
    return len(query_terms & set(terms(text))) / len(query_terms)


__all__ = [
    "FusedHit",
    "lexical_overlap_score",
    "maximal_marginal_relevance",
    "reciprocal_rank_fusion",
]
