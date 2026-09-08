"""Sparse retrieval with Okapi BM25 over the inverted index.

Dense retrieval alone fails on the queries users actually ask about a document
corpus: exact identifiers, error codes, product names, version numbers, quoted
phrases. An embedding of ``ERR_4417`` is close to an embedding of ``ERR_4418``.
BM25 is not, and that is why both are run.

The implementation reads postings from the database rather than holding an
in-memory index, so deletions take effect immediately and the ranking survives a
process restart. Term statistics — document frequency, average length — are
computed from the same rows, so they cannot drift from the corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from rag_assistant.indexing.text import terms

if TYPE_CHECKING:
    from rag_assistant.storage.repositories import ChunkRepository

#: Term-frequency saturation. 1.2 is the standard Okapi value; higher values make
#: repetition count for more, which suits long documents less well.
K1: Final[float] = 1.2
#: Length normalisation strength. 0.75 is the standard Okapi value.
B: Final[float] = 0.75


@dataclass(frozen=True, slots=True)
class SparseHit:
    """One BM25 result."""

    chunk_id: str
    score: float


class BM25Index:
    """Okapi BM25 ranking over persisted postings."""

    def __init__(self, repository: ChunkRepository, *, k1: float = K1, b: float = B) -> None:
        """Bind the index to a chunk repository."""
        self._repository = repository
        self._k1 = k1
        self._b = b

    async def search(self, tenant_id: str, query: str, *, limit: int = 40) -> list[SparseHit]:
        """Rank chunks for a query within one tenant."""
        query_terms = terms(query)
        if not query_terms:
            return []

        corpus_size, average_length = await self._repository.corpus_statistics(tenant_id)
        if corpus_size == 0:
            return []
        if average_length <= 0:
            average_length = 1.0

        postings = await self._repository.term_statistics(tenant_id, sorted(set(query_terms)))
        if not postings:
            return []

        candidate_ids = {chunk_id for entries in postings.values() for chunk_id, _ in entries}
        lengths = await self._repository.lengths(tenant_id, sorted(candidate_ids))

        scores: dict[str, float] = {}
        for term in set(query_terms):
            entries = postings.get(term)
            if not entries:
                continue
            document_frequency = len(entries)
            idf = self._idf(corpus_size, document_frequency)
            if idf <= 0.0:
                # A term present in essentially every chunk carries no signal.
                continue
            for chunk_id, frequency in entries:
                length = lengths.get(chunk_id, int(average_length)) or 1
                normalisation = self._k1 * (1.0 - self._b + self._b * (length / average_length))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                    frequency * (self._k1 + 1.0) / (frequency + normalisation)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [SparseHit(chunk_id=chunk_id, score=score) for chunk_id, score in ranked[:limit]]

    @staticmethod
    def _idf(corpus_size: int, document_frequency: int) -> float:
        """Probabilistic IDF with the standard +0.5 smoothing.

        Clamped at zero rather than allowed to go negative: a term in more than
        half the corpus would otherwise subtract score from chunks that contain
        it, which inverts the ranking for common domain vocabulary.
        """
        numerator = corpus_size - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return max(0.0, math.log(1.0 + numerator / denominator))


__all__ = ["K1", "B", "BM25Index", "SparseHit"]
