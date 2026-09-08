"""Dense retrieval over persisted vectors.

Implementation
--------------
Vectors are loaded from the database into a per-tenant NumPy matrix and scored
by a single matrix-vector product. That is an exact nearest-neighbour search: no
approximate index, no recall loss, no index build step, nothing to tune, and a
deletion takes effect the moment the cache generation changes.

Why not an ANN index
--------------------
An HNSW or IVF index is the right answer above roughly a few hundred thousand
chunks per tenant. Below that, a float32 matrix product over 100k x 384 is a
~150 MB working set and a few tens of milliseconds on one core — faster than the
embedding call that produced the query vector, and without the operational cost
of an index that must be rebuilt, tuned and monitored for recall drift.
:class:`VectorStore` is the seam where that decision is reversed: swapping in
``pgvector`` or FAISS means implementing :meth:`VectorStore.search` and nothing
else. ``ARCHITECTURE.md`` records this and the point at which to revisit it.

Cache invalidation
------------------
The cache is keyed by ``(tenant, provider)`` and carries a generation counter
that ingestion and deletion bump. A stale cache would keep answering from
deleted documents, which is a data-exposure bug rather than a performance one,
so the generation check happens on every search.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

#: Loads every ``(chunk_id, vector)`` pair for one tenant and embedding provider.
#: The store is given a loader rather than a repository because it outlives any
#: single database session: it is built once at startup and cached across
#: requests, while a session belongs to one request.
type VectorLoader = Callable[[str, str], Awaitable[tuple[list[str], list[list[float]]]]]


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One dense-retrieval result."""

    chunk_id: str
    score: float


class VectorStore(Protocol):
    """Dense nearest-neighbour search over a tenant's chunk vectors."""

    async def search(
        self, tenant_id: str, provider: str, query_vector: list[float], *, limit: int
    ) -> list[DenseHit]:
        """Return the ``limit`` nearest chunks to ``query_vector``."""
        ...

    def invalidate(self, tenant_id: str) -> None:
        """Discard any cached state for a tenant after a write."""
        ...


@dataclass
class _CacheEntry:
    matrix: npt.NDArray[np.float32]
    chunk_ids: list[str]
    generation: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class NumpyVectorStore:
    """Exact cosine search backed by the relational store, cached per tenant."""

    def __init__(self, loader: VectorLoader) -> None:
        """Bind the store to a vector loader."""
        self._loader = loader
        self._cache: dict[tuple[str, str], _CacheEntry] = {}
        self._generations: dict[str, int] = {}
        self._load_lock = asyncio.Lock()

    def invalidate(self, tenant_id: str) -> None:
        """Bump a tenant's generation so the next search reloads."""
        self._generations[tenant_id] = self._generations.get(tenant_id, 0) + 1

    async def _entry(self, tenant_id: str, provider: str) -> _CacheEntry | None:
        key = (tenant_id, provider)
        current_generation = self._generations.get(tenant_id, 0)
        cached = self._cache.get(key)
        if cached is not None and cached.generation == current_generation:
            return cached

        async with self._load_lock:
            # Re-check under the lock: a concurrent request may have loaded it.
            cached = self._cache.get(key)
            if cached is not None and cached.generation == self._generations.get(tenant_id, 0):
                return cached

            chunk_ids, vectors = await self._loader(tenant_id, provider)
            if not chunk_ids:
                self._cache.pop(key, None)
                return None

            matrix = np.asarray(vectors, dtype=np.float32)
            # Vectors are stored unnormalised because a provider may not
            # normalise; normalising once here makes cosine similarity a plain
            # dot product at query time.
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms

            entry = _CacheEntry(
                matrix=matrix,
                chunk_ids=chunk_ids,
                generation=self._generations.get(tenant_id, 0),
            )
            self._cache[key] = entry
            return entry

    async def search(
        self, tenant_id: str, provider: str, query_vector: list[float], *, limit: int
    ) -> list[DenseHit]:
        """Return the ``limit`` nearest chunks by cosine similarity."""
        entry = await self._entry(tenant_id, provider)
        if entry is None or limit <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != entry.matrix.shape[1]:
            # A dimension mismatch means the configured model changed without a
            # reindex. Returning nothing is safer than ranking against vectors
            # from a different embedding space.
            return []

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm

        scores = entry.matrix @ query
        count = min(limit, scores.shape[0])
        # argpartition is O(n) against O(n log n) for a full sort, which matters
        # once a tenant has six figures of chunks.
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]

        return [
            DenseHit(chunk_id=entry.chunk_ids[int(index)], score=float(scores[int(index)]))
            for index in top
        ]

    def cached_tenants(self) -> int:
        """Return the number of cached tenant matrices, for the readiness endpoint."""
        return len(self._cache)


__all__ = ["DenseHit", "NumpyVectorStore", "VectorLoader", "VectorStore"]
