"""Local ONNX embeddings via fastembed.

This is the recommended production-quality embedding backend for a deployment
that must not call a hosted API: real transformer embeddings, quantised ONNX
weights, CPU inference, no accelerator and no credentials. Weights are
downloaded once into ``RAG_EMBEDDING__CACHE_DIR`` and reused, so a container can
be built with the model baked in and run fully offline.

fastembed's API is synchronous and CPU-bound, so calls run on the default
thread pool. The model is loaded lazily on first use rather than at import
time, which keeps process startup fast and keeps a missing optional dependency
from breaking unrelated code paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_assistant.errors import ConfigurationError, ProviderError

if TYPE_CHECKING:
    from collections.abc import Iterable

PROVIDER_NAME = "fastembed"

#: Query-side instruction prefix. BGE-family models are trained asymmetrically:
#: omitting this on the query side measurably reduces recall.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class FastEmbedProvider:
    """Transformer embeddings served locally through ONNX Runtime."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        cache_dir: Path,
        batch_size: int = 32,
        threads: int | None = None,
    ) -> None:
        """Record configuration; the model itself is loaded on first use."""
        self._model_name = model
        self._dimensions = dimensions
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._threads = threads
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Provider identifier stored with every vector."""
        return f"{PROVIDER_NAME}:{self._model_name}"

    @property
    def dimensions(self) -> int:
        """Configured vector width."""
        return self._dimensions

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            # Re-check under the lock: a concurrent caller may have loaded it
            # while this coroutine was suspended.
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            return self._model

    def _load(self) -> Any:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ConfigurationError(
                "the fastembed embedding backend requires the 'local-embeddings' extra; "
                "install it with: uv sync --extra local-embeddings"
            ) from exc

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return TextEmbedding(
            model_name=self._model_name,
            cache_dir=str(self._cache_dir),
            threads=self._threads,
        )

    def _run(self, model: Any, texts: list[str], *, is_query: bool) -> list[list[float]]:
        vectors: Iterable[Any] = (
            model.query_embed(texts)
            if is_query
            else model.embed(texts, batch_size=self._batch_size)
        )
        result: list[list[float]] = []
        for vector in vectors:
            values = [float(value) for value in vector]
            if len(values) != self._dimensions:
                raise ProviderError(
                    "fastembed embedding width does not match configuration; "
                    "set RAG_EMBEDDING__DIMENSIONS to the model's native width",
                    detail={"expected": self._dimensions, "received": len(values)},
                )
            result.append(values)
        return result

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        if not texts:
            return []
        model = await self._ensure_model()
        return await asyncio.to_thread(self._run, model, texts, is_query=False)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query, applying the model's query-side instruction prefix."""
        model = await self._ensure_model()
        vectors = await asyncio.to_thread(
            self._run, model, [_QUERY_INSTRUCTION + text], is_query=False
        )
        return vectors[0]

    async def aclose(self) -> None:
        """Drop the loaded model so its ONNX session is released."""
        self._model = None


__all__ = ["PROVIDER_NAME", "FastEmbedProvider"]
