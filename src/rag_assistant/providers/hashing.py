"""Feature-hashing embedding provider.

This is a real, if deliberately simple, embedding technique — the hashing trick
applied to word unigrams, word bigrams and character 4-grams, with sublinear
term weighting and L2 normalisation. It captures lexical and sub-word overlap
and nothing else.

Why it exists
-------------
It has no model download, no network call, no credentials and no accelerator
requirement, and it is deterministic across processes and platforms. That makes
it the right default for CI, for the first five minutes of a new contributor's
day, and for reproducible retrieval regression tests where a neural model's
version drift would make expectations meaningless.

What it is not
--------------
It is not a semantic embedding. Paraphrases with no shared surface form score
near zero. Do not deploy it as the production retrieval backend; set
``RAG_EMBEDDING__BACKEND=fastembed`` (local, still credential-free) or
``openai`` instead. :meth:`HashingEmbeddingProvider.quality_notice` returns this
warning, and the ``/readyz`` endpoint surfaces it so a production deployment
running on this backend is visible rather than silent.
"""

from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise
from typing import Final

import numpy as np

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
_CHAR_NGRAM: Final[int] = 4
#: Below this width the hashing collision rate makes similarity meaningless.
_MIN_DIMENSIONS: Final[int] = 8


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _features(text: str) -> list[tuple[str, float]]:
    """Extract weighted features: unigrams, bigrams and character 4-grams.

    Bigrams give a little word-order sensitivity; character n-grams give
    robustness to inflection and typos. Character features are down-weighted so
    they refine rather than dominate the word signal.
    """
    tokens = _tokenize(text)
    features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
    features.extend((f"b:{first}_{second}", 0.7) for first, second in pairwise(tokens))
    compact = " ".join(tokens)
    features.extend(
        (f"c:{compact[index : index + _CHAR_NGRAM]}", 0.35)
        for index in range(max(0, len(compact) - _CHAR_NGRAM + 1))
    )
    return features


def _hash_bucket(feature: str, dimensions: int) -> tuple[int, float]:
    """Map a feature to a bucket and a stable sign.

    Signed hashing keeps collisions unbiased in expectation, which is the
    standard justification for the signed variant of the hashing trick.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbeddingProvider:
    """Deterministic, dependency-free embeddings via signed feature hashing."""

    def __init__(self, dimensions: int = 384) -> None:
        """Configure the output width."""
        if dimensions < _MIN_DIMENSIONS:
            msg = f"dimensions must be at least {_MIN_DIMENSIONS}"
            raise ValueError(msg)
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        """Provider identifier stored with every vector."""
        return f"hashing:{self._dimensions}"

    @property
    def dimensions(self) -> int:
        """Vector width."""
        return self._dimensions

    @staticmethod
    def quality_notice() -> str:
        """Warning surfaced by the readiness endpoint when this backend is active."""
        return (
            "The 'hashing' embedding backend matches on lexical overlap only and cannot "
            "retrieve paraphrases. It is intended for tests and first-run setup. Set "
            "RAG_EMBEDDING__BACKEND=fastembed or openai for semantic retrieval."
        )

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self._dimensions, dtype=np.float64)
        counts: dict[int, float] = {}
        for feature, weight in _features(text):
            bucket, sign = _hash_bucket(feature, self._dimensions)
            counts[bucket] = counts.get(bucket, 0.0) + sign * weight

        for bucket, raw in counts.items():
            # Sublinear scaling: a term appearing 100 times is not 100x as
            # informative as one appearing once.
            vector[bucket] = math.copysign(1.0 + math.log1p(abs(raw)), raw)

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return [0.0] * self._dimensions
        normalised: list[float] = (vector / norm).tolist()
        return normalised

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        No instruction prefix is applied. Asymmetric prefixes help models that
        were trained with them; here they would only inject features that
        differ between the query and passage sides and reduce similarity.
        """
        return self._embed(text)

    async def aclose(self) -> None:
        """No resources are held; present to satisfy the provider protocol."""
        return


__all__ = ["HashingEmbeddingProvider"]
