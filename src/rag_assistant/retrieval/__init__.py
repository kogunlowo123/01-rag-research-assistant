"""Retrieval: query rewriting, hybrid search, fusion and reranking."""

from rag_assistant.retrieval.fusion import (
    FusedHit,
    lexical_overlap_score,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from rag_assistant.retrieval.pipeline import RetrievalPipeline, RetrievalRequest
from rag_assistant.retrieval.rewrite import rewrite

__all__ = [
    "FusedHit",
    "RetrievalPipeline",
    "RetrievalRequest",
    "lexical_overlap_score",
    "maximal_marginal_relevance",
    "reciprocal_rank_fusion",
    "rewrite",
]
