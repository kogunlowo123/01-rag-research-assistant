"""Fusion, reranking and query rewriting."""

from __future__ import annotations

import pytest

from rag_assistant.retrieval.fusion import (
    lexical_overlap_score,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from rag_assistant.retrieval.rewrite import (
    decompose,
    expand_identifiers,
    needs_context,
    rewrite,
)

pytestmark = pytest.mark.unit


class TestReciprocalRankFusion:
    def test_a_document_ranked_by_both_retrievers_outranks_one_ranked_by_either(self) -> None:
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("c", 12.0), ("d", 9.0), ("a", 3.0)]
        fused = reciprocal_rank_fusion(dense, sparse)
        assert fused[0].chunk_id in {"a", "c"}
        by_id = {hit.chunk_id: hit for hit in fused}
        assert by_id["a"].score > by_id["b"].score
        assert by_id["c"].score > by_id["d"].score

    def test_scores_from_incomparable_scales_are_never_added(self) -> None:
        """BM25 scores are unbounded; a large one must not dominate the fusion."""
        dense = [("a", 0.99)]
        sparse = [("b", 5000.0)]
        fused = {hit.chunk_id: hit.score for hit in reciprocal_rank_fusion(dense, sparse)}
        assert fused["a"] == pytest.approx(fused["b"])

    def test_each_retriever_contribution_is_retained(self) -> None:
        fused = reciprocal_rank_fusion([("a", 0.5)], [("a", 4.0)])
        hit = fused[0]
        assert hit.dense_rank == 1
        assert hit.sparse_rank == 1
        assert hit.dense_score == 0.5
        assert hit.sparse_score == 4.0

    def test_a_document_found_by_only_one_retriever_still_appears(self) -> None:
        fused = reciprocal_rank_fusion([("a", 0.5)], [("b", 4.0)])
        assert {hit.chunk_id for hit in fused} == {"a", "b"}

    def test_ordering_is_deterministic_for_tied_scores(self) -> None:
        dense = [("z", 0.5), ("y", 0.5)]
        sparse = [("y", 1.0), ("z", 1.0)]
        first = [hit.chunk_id for hit in reciprocal_rank_fusion(dense, sparse)]
        second = [hit.chunk_id for hit in reciprocal_rank_fusion(dense, sparse)]
        assert first == second

    def test_empty_inputs_produce_no_results(self) -> None:
        assert reciprocal_rank_fusion([], []) == []


class TestMaximalMarginalRelevance:
    NEAR_DUPLICATES = [
        ("a", 1.0, "Refunds are issued within thirty days of purchase."),
        ("b", 0.99, "Refunds are issued within thirty days of the purchase."),
        ("c", 0.98, "Refunds are issued within thirty days after purchase."),
        ("d", 0.5, "Standard shipping takes three to seven business days."),
    ]

    def test_near_duplicates_are_suppressed_in_favour_of_coverage(self) -> None:
        selected = maximal_marginal_relevance(self.NEAR_DUPLICATES, limit=2, lambda_=0.5)
        assert selected[0] == "a"
        assert "d" in selected

    def test_near_exact_restatements_are_skipped_at_any_lambda(self) -> None:
        """A duplicate adds no information, so pure-relevance mode drops it too."""
        selected = maximal_marginal_relevance(self.NEAR_DUPLICATES, limit=2, lambda_=1.0)
        assert selected == ["a", "d"]

    def test_higher_lambda_favours_relevance_among_distinct_candidates(self) -> None:
        candidates = [
            ("a", 1.0, "Refunds are issued within thirty days of purchase."),
            ("b", 0.9, "Refund requests require proof of purchase and an order number."),
            ("c", 0.1, "Standard shipping takes three to seven business days."),
        ]
        assert maximal_marginal_relevance(candidates, limit=2, lambda_=1.0) == ["a", "b"]
        assert maximal_marginal_relevance(candidates, limit=2, lambda_=0.0) == ["a", "c"]

    def test_limit_is_respected(self) -> None:
        distinct = [
            ("a", 1.0, "Refunds are issued within thirty days of purchase."),
            ("b", 0.8, "Standard shipping takes three to seven business days."),
            ("c", 0.6, "Severity one incidents receive a response within fifteen minutes."),
            ("d", 0.4, "Gift cards and store credit cannot be returned."),
        ]
        assert len(maximal_marginal_relevance(distinct, limit=3)) == 3

    def test_a_set_of_pure_duplicates_collapses_to_one(self) -> None:
        duplicates = [
            (f"c{i}", 1.0 - i * 0.01, "The same sentence repeated verbatim here.") for i in range(4)
        ]
        assert maximal_marginal_relevance(duplicates, limit=4) == ["c0"]

    def test_empty_candidates_and_zero_limit_are_safe(self) -> None:
        assert maximal_marginal_relevance([], limit=5) == []
        assert maximal_marginal_relevance(self.NEAR_DUPLICATES, limit=0) == []

    def test_identical_relevance_scores_do_not_divide_by_zero(self) -> None:
        candidates = [
            ("a", 0.5, "Refunds are issued within thirty days of purchase."),
            ("b", 0.5, "Standard shipping takes three to seven business days."),
            ("c", 0.5, "Severity one incidents receive a response within fifteen minutes."),
            ("d", 0.5, "Gift cards and store credit cannot be returned."),
        ]
        assert len(maximal_marginal_relevance(candidates, limit=3)) == 3


class TestLexicalOverlap:
    def test_full_overlap_scores_one(self) -> None:
        assert lexical_overlap_score("refund window", "the refund window is 30 days") == 1.0

    def test_no_overlap_scores_zero(self) -> None:
        assert lexical_overlap_score("refund window", "shipping takes seven days") == 0.0

    def test_empty_query_scores_zero(self) -> None:
        assert lexical_overlap_score("   ", "anything") == 0.0

    def test_partial_overlap_is_between(self) -> None:
        score = lexical_overlap_score("refund shipping window", "the refund window")
        assert 0.0 < score < 1.0


class TestQueryRewrite:
    def test_the_original_query_is_always_first(self) -> None:
        assert rewrite("what is the refund window?")[0] == "what is the refund window?"

    def test_disabled_rewriting_returns_only_the_original(self) -> None:
        assert rewrite("anything at all", history="Q: x\nA: y", enabled=False) == [
            "anything at all"
        ]

    def test_empty_query_yields_nothing(self) -> None:
        assert rewrite("   ") == []

    def test_a_follow_up_is_expanded_with_the_previous_turn(self) -> None:
        history = "Q: What is the refund policy?\nA: Refunds are available for 30 days."
        variants = rewrite("what about digital goods?", history=history)
        assert len(variants) > 1
        assert any("refund" in variant.lower() for variant in variants[1:])

    def test_a_self_contained_query_is_not_expanded_with_history(self) -> None:
        history = "Q: What is the refund policy?\nA: Refunds last 30 days."
        variants = rewrite(
            "How much does express international shipping cost for heavy parcels?",
            history=history,
        )
        assert not any("refund" in variant.lower() for variant in variants)

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("what about it?", True),
            ("and how long does that take?", True),
            ("what is the refund window for digital goods sold in Europe?", False),
            ("", False),
        ],
    )
    def test_context_dependence_detection(self, query: str, expected: bool) -> None:
        assert needs_context(query) is expected

    def test_identifier_surface_forms_are_expanded(self) -> None:
        variants = expand_identifiers("What does ERR_4417 mean?")
        assert any("ERR 4417" in variant for variant in variants)

    def test_compound_questions_are_decomposed(self) -> None:
        parts = decompose("What is the refund window and how long does shipping take?")
        assert len(parts) == 2
        assert "refund" in parts[0]
        assert "shipping" in parts[1]

    def test_a_simple_question_is_not_decomposed(self) -> None:
        assert decompose("What is the refund window for digital goods?") == []

    def test_the_number_of_variants_is_bounded(self) -> None:
        history = "Q: A long previous question about many topics\nA: A long answer"
        variants = rewrite(
            "and what about it and how does that work and why is ERR_4417 raised?",
            history=history,
        )
        assert len(variants) <= 4

    def test_variants_are_deduplicated(self) -> None:
        variants = rewrite("refund window")
        assert len(variants) == len(set(variants))
