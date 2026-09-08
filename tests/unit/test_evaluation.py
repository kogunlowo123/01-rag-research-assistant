"""Evaluation dataset format, metrics and threshold gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_assistant.domain.models import (
    Answer,
    Chunk,
    ChunkMatch,
    Citation,
    GroundingReport,
    RetrievalDiagnostics,
    RetrievalResult,
)
from rag_assistant.evaluation.dataset import CaseCategory, EvalCase, EvalDataset
from rag_assistant.evaluation.metrics import (
    CaseResult,
    EvaluationReport,
    Thresholds,
    citation_precision,
    recall_at_k,
    reciprocal_rank,
)
from rag_assistant.evaluation.runner import render_summary, score_case, write_report

pytestmark = pytest.mark.unit


def make_result(**overrides: object) -> CaseResult:
    defaults: dict[str, object] = {
        "case_id": "c1",
        "category": CaseCategory.FACTUAL,
        "passed": True,
        "failures": (),
        "recall_at_k": 1.0,
        "reciprocal_rank": 1.0,
        "citation_precision": 1.0,
        "grounding": 1.0,
        "refused": False,
        "citations": 1,
        "latency_ms": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "provider": "extractive",
    }
    return CaseResult(**{**defaults, **overrides})  # type: ignore[arg-type]


def make_answer(
    *,
    text: str = "Refunds last 30 days. [1]",
    refused: bool = False,
    grounding: float = 1.0,
    titles: tuple[str, ...] = ("policy.md",),
) -> Answer:
    return Answer(
        query="q",
        text="" if refused else text,
        refused=refused,
        refusal_reason="no evidence" if refused else None,
        citations=()
        if refused
        else tuple(
            Citation(
                marker=f"[{index}]",
                document_id=f"doc_{index}",
                document_title=title,
                chunk_id=f"chk_{index}",
                locator="#0",
                quote="quote",
                support_score=1.0,
            )
            for index, title in enumerate(titles, start=1)
        ),
        grounding=GroundingReport(
            total_sentences=1,
            supported_sentences=1 if grounding >= 1.0 else 0,
            unsupported_sentences=(),
            score=grounding,
            citation_coverage=1.0,
        ),
        provider="extractive",
    )


def make_retrieval(*document_ids: str) -> RetrievalResult:
    return RetrievalResult(
        matches=tuple(
            ChunkMatch(
                chunk=Chunk(
                    id=f"chk_{index}",
                    document_id=document_id,
                    tenant_id="evaluation",
                    ordinal=index,
                    text="Refunds last 30 days.",
                    token_estimate=6,
                ),
                score=1.0 - index * 0.1,
            )
            for index, document_id in enumerate(document_ids)
        ),
        diagnostics=RetrievalDiagnostics(original_query="q"),
    )


class TestRankingMetrics:
    def test_recall_counts_only_the_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
        assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0

    def test_recall_is_a_fraction_of_the_expected_set(self) -> None:
        assert recall_at_k(["a", "b"], ["a", "z"], 5) == 0.5

    def test_a_case_with_no_expectation_does_not_penalise_recall(self) -> None:
        assert recall_at_k([], [], 5) == 1.0

    def test_reciprocal_rank_rewards_an_earlier_hit(self) -> None:
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
        assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
        assert reciprocal_rank(["a", "b"], ["z"]) == 0.0

    def test_citation_precision_penalises_citing_everything(self) -> None:
        assert citation_precision(["a", "b", "c", "d"], ["a"]) == 0.25
        assert citation_precision(["a"], ["a"]) == 1.0

    def test_no_citation_is_zero_precision(self) -> None:
        assert citation_precision([], ["a"]) == 0.0


class TestScoreCase:
    def test_a_correct_answer_passes(self) -> None:
        case = EvalCase(
            id="c1",
            query="refund window",
            expected_documents=("policy.md",),
            must_contain=("30 days",),
        )
        result = score_case(
            case,
            make_answer(),
            make_retrieval("doc_1"),
            {"doc_1": "policy.md"},
            top_k=8,
        )
        assert result.passed
        assert result.failures == ()

    def test_missing_expected_content_fails(self) -> None:
        case = EvalCase(id="c1", query="q", must_contain=("14 days",))
        result = score_case(case, make_answer(), make_retrieval("doc_1"), {"doc_1": "p"}, top_k=8)
        assert "missing_expected_content" in result.failures

    def test_forbidden_content_fails(self) -> None:
        case = EvalCase(id="c1", query="q", must_not_contain=("30 days",))
        result = score_case(case, make_answer(), make_retrieval("doc_1"), {"doc_1": "p"}, top_k=8)
        assert "contains_forbidden_content" in result.failures

    def test_an_expected_refusal_that_answers_fails(self) -> None:
        case = EvalCase(id="c1", query="q", must_refuse=True, require_citation=False)
        result = score_case(case, make_answer(), make_retrieval("doc_1"), {"doc_1": "p"}, top_k=8)
        assert "expected_refusal_but_answered" in result.failures

    def test_an_unexpected_refusal_fails(self) -> None:
        case = EvalCase(id="c1", query="q")
        result = score_case(
            case, make_answer(refused=True), make_retrieval("doc_1"), {"doc_1": "p"}, top_k=8
        )
        assert "unexpected_refusal" in result.failures

    def test_a_correct_refusal_passes(self) -> None:
        case = EvalCase(id="c1", query="q", must_refuse=True, require_citation=False)
        result = score_case(case, make_answer(refused=True), make_retrieval(), {}, top_k=8)
        assert result.passed

    def test_grounding_below_the_case_minimum_fails(self) -> None:
        case = EvalCase(id="c1", query="q", min_grounding=0.9)
        result = score_case(
            case,
            make_answer(grounding=0.4),
            make_retrieval("doc_1"),
            {"doc_1": "p"},
            top_k=8,
        )
        assert "grounding_below_case_minimum" in result.failures

    def test_a_missing_expected_document_fails(self) -> None:
        case = EvalCase(id="c1", query="q", expected_documents=("other.md",))
        result = score_case(
            case, make_answer(), make_retrieval("doc_1"), {"doc_1": "policy.md"}, top_k=8
        )
        assert "retrieval_missed_expected_document" in result.failures
        assert result.recall_at_k == 0.0

    def test_a_missing_citation_fails_when_required(self) -> None:
        answer = make_answer().model_copy(update={"citations": ()})
        case = EvalCase(id="c1", query="q", require_citation=True)
        result = score_case(case, answer, make_retrieval("doc_1"), {"doc_1": "p"}, top_k=8)
        assert "no_resolvable_citation" in result.failures


class TestEvaluationReport:
    @staticmethod
    def _report(results: list[CaseResult], **thresholds: float) -> EvaluationReport:
        return EvaluationReport(
            dataset="d",
            results=results,
            thresholds=Thresholds(**thresholds),
        )

    def test_an_all_passing_run_meets_the_gate(self) -> None:
        report = self._report([make_result(), make_result(case_id="c2")])
        assert report.pass_rate == 1.0
        assert report.passed_gate
        assert report.threshold_failures() == []

    def test_a_low_pass_rate_fails_the_gate(self) -> None:
        report = self._report(
            [make_result(), make_result(case_id="c2", passed=False, failures=("x",))]
        )
        assert not report.passed_gate
        assert any("pass rate" in failure for failure in report.threshold_failures())

    def test_low_recall_fails_the_gate(self) -> None:
        report = self._report([make_result(recall_at_k=0.1)])
        assert any("recall" in failure for failure in report.threshold_failures())

    def test_low_grounding_fails_the_gate(self) -> None:
        report = self._report([make_result(grounding=0.1)])
        assert any("grounding" in failure for failure in report.threshold_failures())

    def test_a_single_adversarial_failure_fails_the_gate(self) -> None:
        """Adversarial cases are gated at 100 percent by design."""
        results = [make_result(case_id=f"c{i}") for i in range(20)]
        results.append(
            make_result(
                case_id="adv",
                category=CaseCategory.ADVERSARIAL,
                passed=False,
                failures=("contains_forbidden_content",),
            )
        )
        report = self._report(results)
        assert not report.passed_gate
        assert any("adversarial" in failure for failure in report.threshold_failures())

    def test_slow_runs_fail_the_latency_gate(self) -> None:
        report = self._report([make_result(latency_ms=60_000.0)])
        assert any("p95 latency" in failure for failure in report.threshold_failures())

    def test_refusals_are_excluded_from_the_grounding_average(self) -> None:
        """Otherwise a system that correctly refuses looks worse than one that guesses."""
        report = self._report(
            [make_result(grounding=1.0), make_result(case_id="c2", refused=True, grounding=0.0)]
        )
        assert report.mean_grounding == 1.0

    def test_category_breakdown_is_reported(self) -> None:
        report = self._report(
            [
                make_result(),
                make_result(case_id="c2", category=CaseCategory.ADVERSARIAL, passed=False),
            ]
        )
        categories = report.by_category()
        assert categories["factual"].pass_rate == 1.0
        assert categories["adversarial"].pass_rate == 0.0

    def test_failure_reasons_are_counted(self) -> None:
        report = self._report(
            [
                make_result(passed=False, failures=("unexpected_refusal",)),
                make_result(case_id="c2", passed=False, failures=("unexpected_refusal",)),
            ]
        )
        assert report.failure_reasons()["unexpected_refusal"] == 2

    def test_tokens_are_totalled(self) -> None:
        report = self._report([make_result(), make_result(case_id="c2")])
        assert report.total_tokens == 220

    def test_an_empty_run_does_not_divide_by_zero(self) -> None:
        report = self._report([])
        assert report.pass_rate == 0.0
        assert report.mean_recall == 0.0
        assert report.p95_latency_ms == 0.0

    def test_the_report_serialises_to_a_machine_readable_shape(self) -> None:
        report = self._report([make_result()])
        payload = report.to_dict()
        gate = payload["gate"]
        summary = payload["summary"]
        cases = payload["cases"]
        assert isinstance(gate, dict)
        assert isinstance(summary, dict)
        assert isinstance(cases, list)
        assert gate["passed"] is True
        assert summary["total"] == 1
        assert cases[0]["id"] == "c1"
        assert json.dumps(payload)

    def test_the_summary_renders_the_gate_verdict(self) -> None:
        passing = render_summary(self._report([make_result()]))
        failing = render_summary(self._report([make_result(passed=False, failures=("x",))]))
        assert "QUALITY GATE: PASSED" in passing
        assert "QUALITY GATE: FAILED" in failing

    def test_the_report_is_written_to_disk(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "report.json"
        write_report(self._report([make_result()]), destination)
        assert json.loads(destination.read_text(encoding="utf-8"))["summary"]["total"] == 1


class TestDatasetLoading:
    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        path = tmp_path / "dataset.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_a_dataset_loads_its_header_and_cases(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                json.dumps({"name": "d", "description": "x", "corpus_dir": "docs"}),
                json.dumps({"id": "c1", "query": "q", "category": "factual"}),
            ],
        )
        dataset = EvalDataset.load(path)
        assert dataset.name == "d"
        assert len(dataset.cases) == 1
        assert dataset.resolve_corpus(path).name == "docs"

    def test_comments_and_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                json.dumps({"name": "d"}),
                "",
                "# a comment explaining the next case",
                json.dumps({"id": "c1", "query": "q"}),
            ],
        )
        assert len(EvalDataset.load(path).cases) == 1

    def test_an_empty_dataset_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            EvalDataset.load(self._write(tmp_path, [""]))

    def test_an_unknown_field_in_a_case_is_refused(self, tmp_path: Path) -> None:
        """A typo in an expectation must fail loudly, not be silently ignored."""
        path = self._write(
            tmp_path,
            [json.dumps({"name": "d"}), json.dumps({"id": "c", "query": "q", "must_contian": []})],
        )
        with pytest.raises(Exception, match="must_contian"):
            EvalDataset.load(path)

    def test_the_shipped_regression_dataset_parses(self) -> None:
        path = Path(__file__).resolve().parents[2] / "data" / "regression" / "dataset.jsonl"
        dataset = EvalDataset.load(path)
        assert dataset.cases
        assert {case.category for case in dataset.cases} >= {
            CaseCategory.FACTUAL,
            CaseCategory.ADVERSARIAL,
            CaseCategory.UNANSWERABLE,
        }
        assert dataset.resolve_corpus(path).is_dir()
