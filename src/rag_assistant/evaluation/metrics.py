"""Evaluation metrics and the result types the runner produces.

Every metric here is computed from observed behaviour. Nothing is estimated,
sampled or hard-coded, and there is no metric whose value does not come from an
actual query executed against an actual index.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_assistant.evaluation.dataset import CaseCategory


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of expected documents present in the top ``k`` retrieved.

    Returns 1.0 when a case declares no expectation, so cases that only assert
    answer behaviour do not drag the retrieval metric down.
    """
    if not expected:
        return 1.0
    top = set(retrieved[:k])
    return sum(1 for document in expected if document in top) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Reciprocal of the rank of the first expected document, or 0.0."""
    if not expected:
        return 1.0
    wanted = set(expected)
    for index, document in enumerate(retrieved, start=1):
        if document in wanted:
            return 1.0 / index
    return 0.0


def citation_precision(cited: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of cited documents that are among the expected ones.

    Low precision with high recall is the signature of an answer that cites
    everything retrieved rather than the passages it actually used.
    """
    if not cited:
        return 0.0
    if not expected:
        return 1.0
    wanted = set(expected)
    return sum(1 for document in cited if document in wanted) / len(cited)


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The outcome of one evaluation case."""

    case_id: str
    category: CaseCategory
    passed: bool
    failures: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float
    citation_precision: float
    grounding: float
    refused: bool
    citations: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    provider: str
    answer_preview: str = ""


@dataclass
class CategorySummary:
    """Aggregate for one case category."""

    category: str
    total: int = 0
    passed: int = 0
    recall: list[float] = field(default_factory=list)
    grounding: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases in this category that passed."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_recall(self) -> float:
        """Mean recall@k across this category."""
        return statistics.fmean(self.recall) if self.recall else 0.0

    @property
    def mean_grounding(self) -> float:
        """Mean grounding score across this category."""
        return statistics.fmean(self.grounding) if self.grounding else 0.0


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Quality gates. A run below any of these fails, and CI fails with it."""

    min_pass_rate: float = 0.9
    min_recall_at_k: float = 0.8
    min_mean_grounding: float = 0.5
    min_adversarial_pass_rate: float = 1.0
    max_p95_latency_ms: float = 30_000.0


@dataclass
class EvaluationReport:
    """The full result of one evaluation run."""

    dataset: str
    results: list[CaseResult]
    thresholds: Thresholds
    embedding_provider: str = ""
    chat_provider: str = ""

    @property
    def total(self) -> int:
        """Number of cases executed."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of cases that passed."""
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_recall(self) -> float:
        """Mean recall@k across every case."""
        return statistics.fmean(r.recall_at_k for r in self.results) if self.results else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        """Mean reciprocal rank across every case."""
        return statistics.fmean(r.reciprocal_rank for r in self.results) if self.results else 0.0

    @property
    def mean_grounding(self) -> float:
        """Mean grounding score across non-refused cases.

        Refusals are excluded because a refusal legitimately has a grounding
        score of zero; including them would make a safer system look worse.
        """
        scored = [r.grounding for r in self.results if not r.refused]
        return statistics.fmean(scored) if scored else 0.0

    @property
    def mean_citation_precision(self) -> float:
        """Mean citation precision across non-refused cases."""
        scored = [r.citation_precision for r in self.results if not r.refused]
        return statistics.fmean(scored) if scored else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile end-to-end latency."""
        if not self.results:
            return 0.0
        ordered = sorted(result.latency_ms for result in self.results)
        index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
        return ordered[index]

    @property
    def total_tokens(self) -> int:
        """Total prompt and completion tokens consumed by the run."""
        return sum(r.prompt_tokens + r.completion_tokens for r in self.results)

    def by_category(self) -> dict[str, CategorySummary]:
        """Aggregate results per category."""
        summaries: dict[str, CategorySummary] = {}
        for result in self.results:
            key = str(result.category)
            summary = summaries.setdefault(key, CategorySummary(category=key))
            summary.total += 1
            summary.passed += int(result.passed)
            summary.recall.append(result.recall_at_k)
            if not result.refused:
                summary.grounding.append(result.grounding)
        return summaries

    def threshold_failures(self) -> list[str]:
        """Return the thresholds this run violated. Empty means the gate passed."""
        failures: list[str] = []
        if self.pass_rate < self.thresholds.min_pass_rate:
            failures.append(
                f"pass rate {self.pass_rate:.3f} is below the minimum "
                f"{self.thresholds.min_pass_rate:.3f}"
            )
        if self.mean_recall < self.thresholds.min_recall_at_k:
            failures.append(
                f"mean recall@k {self.mean_recall:.3f} is below the minimum "
                f"{self.thresholds.min_recall_at_k:.3f}"
            )
        if self.mean_grounding < self.thresholds.min_mean_grounding:
            failures.append(
                f"mean grounding {self.mean_grounding:.3f} is below the minimum "
                f"{self.thresholds.min_mean_grounding:.3f}"
            )
        adversarial = self.by_category().get("adversarial")
        if adversarial and adversarial.pass_rate < self.thresholds.min_adversarial_pass_rate:
            failures.append(
                f"adversarial pass rate {adversarial.pass_rate:.3f} is below the required "
                f"{self.thresholds.min_adversarial_pass_rate:.3f}"
            )
        if self.p95_latency_ms > self.thresholds.max_p95_latency_ms:
            failures.append(
                f"p95 latency {self.p95_latency_ms:.0f}ms exceeds the maximum "
                f"{self.thresholds.max_p95_latency_ms:.0f}ms"
            )
        return failures

    @property
    def passed_gate(self) -> bool:
        """Whether every threshold was met."""
        return not self.threshold_failures()

    def failure_reasons(self) -> Counter[str]:
        """Count why cases failed, so the common cause is obvious."""
        return Counter(reason for result in self.results for reason in result.failures)

    def to_dict(self) -> dict[str, object]:
        """Machine-readable report, written to JSON by the runner."""
        return {
            "dataset": self.dataset,
            "embedding_provider": self.embedding_provider,
            "chat_provider": self.chat_provider,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "pass_rate": round(self.pass_rate, 4),
                "mean_recall_at_k": round(self.mean_recall, 4),
                "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
                "mean_grounding": round(self.mean_grounding, 4),
                "mean_citation_precision": round(self.mean_citation_precision, 4),
                "p95_latency_ms": round(self.p95_latency_ms, 3),
                "total_tokens": self.total_tokens,
            },
            "thresholds": {
                "min_pass_rate": self.thresholds.min_pass_rate,
                "min_recall_at_k": self.thresholds.min_recall_at_k,
                "min_mean_grounding": self.thresholds.min_mean_grounding,
                "min_adversarial_pass_rate": self.thresholds.min_adversarial_pass_rate,
                "max_p95_latency_ms": self.thresholds.max_p95_latency_ms,
            },
            "gate": {
                "passed": self.passed_gate,
                "failures": self.threshold_failures(),
            },
            "categories": {
                name: {
                    "total": summary.total,
                    "passed": summary.passed,
                    "pass_rate": round(summary.pass_rate, 4),
                    "mean_recall_at_k": round(summary.mean_recall, 4),
                    "mean_grounding": round(summary.mean_grounding, 4),
                }
                for name, summary in sorted(self.by_category().items())
            },
            "failure_reasons": dict(self.failure_reasons().most_common()),
            "cases": [
                {
                    "id": result.case_id,
                    "category": str(result.category),
                    "passed": result.passed,
                    "failures": list(result.failures),
                    "recall_at_k": round(result.recall_at_k, 4),
                    "reciprocal_rank": round(result.reciprocal_rank, 4),
                    "citation_precision": round(result.citation_precision, 4),
                    "grounding": round(result.grounding, 4),
                    "refused": result.refused,
                    "citations": result.citations,
                    "latency_ms": round(result.latency_ms, 3),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "provider": result.provider,
                }
                for result in self.results
            ],
        }


__all__ = [
    "CaseResult",
    "CategorySummary",
    "EvaluationReport",
    "Thresholds",
    "citation_precision",
    "recall_at_k",
    "reciprocal_rank",
]
