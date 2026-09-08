"""Evaluation: datasets, metrics and the regression runner."""

from rag_assistant.evaluation.dataset import CaseCategory, EvalCase, EvalDataset
from rag_assistant.evaluation.metrics import (
    CaseResult,
    EvaluationReport,
    Thresholds,
    citation_precision,
    recall_at_k,
    reciprocal_rank,
)
from rag_assistant.evaluation.runner import render_summary, run_dataset, write_report

__all__ = [
    "CaseCategory",
    "CaseResult",
    "EvalCase",
    "EvalDataset",
    "EvaluationReport",
    "Thresholds",
    "citation_precision",
    "recall_at_k",
    "reciprocal_rank",
    "render_summary",
    "run_dataset",
    "write_report",
]
