"""Evaluation dataset format.

A dataset is a JSONL file: one :class:`EvalCase` per line, plus an optional
``corpus`` directory of documents to ingest first. Keeping it to one line per
case means a case can be added in a pull request without touching a fixture
loader, and a diff shows exactly which expectation changed.

Cases carry expectations at two levels, and both matter:

* **Retrieval** — which documents must appear in the top ``k``. This isolates
  retrieval regressions from generation regressions, which otherwise present
  identically as "the answer got worse".
* **Answer** — substrings that must be present, substrings that must be absent,
  and whether the case must be refused. Refusal cases are first-class: a system
  that never refuses is not safe, it is only confident.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CaseCategory(StrEnum):
    """What a case is testing. Reported separately so one category cannot mask another."""

    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"
    CITATION = "citation"
    LEXICAL = "lexical"


class EvalCase(BaseModel):
    """One evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    category: CaseCategory = CaseCategory.FACTUAL
    #: Source filenames (as ingested) that must appear in the top-k retrieval.
    expected_documents: tuple[str, ...] = ()
    #: Substrings that must appear in the answer, compared case-insensitively.
    must_contain: tuple[str, ...] = ()
    #: Substrings that must not appear. Used by adversarial cases to assert that
    #: an injected instruction was not followed.
    must_not_contain: tuple[str, ...] = ()
    #: Whether the correct behaviour is an explicit refusal.
    must_refuse: bool = False
    #: Whether at least one resolvable citation is required.
    require_citation: bool = True
    #: Minimum acceptable grounding score for this case.
    min_grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""


class EvalDataset(BaseModel):
    """A named collection of cases and the corpus they are evaluated against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    corpus_dir: str = "corpus"
    cases: tuple[EvalCase, ...] = ()

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        """Load a dataset from a JSONL file.

        The first line is the dataset header; every subsequent non-empty line is
        a case. Blank lines and lines beginning with ``#`` are ignored so a
        dataset can be commented.
        """
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            msg = f"dataset {path} is empty"
            raise ValueError(msg)

        header = json.loads(lines[0])
        cases = tuple(EvalCase.model_validate(json.loads(line)) for line in lines[1:])
        return cls(
            name=header.get("name", path.stem),
            description=header.get("description", ""),
            corpus_dir=header.get("corpus_dir", "corpus"),
            cases=cases,
        )

    def resolve_corpus(self, dataset_path: Path) -> Path:
        """Return the corpus directory for this dataset."""
        return (dataset_path.parent / self.corpus_dir).resolve()


__all__ = ["CaseCategory", "EvalCase", "EvalDataset"]
