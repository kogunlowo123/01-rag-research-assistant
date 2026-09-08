"""Citation extraction and grounding measurement.

An LLM asked to cite its sources will produce citation markers whether or not
the cited passage supports the claim. This module checks, rather than trusting.

For every sentence of the answer:

* the citation markers are extracted and resolved back to the retrieved chunk
  they refer to. A marker with no corresponding passage is a fabricated
  citation and is recorded as such;
* the sentence's content terms are compared against the cited passage's terms.
  A sentence is *supported* when enough of its content is traceable to the
  passage it points at.

Support is measured by term coverage rather than by a second model call. That is
a deliberate trade: it is deterministic, adds no latency or cost, cannot itself
hallucinate, and is testable with a fixed dataset. It is also weaker than an
entailment model — it detects a sentence that talks about something the passage
does not mention, but not a subtle reversal of meaning such as a dropped "not".
:mod:`rag_assistant.evaluation` measures that residual gap on a labelled set,
and ``README.md`` states the limitation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from rag_assistant.domain.models import Citation, GroundingReport
from rag_assistant.generation.prompt import INSUFFICIENT_EVIDENCE
from rag_assistant.indexing.text import stem, terms
from rag_assistant.providers.extractive import split_sentences

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rag_assistant.domain.models import ChunkMatch

_MARKER_RE: Final[re.Pattern[str]] = re.compile(r"\[(\d{1,3})\]")
#: Sentence boundary within an *answer*, as opposed to within source text.
#:
#: A boundary is terminal punctuation, then any citation markers belonging to
#: the sentence just ended, then whitespace and the start of the next sentence
#: (or the end of the answer). Consuming the markers as part of the boundary is
#: what makes "…30 days. [1] Refunds are…" split into two attributable
#: sentences: a plain lookbehind on ".!?" cannot split there at all, because the
#: character before the space is "]", and treating "[" as a sentence start would
#: instead orphan every marker from the claim it supports.
_ANSWER_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    r"[.!?]+(?:\s*\[\d{1,3}\])*(?=\s+[A-Z0-9\"'(]|$)"
)
#: Fraction of a sentence's content terms that must appear in the cited passage
#: for the sentence to count as supported.
SUPPORT_THRESHOLD: Final[float] = 0.5
#: Sentences shorter than this carry too few terms for coverage to mean anything
#: ("Yes.", "It does."), so they inherit the verdict of the preceding sentence.
_MIN_SCORABLE_TERMS: Final[int] = 3
_MAX_QUOTE_CHARS: Final[int] = 400


#: Interrogative scaffolding. These words shape a question without naming its
#: subject, and a corpus is not required to contain them: "How long is the
#: refund window?" is answered by a passage that says "within 30 days" and never
#: uses the words "long" or "window". Counting them as subject matter makes the
#: coverage gate refuse ordinary, answerable questions.
_SCAFFOLDING_WORDS = (
    # Interrogatives and degree adverbs.
    "how long many much often far soon quickly what which who whom whose when "
    "where why whether does do did is are was were can could should would will "
    "shall may might "
    # Request verbs.
    "tell explain describe list give show say "
    # Generic nouns a question uses to refer to an answer it does not yet have.
    "mean means kind sort type way ways window period timeframe duration amount "
    "number rule detail information"
)
_QUESTION_SCAFFOLDING: Final[frozenset[str]] = frozenset(_SCAFFOLDING_WORDS.split())


def evidence_coverage(query: str, passages: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    """Fraction of the query's content terms that appear anywhere in the evidence.

    This is a pre-generation check for the case retrieval cannot signal on its
    own. Retrieval always returns *something* — the nearest passages exist even
    when nothing in the corpus is about the question — and a model handed
    plausible but irrelevant evidence will often answer from it anyway.

    Asking "does the retrieved text even mention what was asked about?" catches
    that cheaply and deterministically. A query about cryptocurrency against a
    corpus of shipping and refund policies scores near zero, and the request is
    refused before a model is called.

    Returns the coverage ratio and the query terms that were absent, so the
    refusal can say what was missing rather than only that something was.
    """
    scaffolding = {stem(word) for word in _QUESTION_SCAFFOLDING}
    query_terms = [term for term in dict.fromkeys(terms(query)) if term not in scaffolding]
    if not query_terms:
        # The query is entirely scaffolding ("what is it?"). There is nothing to
        # check coverage against, so this gate abstains and the later grounding
        # and citation checks decide the outcome.
        return 1.0, ()
    present = {term for passage in passages for term in set(terms(passage))}
    missing = tuple(term for term in query_terms if term not in present)
    return (len(query_terms) - len(missing)) / len(query_terms), missing


def split_answer_sentences(text: str) -> list[str]:
    """Split an answer into sentences, keeping trailing citation markers attached."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences: list[str] = []
    start = 0
    for boundary in _ANSWER_SENTENCE_BOUNDARY.finditer(cleaned):
        if sentence := cleaned[start : boundary.end()].strip():
            sentences.append(sentence)
        start = boundary.end()
    if remainder := cleaned[start:].strip():
        sentences.append(remainder)
    return sentences


def extract_markers(sentence: str) -> list[str]:
    """Return the citation markers appearing in a sentence."""
    return _MARKER_RE.findall(sentence)


def strip_markers(sentence: str) -> str:
    """Remove citation markers so they do not count as content terms."""
    return _MARKER_RE.sub(" ", sentence)


def _coverage(sentence: str, passage: str) -> float:
    sentence_terms = set(terms(strip_markers(sentence)))
    if not sentence_terms:
        return 1.0
    return len(sentence_terms & set(terms(passage))) / len(sentence_terms)


def _best_quote(sentence: str, passage: str) -> str:
    """Select the passage sentence that best supports ``sentence``."""
    sentence_terms = set(terms(strip_markers(sentence)))
    best = ""
    best_score = 0.0
    for candidate in split_sentences(passage):
        candidate_terms = set(terms(candidate))
        if not candidate_terms:
            continue
        score = len(sentence_terms & candidate_terms) / len(sentence_terms or candidate_terms)
        if score > best_score:
            best_score, best = score, candidate
    quote = best or passage
    return quote[:_MAX_QUOTE_CHARS]


def analyse(
    answer_text: str,
    marker_to_match: Mapping[str, ChunkMatch],
    *,
    document_titles: Mapping[str, str] | None = None,
    support_threshold: float = SUPPORT_THRESHOLD,
) -> tuple[GroundingReport, tuple[Citation, ...], tuple[str, ...]]:
    """Measure grounding and build the citation list.

    Returns the report, the resolved citations, and any warnings that should be
    surfaced to the caller (fabricated markers, uncited sentences).
    """
    titles = document_titles or {}
    warnings: list[str] = []

    stripped = answer_text.strip()
    if not stripped or stripped.startswith(INSUFFICIENT_EVIDENCE):
        return (
            GroundingReport(
                total_sentences=0,
                supported_sentences=0,
                unsupported_sentences=(),
                score=0.0,
                citation_coverage=0.0,
            ),
            (),
            (),
        )

    sentences = split_answer_sentences(stripped)
    if not sentences:
        sentences = [stripped]

    citations: dict[str, Citation] = {}
    supported = 0
    cited = 0
    unsupported: list[str] = []
    fabricated: set[str] = set()
    previous_supported = False

    for sentence in sentences:
        markers = extract_markers(sentence)
        resolved = [marker for marker in markers if marker in marker_to_match]
        fabricated.update(marker for marker in markers if marker not in marker_to_match)

        if resolved:
            cited += 1

        content_terms = terms(strip_markers(sentence))
        if len(content_terms) < _MIN_SCORABLE_TERMS:
            # Too short to score on its own; inherit the previous verdict rather
            # than penalising an answer for saying "It does. [1]".
            if previous_supported:
                supported += 1
            continue

        best_coverage = 0.0
        best_marker: str | None = None
        for marker in resolved:
            match = marker_to_match[marker]
            coverage = _coverage(sentence, match.chunk.text)
            if coverage > best_coverage:
                best_coverage, best_marker = coverage, marker

        is_supported = best_marker is not None and best_coverage >= support_threshold
        previous_supported = is_supported

        if is_supported and best_marker is not None:
            supported += 1
            match = marker_to_match[best_marker]
            citations.setdefault(
                best_marker,
                Citation(
                    marker=f"[{best_marker}]",
                    document_id=match.chunk.document_id,
                    document_title=titles.get(match.chunk.document_id)
                    or match.chunk.metadata.get("document_title")
                    or match.chunk.document_id,
                    chunk_id=match.chunk.id,
                    locator=match.chunk.locator,
                    quote=_best_quote(sentence, match.chunk.text),
                    support_score=round(best_coverage, 4),
                ),
            )
        else:
            unsupported.append(sentence[:300])

    total = len(sentences)
    if fabricated:
        warnings.append(
            "the answer cited "
            f"{len(fabricated)} source marker(s) that do not correspond to any retrieved passage"
        )
    if unsupported:
        warnings.append(f"{len(unsupported)} sentence(s) could not be traced to the cited evidence")

    report = GroundingReport(
        total_sentences=total,
        supported_sentences=supported,
        unsupported_sentences=tuple(unsupported),
        score=round(supported / total, 4) if total else 0.0,
        citation_coverage=round(cited / total, 4) if total else 0.0,
    )
    ordered = tuple(citations[key] for key in sorted(citations, key=int))
    return report, ordered, tuple(warnings)


__all__ = [
    "SUPPORT_THRESHOLD",
    "analyse",
    "evidence_coverage",
    "extract_markers",
    "split_answer_sentences",
    "strip_markers",
]
