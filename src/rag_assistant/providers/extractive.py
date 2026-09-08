"""Extractive answer provider: answers without a language model.

This is a genuine extractive question-answering implementation, not a stand-in.
It scores every sentence in the retrieved evidence against the query using
IDF-weighted term overlap computed over the evidence set itself, applies
redundancy suppression so the answer does not repeat one idea three times, and
emits the selected sentences in their original order with the citation marker
of the passage they came from.

Why ship it
-----------
1. **It is the zero-dependency default.** A new clone answers real questions
   over real documents with no model download, no API key and no GPU.
2. **It is the degraded mode.** When the configured language model is
   unavailable, an extractive answer built only from retrieved text is more
   useful than an error, and it cannot hallucinate: every sentence it returns
   appears verbatim in a source document.
3. **It is the grounding control.** Because its output is verbatim source text,
   it gives the grounding scorer a known-good upper bound to compare a
   generative answer against.

Its limits are equally real: it cannot synthesise across passages, cannot
rephrase, and answers questions whose answer is not stated as a contiguous
sentence poorly. The provider reports ``finish_reason="extractive"`` so those
answers are distinguishable in telemetry.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Final

from rag_assistant.indexing.text import terms as _terms
from rag_assistant.providers.base import GenerationRequest, GenerationResponse

PROVIDER_NAME = "extractive"

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_MIN_SENTENCE_CHARS: Final[int] = 25
_MAX_SENTENCE_CHARS: Final[int] = 600
_REDUNDANCY_THRESHOLD: Final[float] = 0.7
#: A sentence is kept only if it scores at least this fraction of the best
#: sentence's score. Keeps an answer focused on what was asked.
_RELATIVE_CUTOFF: Final[float] = 0.45
#: Weight of the passage-heading prior relative to a sentence's own score.
#: Deliberately below 1.0: the heading is a hint about where the answer lives,
#: not evidence in its own right.
_HEADER_PRIOR_WEIGHT: Final[float] = 0.6


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    A regex splitter is used rather than a statistical model because the input
    is already a retrieved passage of bounded length, and because a dependency
    on a sentence tokeniser with downloadable data would defeat the purpose of
    a zero-setup default.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]


class ExtractiveChatProvider:
    """Selects and returns the best-supported sentences from retrieved evidence."""

    def __init__(self, *, max_sentences: int = 4) -> None:
        """Configure how many sentences the answer may contain."""
        self._max_sentences = max_sentences

    @property
    def name(self) -> str:
        """Provider identifier recorded on every answer."""
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        """Model identifier recorded on every answer."""
        return "extractive-idf-v1"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Produce an answer composed entirely of verbatim source sentences."""
        started = time.perf_counter()
        query_terms = Counter(_terms(request.question()))

        candidates: list[tuple[int, str, str]] = []
        contexts: dict[str, str] = {}
        for index, segment in enumerate(request.untrusted_segments(), start=1):
            marker = segment.label or str(index)
            contexts[marker] = segment.context
            candidates.extend(
                (index, marker, sentence)
                for sentence in split_sentences(segment.content)
                if _MIN_SENTENCE_CHARS <= len(sentence) <= _MAX_SENTENCE_CHARS
            )

        if not query_terms or not candidates:
            return GenerationResponse(
                text="",
                model=self.model,
                provider=PROVIDER_NAME,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                finish_reason="no_evidence",
            )

        # Headings are part of the IDF corpus: a term that appears only in a
        # section title ("keyword", "insufficient") would otherwise have an IDF
        # of zero and contribute nothing to the prior that exists to surface it.
        idf = self._inverse_document_frequency(
            [sentence for _, _, sentence in candidates] + list(contexts.values())
        )
        # A passage whose provenance line matches the query is more likely to
        # contain the answer, even when its sentences do not repeat the heading's
        # words. "Why keyword filtering is insufficient" is a section title; the
        # paragraph beneath it says "a filter that matches the literal phrase…"
        # and never repeats "keyword" or "insufficient". Without this prior the
        # scorer cannot see the heading at all.
        priors = {
            marker: _HEADER_PRIOR_WEIGHT * self._score(context, query_terms, idf)
            for marker, context in contexts.items()
        }
        # The prior reorders sentences that already have some overlap with the
        # query; it never creates relevance from nothing. Without this guard a
        # document merely *titled* "policy.md" would answer a question about a
        # policy the document does not mention.
        scored = [
            (own + priors.get(marker, 0.0), position, marker, sentence)
            for position, (_, marker, sentence) in enumerate(candidates)
            if (own := self._score(sentence, query_terms, idf)) > 0.0
        ]
        if not scored:
            return GenerationResponse(
                text="",
                model=self.model,
                provider=PROVIDER_NAME,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                finish_reason="no_supporting_sentence",
            )
        scored.sort(key=lambda item: (-item[0], item[1]))

        # Keep only sentences close to the best one. Without this the provider
        # returns every sentence that shares any term with the query, which on a
        # stitched multi-chunk passage means most of a document section. A
        # relative cutoff adapts to the query: a question whose answer is stated
        # once yields one sentence, while a genuinely multi-part answer keeps
        # all of its parts.
        best_score = scored[0][0]
        floor = best_score * _RELATIVE_CUTOFF

        selected: list[tuple[int, str, str]] = []
        for score, position, marker, sentence in scored:
            if score <= 0.0 or score < floor or len(selected) >= self._max_sentences:
                break
            if self._is_redundant(sentence, [existing for _, _, existing in selected]):
                continue
            selected.append((position, marker, sentence))

        if not selected:
            return GenerationResponse(
                text="",
                model=self.model,
                provider=PROVIDER_NAME,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                finish_reason="no_supporting_sentence",
            )

        selected.sort(key=lambda item: item[0])
        text = " ".join(f"{sentence} [{marker}]" for _, marker, sentence in selected)

        return GenerationResponse(
            text=text,
            model=self.model,
            provider=PROVIDER_NAME,
            prompt_tokens=sum(len(_terms(segment.content)) for segment in request.segments),
            completion_tokens=len(_terms(text)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            finish_reason="extractive",
        )

    @staticmethod
    def _inverse_document_frequency(sentences: list[str]) -> dict[str, float]:
        """Compute IDF over the retrieved evidence set.

        Using the evidence itself as the corpus means a term that is common in
        *these* passages — typically the topic word the query already matched —
        contributes less than a term that discriminates between them.
        """
        total = len(sentences)
        document_frequency: Counter[str] = Counter()
        for sentence in sentences:
            document_frequency.update(set(_terms(sentence)))
        return {
            term: math.log((total + 1) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @staticmethod
    def _score(sentence: str, query_terms: Counter[str], idf: dict[str, float]) -> float:
        sentence_terms = Counter(_terms(sentence))
        if not sentence_terms:
            return 0.0
        overlap = sum(
            idf.get(term, 0.0) * min(count, sentence_terms[term])
            for term, count in query_terms.items()
            if term in sentence_terms
        )
        # Length normalisation stops a long paragraph from winning purely by
        # containing more words than a precise one-line answer.
        return overlap / math.sqrt(len(sentence_terms) + 1)

    @staticmethod
    def _is_redundant(candidate: str, chosen: list[str]) -> bool:
        candidate_terms = set(_terms(candidate))
        if not candidate_terms:
            return True
        for existing in chosen:
            existing_terms = set(_terms(existing))
            if not existing_terms:
                continue
            jaccard = len(candidate_terms & existing_terms) / len(candidate_terms | existing_terms)
            if jaccard >= _REDUNDANCY_THRESHOLD:
                return True
        return False

    async def health(self) -> bool:
        """Report healthy always: the provider is in-process and has no dependencies."""
        return True

    async def aclose(self) -> None:
        """No resources are held; present to satisfy the provider protocol."""
        return


__all__ = ["PROVIDER_NAME", "ExtractiveChatProvider", "split_sentences"]
