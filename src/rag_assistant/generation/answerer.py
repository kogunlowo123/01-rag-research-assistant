"""Answer generation: the policy layer around the model call.

The model produces text. Everything that decides whether that text is returned
to the caller lives here, outside the model:

* **No evidence, no answer.** If retrieval returned nothing, generation is
  skipped entirely — there is no prompt to build and nothing to hallucinate
  from.
* **Refusal is honoured, not paraphrased.** A model that emits the
  ``INSUFFICIENT_EVIDENCE`` token gets its refusal passed through verbatim.
* **Grounding gates the answer.** When ``refuse_when_unsupported`` is set, an
  answer whose measured grounding falls below the threshold is replaced by an
  explicit refusal that names the shortfall. An unsupported answer is worse than
  no answer, because it looks like a supported one.
* **Citations are required and verified.** An answer with no resolvable
  citation is refused when ``require_citations`` is set.
* **Every decision is recorded.** The returned
  :class:`~rag_assistant.domain.models.Answer` carries the grounding report, the
  warnings, the provider, the token counts and the retrieval diagnostics, so a
  reviewer can reconstruct why an answer looks the way it does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_assistant.config import Settings
from rag_assistant.domain.models import Answer, GroundingReport, RetrievalDiagnostics
from rag_assistant.generation import grounding
from rag_assistant.generation.prompt import INSUFFICIENT_EVIDENCE, build_prompt
from rag_assistant.observability.logging import get_logger
from rag_assistant.observability.tracing import (
    get_tracer,
    grounding_score,
    queries_answered,
    query_latency,
)
from rag_assistant.security.injection import context_dilution_score

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rag_assistant.domain.models import RetrievalResult
    from rag_assistant.providers.base import ChatProvider

logger = get_logger(__name__)
tracer = get_tracer()

#: Fraction of retrieved passages carrying injection findings above which the
#: answer is flagged as resting on a possibly compromised corpus.
_CONTEXT_DILUTION_WARNING = 0.5

_EMPTY_GROUNDING = GroundingReport(
    total_sentences=0,
    supported_sentences=0,
    unsupported_sentences=(),
    score=0.0,
    citation_coverage=0.0,
)


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """One answer request: a query and the evidence retrieved for it."""

    query: str
    retrieval: RetrievalResult
    document_titles: Mapping[str, str]
    include_diagnostics: bool = False


class Answerer:
    """Turns retrieved evidence into a grounded, cited answer or an explicit refusal."""

    def __init__(self, *, settings: Settings, provider: ChatProvider) -> None:
        """Wire the generation policy to a chat provider."""
        self._settings = settings
        self._provider = provider

    async def answer(self, request: AnswerRequest) -> Answer:  # noqa: PLR0911
        """Generate and validate an answer.

        The several early returns are the policy: each one is a distinct,
        independently auditable reason to withhold an answer. Collapsing them
        into nested conditionals would hide which gate fired.
        """
        config = self._settings.generation
        started = time.perf_counter()
        matches = list(request.retrieval.matches)
        diagnostics = request.retrieval.diagnostics if request.include_diagnostics else None

        if not matches:
            queries_answered.add(1, {"outcome": "no_evidence"})
            return self._refusal(
                request.query,
                "No document in the accessible corpus matched this question.",
                diagnostics=diagnostics,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        warnings: list[str] = []
        dilution = context_dilution_score(
            len(matches), sum(1 for match in matches if match.injection_findings)
        )
        if dilution >= _CONTEXT_DILUTION_WARNING:
            # Most of the retrieved context is instruction-shaped. That is not a
            # document-quality problem, it is a corpus under attack.
            warnings.append(
                "a majority of the retrieved passages contained instruction-like text; "
                "the answer may be based on a compromised corpus"
            )
            logger.warning(
                "security.context_dilution",
                dilution=dilution,
                passages=len(matches),
                flagged=sum(1 for match in matches if match.injection_findings),
            )

        coverage, missing = grounding.evidence_coverage(
            request.query, [match.chunk.indexing_text for match in matches]
        )
        if coverage < config.min_evidence_coverage:
            # The corpus does not discuss what was asked. Refusing here rather
            # than after generation saves a model call and, more importantly,
            # removes the opportunity for a model to answer from evidence that
            # is merely nearby.
            queries_answered.add(1, {"outcome": "insufficient_coverage"})
            logger.info(
                "generation.refused_low_coverage",
                coverage=round(coverage, 3),
                threshold=config.min_evidence_coverage,
                missing_terms=len(missing),
            )
            return self._refusal(
                request.query,
                "The retrieved passages do not discuss "
                + ", ".join(sorted(missing)[:5])
                + ". No document in the accessible corpus appears to cover this question.",
                diagnostics=diagnostics,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                warnings=tuple(warnings),
            )

        with tracer.start_as_current_span("generation.answer") as span:
            built = build_prompt(request.query, matches, max_context_chars=config.max_context_chars)
            if built.truncated_passages:
                warnings.append(
                    f"{built.truncated_passages} passage(s) were truncated "
                    "to fit the context budget"
                )

            response = await self._provider.generate(built.request)
            span.set_attribute("generation.provider", response.provider)
            span.set_attribute("generation.finish_reason", response.finish_reason)

        if response.metadata.get("degraded") == "true":
            warnings.append(
                "the configured language model was unavailable; this answer was produced by "
                "the extractive fallback and contains only verbatim source sentences"
            )

        text = response.text.strip()
        if not text or response.finish_reason in {"no_evidence", "no_supporting_sentence"}:
            queries_answered.add(1, {"outcome": "no_supporting_sentence"})
            return self._refusal(
                request.query,
                "The retrieved passages do not contain a statement that answers this question.",
                diagnostics=diagnostics,
                response_provider=response.provider,
                response_model=response.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                warnings=tuple(warnings),
            )

        if text.startswith(INSUFFICIENT_EVIDENCE):
            detail = text[len(INSUFFICIENT_EVIDENCE) :].strip(" .:\n") or (
                "The evidence does not cover this question."
            )
            queries_answered.add(1, {"outcome": "model_refused"})
            return self._refusal(
                request.query,
                detail,
                diagnostics=diagnostics,
                response_provider=response.provider,
                response_model=response.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                warnings=tuple(warnings),
            )

        report, citations, grounding_warnings = grounding.analyse(
            text, built.marker_to_match, document_titles=request.document_titles
        )
        warnings.extend(grounding_warnings)

        if config.require_citations and not citations:
            queries_answered.add(1, {"outcome": "uncited"})
            return self._refusal(
                request.query,
                "An answer was produced but none of its claims could be traced to a "
                "retrieved passage, so it was withheld.",
                diagnostics=diagnostics,
                response_provider=response.provider,
                response_model=response.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                warnings=tuple(warnings),
            )

        if config.refuse_when_unsupported and report.score < config.min_grounding_score:
            queries_answered.add(1, {"outcome": "ungrounded"})
            logger.warning(
                "generation.refused_ungrounded",
                grounding=report.score,
                threshold=config.min_grounding_score,
                provider=response.provider,
            )
            return self._refusal(
                request.query,
                "The generated answer could not be sufficiently traced to the retrieved "
                f"evidence (grounding {report.score:.2f}, minimum "
                f"{config.min_grounding_score:.2f}), so it was withheld.",
                diagnostics=diagnostics,
                response_provider=response.provider,
                response_model=response.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                warnings=tuple(warnings),
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        queries_answered.add(1, {"outcome": "answered", "grounded": str(report.is_grounded)})
        grounding_score.record(report.score, {"provider": response.provider})
        query_latency.record(latency_ms, {"provider": response.provider})

        return Answer(
            query=request.query,
            text=text,
            citations=citations,
            grounding=report,
            refused=False,
            warnings=tuple(warnings),
            model=response.model,
            provider=response.provider,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=round(latency_ms, 3),
            diagnostics=diagnostics,
        )

    def _refusal(
        self,
        query: str,
        reason: str,
        *,
        diagnostics: RetrievalDiagnostics | None = None,
        response_provider: str = "",
        response_model: str = "",
        latency_ms: float = 0.0,
        warnings: tuple[str, ...] = (),
    ) -> Answer:
        """Build an explicit refusal.

        Refusals carry the same envelope as answers — provider, latency,
        diagnostics — so a client does not need a second code path, and so
        refusal rate is measurable from the same telemetry.
        """
        query_latency.record(latency_ms, {"provider": response_provider or "none"})
        return Answer(
            query=query,
            text="",
            citations=(),
            grounding=_EMPTY_GROUNDING,
            refused=True,
            refusal_reason=reason,
            warnings=warnings,
            model=response_model,
            provider=response_provider,
            latency_ms=round(latency_ms, 3),
            diagnostics=diagnostics,
        )


__all__ = ["AnswerRequest", "Answerer"]
