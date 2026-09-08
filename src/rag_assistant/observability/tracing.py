"""OpenTelemetry tracing and metrics setup.

Tracing is opt-in. When it is disabled the module still returns a working
tracer and metric handles, because the no-op implementations shipped with the
OpenTelemetry API cost almost nothing and keep call sites free of conditionals.

RAG pipelines are hard to debug from logs alone: a slow answer might be slow in
embedding, in fusion, in reranking or at the model. Each of those is a span, so
the breakdown is visible without adding timing code to every function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

if TYPE_CHECKING:
    from rag_assistant.config import Settings

INSTRUMENTATION_NAME: Final[str] = "rag_assistant"

_provider: TracerProvider | None = None


def configure_tracing(settings: Settings) -> None:
    """Install a tracer provider when tracing is enabled.

    An OTLP endpoint is optional: without one, spans are still created and
    sampled, which keeps trace ids flowing into log records for correlation
    even in environments with no collector.
    """
    global _provider  # noqa: PLW0603 - one provider per process is the OTel contract

    if not settings.observability.tracing_enabled:
        return
    if _provider is not None:
        return

    resource = Resource.create(
        {
            "service.name": settings.observability.service_name,
            "service.version": _version(),
            "deployment.environment": str(settings.environment),
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.observability.trace_sample_ratio)),
    )

    if settings.observability.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint))
        )

    trace.set_tracer_provider(provider)
    _provider = provider


def shutdown_tracing() -> None:
    """Flush and tear down the tracer provider during application shutdown."""
    global _provider  # noqa: PLW0603
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def get_tracer(name: str = INSTRUMENTATION_NAME) -> trace.Tracer:
    """Return a tracer. Yields a no-op tracer when tracing is disabled."""
    return trace.get_tracer(name)


def _version() -> str:
    from rag_assistant import __version__

    return __version__


_meter = metrics.get_meter(INSTRUMENTATION_NAME)

#: Documents accepted for ingestion, labelled by outcome.
documents_ingested = _meter.create_counter(
    "rag.documents.ingested",
    unit="1",
    description="Documents that completed ingestion, labelled by status.",
)

#: Queries answered, labelled by whether the answer was refused.
queries_answered = _meter.create_counter(
    "rag.queries.answered",
    unit="1",
    description="Queries answered, labelled by refusal and grounding outcome.",
)

#: Prompt-injection findings raised against retrieved content.
injection_findings = _meter.create_counter(
    "rag.security.injection_findings",
    unit="1",
    description="Prompt-injection findings in retrieved content, labelled by rule and action.",
)

#: End-to-end query latency.
query_latency = _meter.create_histogram(
    "rag.query.duration",
    unit="ms",
    description="End-to-end query latency in milliseconds.",
)

#: Grounding score distribution, the primary answer-quality signal.
grounding_score = _meter.create_histogram(
    "rag.answer.grounding_score",
    unit="1",
    description="Distribution of grounding scores across generated answers.",
)

__all__ = [
    "INSTRUMENTATION_NAME",
    "configure_tracing",
    "documents_ingested",
    "get_tracer",
    "grounding_score",
    "injection_findings",
    "queries_answered",
    "query_latency",
    "shutdown_tracing",
]
