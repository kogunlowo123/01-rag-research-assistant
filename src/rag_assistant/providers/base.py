"""Provider abstractions for embeddings and answer generation.

The application depends on these two protocols and nothing else. Swapping
Ollama for OpenAI, or an ONNX embedding model for a hosted one, is a
configuration change; no pipeline code moves.

Both protocols are asynchronous because every real implementation performs
network or thread-pool I/O, and making the synchronous case async is far
cheaper than retrofitting async later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag_assistant.domain.models import TrustLevel


@dataclass(frozen=True, slots=True)
class PromptSegment:
    """One block of text destined for a prompt, tagged with its trust level.

    Providers receive segments rather than a rendered string so that an
    implementation which supports a structured message API can map trust levels
    onto real roles, while a text-completion implementation can fall back to
    explicit fencing. Neither can accidentally lose the distinction.

    ``content`` is always the *bare* text. Delimiters are added by
    :meth:`GenerationRequest.render_untrusted` at the moment the prompt is
    serialised. Keeping them out of ``content`` matters: a provider that needs
    to analyse the passage — the extractive backend scores its sentences —
    must see the passage, not the packaging around it.
    """

    trust: TrustLevel
    content: str
    label: str = ""
    #: Provenance line rendered above an untrusted passage.
    header: str = ""
    #: Where the passage sits in its document — title and heading breadcrumb.
    #: Separate from ``header`` because ``header`` is prose meant for a model,
    #: while this is the clean signal a provider can score a query against.
    context: str = ""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A generation call described independently of any provider's wire format."""

    segments: tuple[PromptSegment, ...]
    max_output_tokens: int = 800
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    #: Per-request delimiters around untrusted passages. Chosen by the prompt
    #: builder so they cannot be predicted by content written beforehand.
    fence_open: str = "<<<EVIDENCE"
    fence_close: str = "END EVIDENCE>>>"

    def render_untrusted(self) -> str:
        """Serialise the evidence blocks exactly once, with their fences.

        Every provider calls this rather than fencing on its own, so the
        delimiter format is defined in one place and cannot be applied twice.
        """
        blocks = [
            f"{self.fence_open}\n{segment.header}\n---\n{segment.content}\n{self.fence_close}"
            if segment.header
            else f"{self.fence_open}\n{segment.content}\n{self.fence_close}"
            for segment in self.untrusted_segments()
        ]
        return "\n\n".join(blocks)

    def system_text(self) -> str:
        """Concatenate the system-authored segments."""
        return "\n\n".join(s.content for s in self.segments if s.trust is TrustLevel.SYSTEM)

    def user_text(self) -> str:
        """Concatenate the caller-authored segments."""
        return "\n\n".join(s.content for s in self.segments if s.trust is TrustLevel.USER)

    def untrusted_segments(self) -> tuple[PromptSegment, ...]:
        """Return the evidence segments, which carry no authority."""
        return tuple(s for s in self.segments if s.trust is TrustLevel.UNTRUSTED)

    def question(self) -> str:
        """Return the bare user question, without the surrounding task instruction.

        Providers that score text against the query — the extractive backend —
        need the question alone; feeding them the instruction's vocabulary would
        bias sentence selection toward words the user never typed.
        """
        for segment in self.segments:
            if segment.trust is TrustLevel.USER and segment.label == "question":
                return segment.content.removeprefix("Question:").strip()
        return self.user_text()


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """A provider's answer plus the accounting needed for cost and latency views."""

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors."""

    @property
    def name(self) -> str:
        """Stable identifier recorded alongside every stored vector.

        Vectors produced by different models are not comparable, so the index
        records which provider and model produced each segment and refuses to
        mix them.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Vector width produced by this provider."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Kept separate from :meth:`embed_documents` because asymmetric models
        prepend different instruction prefixes to queries and passages, and
        using the passage prefix for a query measurably degrades recall.
        """
        ...

    async def aclose(self) -> None:
        """Release any network or model resources."""
        ...


@runtime_checkable
class ChatProvider(Protocol):
    """Produces an answer from a trust-tagged prompt."""

    @property
    def name(self) -> str:
        """Stable provider identifier, recorded on every answer."""
        ...

    @property
    def model(self) -> str:
        """Model identifier, recorded on every answer."""
        ...

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate an answer."""
        ...

    async def health(self) -> bool:
        """Whether the provider is currently reachable and usable."""
        ...

    async def aclose(self) -> None:
        """Release any network resources."""
        ...


__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "GenerationRequest",
    "GenerationResponse",
    "PromptSegment",
]
