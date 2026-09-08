"""Typed application configuration.

Configuration is loaded from environment variables (and an optional ``.env``
file) into nested, validated Pydantic models. Nesting is expressed with a
double underscore, so ``RAG_RETRIEVAL__TOP_K=12`` sets ``settings.retrieval.top_k``.

Two rules are enforced here rather than left to convention:

1. Secrets are typed :class:`~pydantic.SecretStr`. They never appear in a
   ``repr``, a log line or an error message.
2. Insecure combinations fail at startup rather than at request time. A
   production environment that disables authentication, or that points the
   ingestion fetcher at private address space, is rejected by
   :meth:`Settings.enforce_environment_invariants`.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from rag_assistant.errors import ConfigurationError


def _split_csv(value: object) -> object:
    """Accept a comma-separated environment variable as a collection.

    Collection-typed settings reach this as a raw string because they are
    annotated ``NoDecode``: without it, pydantic-settings tries to JSON-decode
    the value and ``RAG_SECURITY__API_KEYS=acme:secret`` fails at startup with a
    JSON error that says nothing about the real problem.
    """
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class Environment(StrEnum):
    """Deployment environment. Controls which safety invariants are enforced."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class EmbeddingBackend(StrEnum):
    """Selectable embedding implementations."""

    FASTEMBED = "fastembed"
    OPENAI = "openai"
    OLLAMA = "ollama"
    HASHING = "hashing"


class ChatBackend(StrEnum):
    """Selectable answer-generation implementations."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    EXTRACTIVE = "extractive"


class StorageSettings(BaseModel):
    """Relational metadata store and on-disk vector index location."""

    database_url: str = Field(
        default="sqlite+pysqlite:///./var/rag.db",
        description="SQLAlchemy URL. Use postgresql+psycopg for the 'postgres' extra.",
    )
    vector_dir: Path = Field(
        default=Path("./var/vectors"),
        description="Directory holding the persisted dense vector segments.",
    )
    echo_sql: bool = Field(
        default=False, description="Log every SQL statement. Never in production."
    )
    pool_size: int = Field(default=5, ge=1, le=64)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)


class IngestionSettings(BaseModel):
    """Limits and allowlists applied to every ingested document."""

    max_document_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=512 * 1024 * 1024)
    allowed_media_types: Annotated[frozenset[str], NoDecode] = Field(
        default=frozenset(
            {
                "text/plain",
                "text/markdown",
                "text/html",
                "application/pdf",
                "application/json",
            }
        )
    )
    allow_url_ingestion: bool = Field(
        default=False,
        description="Fetching documents by URL is opt-in because it is an SSRF surface.",
    )
    url_allowed_schemes: Annotated[frozenset[str], NoDecode] = Field(default=frozenset({"https"}))
    url_allowed_hosts: Annotated[frozenset[str], NoDecode] = Field(
        default=frozenset(),
        description=("Exact hostnames the fetcher may contact. An empty set refuses every fetch."),
    )
    url_fetch_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    allow_private_network_fetch: bool = Field(
        default=False,
        description="Permit fetching from private or loopback address space. Test fixtures only.",
    )
    chunk_target_tokens: int = Field(default=320, ge=32, le=2048)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=512)
    max_chunks_per_document: int = Field(default=5000, ge=1)

    _split_media_types = field_validator("allowed_media_types", mode="before")(_split_csv)
    _split_schemes = field_validator("url_allowed_schemes", mode="before")(_split_csv)
    _split_hosts = field_validator("url_allowed_hosts", mode="before")(_split_csv)

    @model_validator(mode="after")
    def _overlap_below_target(self) -> IngestionSettings:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            msg = "chunk_overlap_tokens must be smaller than chunk_target_tokens"
            raise ValueError(msg)
        return self


class EmbeddingSettings(BaseModel):
    """Embedding provider selection and dimensionality."""

    backend: EmbeddingBackend = EmbeddingBackend.HASHING
    model: str = Field(default="BAAI/bge-small-en-v1.5")
    dimensions: int = Field(default=384, ge=8, le=8192)
    batch_size: int = Field(default=32, ge=1, le=512)
    timeout_seconds: float = Field(default=30.0, gt=0)
    cache_dir: Path = Field(default=Path("./var/models"))


class ChatSettings(BaseModel):
    """Answer-generation provider selection."""

    backend: ChatBackend = ChatBackend.EXTRACTIVE
    model: str = Field(default="llama3.2:3b")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=800, ge=32, le=8192)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class ProviderEndpoints(BaseModel):
    """Base URLs and credentials for outbound model providers."""

    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: SecretStr | None = Field(default=None)


class RetrievalSettings(BaseModel):
    """Hybrid retrieval and reranking behaviour."""

    top_k: int = Field(default=8, ge=1, le=100, description="Chunks passed to generation.")
    dense_candidates: int = Field(default=40, ge=1, le=1000)
    sparse_candidates: int = Field(default=40, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000, description="Reciprocal rank fusion constant.")
    enable_query_rewrite: bool = Field(default=True)
    enable_rerank: bool = Field(default=True)
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="1.0 is pure relevance, 0.0 is pure diversity, during MMR reranking.",
    )
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    context_window_neighbours: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Adjacent chunks stitched around each hit to restore local context.",
    )


class GenerationSettings(BaseModel):
    """Grounding and citation policy applied to generated answers."""

    max_context_chars: int = Field(default=24_000, ge=500, le=500_000)
    min_grounding_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Answers below this score are returned with low confidence and a warning.",
    )
    min_evidence_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the query's content terms that must appear somewhere in the "
            "retrieved passages before a model is called. Below this the request is "
            "refused: the corpus does not discuss what was asked. Set to 0 to disable."
        ),
    )
    require_citations: bool = Field(default=True)
    refuse_when_unsupported: bool = Field(
        default=True,
        description="Return an explicit refusal instead of an unsupported answer.",
    )
    max_sentences: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Upper bound on answer length for the extractive backend.",
    )


class SecuritySettings(BaseModel):
    """Authentication and content-trust policy."""

    require_api_key: bool = Field(default=True)
    api_keys: Annotated[tuple[SecretStr, ...], NoDecode] = Field(default=())
    injection_action: Literal["annotate", "neutralise", "drop"] = Field(
        default="neutralise",
        description=(
            "What to do with a retrieved chunk that looks like an injection attempt: record the "
            "finding only, strip the instruction-like spans, or exclude the chunk entirely."
        ),
    )
    injection_block_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Aggregate risk at or above which a chunk is dropped regardless of "
            "injection_action. Only applied when at least two distinct rules fired, so a "
            "document that quotes an attack is degraded rather than discarded."
        ),
    )
    max_request_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    max_query_chars: int = Field(default=4000, ge=16, le=100_000)

    _split_api_keys = field_validator("api_keys", mode="before")(_split_csv)


class ObservabilitySettings(BaseModel):
    """Logging, tracing and metrics configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    service_name: str = "rag-research-assistant"
    tracing_enabled: bool = Field(default=False)
    otlp_endpoint: str | None = Field(default=None)
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    log_document_content: bool = Field(
        default=False,
        description="Include document and query text in logs. Off by default; it is user data.",
    )


class Settings(BaseSettings):
    """Root settings object. One instance per process, cached by :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    providers: ProviderEndpoints = Field(default_factory=ProviderEndpoints)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def is_production(self) -> bool:
        """Whether the process believes it is serving production traffic.

        Compared by value rather than identity: ``Environment`` is a
        :class:`~enum.StrEnum`, so a settings object built by ``model_copy`` — or
        by any other path that bypasses field validation — can legitimately hold
        the plain string. An identity check would silently report "not
        production" there and skip every production invariant, which is exactly
        the situation these invariants exist to prevent.
        """
        return self.environment == Environment.PRODUCTION

    def enforce_environment_invariants(self) -> None:
        """Fail fast on configurations that are unsafe for the declared environment.

        Called during application startup. Raising here is deliberate: a
        misconfigured deployment should refuse to serve rather than silently
        drop a control.
        """
        problems: list[str] = []

        if self.security.require_api_key and not self.security.api_keys:
            problems.append(
                "security.require_api_key is set but security.api_keys is empty; "
                "no caller could ever authenticate"
            )

        if self.is_production:
            if not self.security.require_api_key:
                problems.append("authentication cannot be disabled in production")
            if self.ingestion.allow_private_network_fetch:
                problems.append(
                    "ingestion.allow_private_network_fetch is a test-only escape hatch "
                    "and is not permitted in production"
                )
            if self.observability.log_document_content:
                problems.append("observability.log_document_content would log user data")
            if self.storage.echo_sql:
                problems.append("storage.echo_sql would log query parameters in production")
            if self.storage.database_url.startswith("sqlite"):
                problems.append(
                    "storage.database_url points at SQLite; production requires a "
                    "durable multi-writer database such as PostgreSQL"
                )

        if self.ingestion.allow_url_ingestion and not self.ingestion.url_allowed_hosts:
            problems.append(
                "ingestion.allow_url_ingestion is enabled but ingestion.url_allowed_hosts "
                "is empty; every fetch would be refused"
            )

        if problems:
            raise ConfigurationError(
                "invalid configuration for the declared environment",
                detail={"problems": problems, "environment": str(self.environment)},
            )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings. Used by tests that manipulate the environment."""
    get_settings.cache_clear()


__all__ = [
    "ChatBackend",
    "ChatSettings",
    "EmbeddingBackend",
    "EmbeddingSettings",
    "Environment",
    "GenerationSettings",
    "IngestionSettings",
    "ObservabilitySettings",
    "ProviderEndpoints",
    "RetrievalSettings",
    "SecuritySettings",
    "Settings",
    "StorageSettings",
    "get_settings",
    "reset_settings_cache",
]
