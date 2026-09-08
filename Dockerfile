# syntax=docker/dockerfile:1.9
# Multi-stage build. Dependencies are resolved from the lockfile in a builder
# stage; only the virtual environment and the application source reach the
# runtime image, which carries no compiler, no package manager and no shell for
# the application user.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.10.10

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /build

# Dependencies first, so the layer is reused whenever only source changes.
# --no-install-project keeps the project itself out of this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --no-editable

COPY src ./src
# --no-editable matters: uv installs a workspace project in editable mode by
# default, which leaves a .pth file pointing at the build directory. That path
# does not exist in the runtime stage, and the import would fail at startup.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    RAG_STORAGE__DATABASE_URL="sqlite+aiosqlite:////app/var/rag.db" \
    RAG_EMBEDDING__CACHE_DIR="/app/var/models"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --system --gid 10001 app; \
    useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app; \
    install -d -o app -g app /app /app/var

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /build/src /app/src

WORKDIR /app
USER app

# Writable state lives here. Mount a volume for durability, or point
# RAG_STORAGE__DATABASE_URL at PostgreSQL, which is required in production.
VOLUME ["/app/var"]

EXPOSE 8000

# Liveness only: the readiness probe (/readyz) is the one that checks
# dependencies, and a restart is the wrong response to a slow database.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["python", "-m"]
CMD ["rag_assistant"]

LABEL org.opencontainers.image.title="rag-research-assistant" \
      org.opencontainers.image.description="Grounded retrieval-augmented generation service with citation verification and prompt-injection defence" \
      org.opencontainers.image.source="https://github.com/kogunlowo123/01-rag-research-assistant" \
      org.opencontainers.image.licenses="MIT"
