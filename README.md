# RAG Research Assistant

[![CI](https://github.com/kogunlowo123/rag-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/kogunlowo123/rag-research-assistant/actions/workflows/ci.yml)
[![Security](https://github.com/kogunlowo123/rag-research-assistant/actions/workflows/security.yml/badge.svg)](https://github.com/kogunlowo123/rag-research-assistant/actions/workflows/security.yml)
[![Container](https://github.com/kogunlowo123/rag-research-assistant/actions/workflows/docker.yml/badge.svg)](https://github.com/kogunlowo123/rag-research-assistant/actions/workflows/docker.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A retrieval-augmented generation service that answers questions from an ingested
document corpus, cites the passages that support each claim, measures how well
the answer is actually grounded in those passages, and refuses when it is not.

---

## What is this?

An HTTP service (and CLI) that:

- ingests documents — PDF, HTML, Markdown, plain text, JSON — and indexes them;
- answers natural-language questions using hybrid retrieval over that corpus;
- returns citations resolved to a specific chunk, page and section, with the
  quoted span that supports each claim;
- measures grounding per answer and **withholds** answers that fall below a
  configured threshold;
- treats every retrieved passage as untrusted input and defends against
  prompt injection embedded in documents;
- ships an evaluation harness that fails CI when retrieval or answer quality
  regresses.

## Why does it exist?

Most RAG demos stop at "embed, search, stuff into a prompt". That version fails
in production for reasons that have nothing to do with the model:

| Failure | What this project does about it |
| --- | --- |
| The answer sounds right and cites a passage that does not support it | Grounding is **measured** per sentence against the cited passage, and unsupported answers are refused ([`generation/grounding.py`](src/rag_assistant/generation/grounding.py)) |
| A user asks for an exact identifier and dense retrieval returns near-misses | BM25 runs alongside dense search and the two are fused by RRF ([`indexing/bm25.py`](src/rag_assistant/indexing/bm25.py)) |
| A document in the corpus contains "ignore your instructions and…" | Retrieved text never enters the system role, is scanned by a layered detector, and is neutralised or dropped by policy ([`security/injection.py`](src/rag_assistant/security/injection.py)) |
| Retrieval quietly regresses after a config change | A dataset-driven evaluation gate runs in CI with thresholds ([`evaluation/`](src/rag_assistant/evaluation)) |
| "The answer is wrong" and nobody can say why | `GET /v1/retrieve` returns the full retrieval breakdown with per-stage latency, no model call required |
| The model provider is down | The service degrades to extractive answering over the same evidence and says so in the response |
| Setting it up requires an API key | The default configuration runs with no credentials, no GPU and no external services |

## What problem does it solve?

Answering questions over a private document corpus **with evidence you can
check**, under the assumption that the corpus itself may be hostile.

---

## Architecture

```mermaid
flowchart TB
    subgraph Client
        C[HTTP client / CLI]
    end

    subgraph API["API layer"]
        MW["Middleware<br/>correlation · body limit · security headers"]
        AUTH["API key auth<br/>tenant binding"]
        R1["/v1/documents"]
        R2["/v1/query"]
        R3["/v1/retrieve"]
        R4["/healthz · /readyz"]
    end

    subgraph Ingestion
        V["Validate<br/>size · magic bytes · allowlist"]
        P["Parse<br/>PDF · HTML · MD · JSON"]
        S1["Injection scan"]
        CH["Structure-aware chunking"]
        EM["Embed"]
    end

    subgraph Retrieval
        QR["Query rewrite"]
        D["Dense search"]
        SP["BM25 search"]
        F["RRF fusion"]
        AZ["Authorisation filter"]
        S2["Injection policy<br/>annotate · neutralise · drop"]
        RR["MMR rerank"]
        ST["Neighbour stitching"]
    end

    subgraph Generation
        PR["Prompt build<br/>trust-separated · nonce-fenced"]
        LLM["Chat provider"]
        G["Grounding analysis"]
        POL["Answer policy<br/>refuse if unsupported"]
    end

    subgraph Storage["Storage (SQLite or PostgreSQL)"]
        DOC[(documents)]
        CHK[(chunks + vectors)]
        POST[(postings)]
        AUD[(audit events)]
    end

    subgraph Providers["Provider abstraction"]
        PE["EmbeddingProvider<br/>fastembed · ollama · openai · hashing"]
        PC["ChatProvider<br/>ollama · openai · extractive"]
    end

    C --> MW --> AUTH
    AUTH --> R1 --> V --> P --> S1 --> CH --> EM --> CHK
    EM -.-> PE
    CH --> POST
    V --> DOC
    AUTH --> R2 --> QR --> D & SP --> F --> AZ --> S2 --> RR --> ST --> PR --> LLM --> G --> POL --> C
    LLM -.-> PC
    D -.-> CHK
    SP -.-> POST
    AZ -.-> DOC
    R2 --> AUD
    R1 --> AUD
    AUTH --> R3 --> QR
    R4 --> Storage
```

### Trust boundary

The single most important design decision is that **retrieved document text is
never given authority**:

```mermaid
flowchart LR
    SYS["SYSTEM<br/>application policy"]:::sys
    USR["USER<br/>authenticated caller"]:::usr
    UNT["UNTRUSTED<br/>retrieved passages"]:::unt

    SYS -->|"defines rules"| PROMPT["Prompt"]
    USR -->|"expresses intent"| PROMPT
    UNT -->|"evidence only, nonce-fenced"| PROMPT

    classDef sys fill:#1b5e20,color:#fff,stroke:#0d3d12
    classDef usr fill:#0d47a1,color:#fff,stroke:#062a63
    classDef unt fill:#b71c1c,color:#fff,stroke:#7f1010
```

Every string that reaches a prompt carries a `TrustLevel`. `UNTRUSTED` content
is placed only inside per-request, nonce-delimited evidence blocks in the user
role. There is no code path that puts document text into a system message.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and the decision
records, and [THREAT-MODEL.md](THREAT-MODEL.md) for the adversary model.

---

## Key capabilities

**Ingestion**
- PDF (with page attribution), HTML (script/style/comment stripping), Markdown,
  plain text, JSON (path-flattened)
- Magic-byte validation against the declared media type; archives and
  executables are always refused
- Content-digest deduplication per tenant
- Structure-aware chunking: heading breadcrumbs are prepended to chunk text,
  windows never split a sentence, consecutive windows overlap by whole sentences
- Metadata normalisation — attacker-controlled PDF/HTML metadata is bounded,
  stripped of control characters and truncated
- Document lifecycle states (`pending → parsing → chunking → embedding →
  indexed`/`failed`) visible over the API
- Optional URL ingestion behind an explicit SSRF policy (off by default)

**Retrieval**
- Hybrid: exact dense cosine search + Okapi BM25 over a real inverted index
- Reciprocal rank fusion (rank-based, so the two scales never need normalising)
- Deterministic query rewriting: follow-up reference resolution, identifier
  surface-form expansion, compound-question decomposition — all additive
- MMR reranking for redundancy suppression
- Neighbour stitching to restore context cut by chunk boundaries
- Per-stage latency and candidate counts returned as diagnostics

**Generation**
- Trust-separated prompt with per-request nonce fences
- Citation extraction and **verification** — a marker that resolves to no
  retrieved passage is reported as a fabricated citation
- Per-sentence grounding measurement with a configurable refusal threshold
- Explicit refusal path (`INSUFFICIENT_EVIDENCE`) that is honoured, not
  paraphrased
- Extractive fallback when the model provider is unavailable, flagged in the
  response

**Security**
- API-key authentication with constant-time comparison and per-key tenant
  binding
- Tenant isolation enforced in SQL predicates, not by post-filtering
- Per-document ACLs, deny-by-default, checked before content reaches a prompt
- Layered prompt-injection detection: Unicode normalisation, structural signals,
  intent patterns, noisy-OR aggregation
- Configurable enforcement: `annotate`, `neutralise`, `drop`
- Append-only audit trail containing no document or query text
- Secret redaction in the logging pipeline, not at call sites

**Operations**
- Structured JSON logs with request/trace correlation
- OpenTelemetry traces and metrics
- `/healthz` (never touches a dependency) and `/readyz` (does)
- Configuration invariants enforced at startup

---

## Threat model

Summarised here; the full version is in [THREAT-MODEL.md](THREAT-MODEL.md).

| Threat | Control |
| --- | --- |
| Indirect prompt injection via an ingested document | Trust levels, nonce fences, layered detection, configurable neutralise/drop, risk threshold that overrides the configured action |
| System-prompt extraction | Policy rule 5; extraction patterns detected in retrieved content |
| Data exfiltration through generated links | Markdown-link/image interpolation detection; the service makes no outbound request on behalf of an answer |
| SSRF via URL ingestion | Opt-in, scheme + host allowlist, post-DNS address validation, redirects refused, standard ports only |
| Path traversal via filenames | Filenames are display labels only; content is addressed by digest |
| Malicious file content | Magic-byte validation; archives and executables rejected; parsers bounded |
| Cross-tenant data access | Tenant bound to the credential, enforced in every query predicate |
| Document-level unauthorised access | Deny-by-default ACLs, checked before generation |
| Credential leakage into logs | Redaction processor in the logging pipeline; `SecretStr` in configuration |
| Denial of service via oversized input | Body limit middleware, streaming size enforcement, chunk ceilings, response size caps |
| Ungrounded ("hallucinated") answers | Measured grounding + refusal threshold + citation verification |

**This project does not claim to solve prompt injection.** Detection based on
patterns and structure will miss novel phrasings. The controls that do not
depend on detection — trust separation, the absence of any tool the model could
invoke, and grounding enforcement — are the load-bearing ones.

---

## Technology choices

| Choice | Why | Alternative considered |
| --- | --- | --- |
| Python 3.12 | Ecosystem for the AI/ML surface; `StrEnum`, `match`, PEP 695 generics | — |
| FastAPI + Pydantic v2 | Validation and OpenAPI generated from the same types | Litestar; FastAPI's ecosystem is larger |
| SQLAlchemy 2.0 (async) | One codebase for SQLite (zero-setup) and PostgreSQL (production) | Raw SQL — would need two dialect paths |
| SQLite default | A clean clone runs with no services | Requiring PostgreSQL up front |
| NumPy exact search | No index to build, tune or monitor; exact recall below ~100k chunks/tenant | pgvector/FAISS — the documented path above that threshold |
| BM25 over a real inverted index | Deletions take effect immediately; statistics cannot drift | In-memory `rank_bm25` — loses state on restart |
| `fastembed` (ONNX) | Real transformer embeddings, CPU-only, no torch, no credentials | `sentence-transformers` — pulls ~2.5 GB of torch |
| Ollama default LLM | Real generation with no API key | Requiring OpenAI to see the system work |
| `structlog` | Redaction and correlation as pipeline stages | stdlib `logging` — redaction would be per call site |
| `uv` | Fast, lockfile-based, reproducible | `pip-tools`, Poetry |
| Deterministic query rewriting | No extra latency, no second hallucination surface, reproducible tests | LLM-based rewriting |

Rejected: LangChain and LlamaIndex. The pipeline here is ~10 explicit stages;
wrapping them in a framework would hide the retrieval and trust decisions that
this project exists to make visible.

---

## Repository structure

```
src/rag_assistant/
├── api/                 HTTP layer: app factory, routes, middleware, DI, wire schemas
├── domain/              Shared vocabulary: Document, Chunk, Answer, TrustLevel …
├── ingestion/           Loading, validation, parsing, chunking, indexing pipeline
├── indexing/            Term extraction, BM25 inverted index, dense vector store
├── retrieval/           Query rewriting, hybrid search, RRF fusion, MMR, pipeline
├── generation/          Prompt construction, grounding analysis, answer policy
├── providers/           EmbeddingProvider / ChatProvider protocols + implementations
├── security/            Normalisation, injection detection, authz, source validation
├── storage/             SQLAlchemy schema, engine lifecycle, repositories
├── observability/       Structured logging with redaction, OpenTelemetry
├── evaluation/          Dataset format, metrics, regression runner
├── config.py            Typed settings with startup invariants
├── errors.py            Domain exception hierarchy → HTTP mapping
├── runtime.py           In-process object graph shared by CLI, evaluator and tests
└── cli.py               serve · ingest · query · evaluate

tests/
├── unit/                Pure logic, no I/O
├── integration/         Real SQLite, in-process ASGI client
├── security/            Adversarial cases — a failure here is a security regression
└── e2e/                 Full ingest → query path

data/regression/         Executable evaluation dataset and its corpus
docs/                    Operations, API guide, evaluation guide
```

---

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) 0.10+
- Docker (optional — only for the container path)
- [Ollama](https://ollama.com) (optional — for generative rather than extractive answers)

`make` is optional. `python tasks.py` is the cross-platform task runner and the
`Makefile` delegates to it, so both work and neither can drift from the other.

## Installation

```bash
git clone https://github.com/kogunlowo123/rag-research-assistant.git
cd rag-research-assistant
python tasks.py setup          # uv sync --all-extras --dev
cp .env.example .env
```

## Configuration

All configuration is environment variables, prefixed `RAG_`, nested with `__`.
Full reference: [`.env.example`](.env.example) and
[`docs/configuration.md`](docs/configuration.md).

The settings most likely to matter:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_ENVIRONMENT` | `local` | `local`, `test`, `staging`, `production`. Production enforces extra invariants at startup. |
| `RAG_SECURITY__REQUIRE_API_KEY` | `true` | Cannot be disabled when `RAG_ENVIRONMENT=production`. |
| `RAG_SECURITY__API_KEYS` | *(empty)* | Comma-separated. Use `tenant:secret` to bind a key to a tenant. |
| `RAG_EMBEDDING__BACKEND` | `hashing` | `hashing`, `fastembed`, `ollama`, `openai`. |
| `RAG_CHAT__BACKEND` | `extractive` | `extractive`, `ollama`, `openai`. |
| `RAG_STORAGE__DATABASE_URL` | `sqlite+pysqlite:///./var/rag.db` | SQLite by default; PostgreSQL in production. |
| `RAG_SECURITY__INJECTION_ACTION` | `neutralise` | `annotate`, `neutralise`, `drop`. |
| `RAG_GENERATION__MIN_GROUNDING_SCORE` | `0.35` | Answers below this are refused. |
| `RAG_INGESTION__ALLOW_URL_INGESTION` | `false` | Requires `RAG_INGESTION__URL_ALLOWED_HOSTS`. |
| `RAG_OBSERVABILITY__TRACING_ENABLED` | `false` | Set `RAG_OBSERVABILITY__OTLP_ENDPOINT` to export. |

**Defaults are deliberately the zero-setup pair** (`hashing` + `extractive`):
they work immediately with no downloads and no credentials. `/readyz` returns a
warning naming each one so a deployment running on them is never silent about
it. For real use:

```bash
# Local, still no credentials — recommended
RAG_EMBEDDING__BACKEND=fastembed
RAG_EMBEDDING__MODEL=BAAI/bge-small-en-v1.5
RAG_EMBEDDING__DIMENSIONS=384
RAG_CHAT__BACKEND=ollama
RAG_CHAT__MODEL=llama3.2:3b
```

Changing the embedding backend or model requires re-ingesting: vectors record
which provider produced them and the index refuses to rank across providers.

## Local development

```bash
python tasks.py --list        # every available task

python tasks.py fmt           # ruff format + fix
python tasks.py lint          # ruff check + format --check
python tasks.py typecheck     # mypy --strict
python tasks.py test          # full suite + 80% coverage gate
python tasks.py security      # bandit + pip-audit
python tasks.py build         # wheel + sdist
python tasks.py docker-build  # container image
```

## Running the application

### CLI

```bash
# Index the sample corpus that ships with the repository
uv run rag-assistant ingest data/regression/corpus

# Ask a question
uv run rag-assistant query "How long do customers have to request a refund?"

# See exactly what retrieval did
uv run rag-assistant query "refund window" --diagnostics
```

Sample output:

```
Customers may request a refund within 30 days of purchase. [1]

confidence 0.86  grounding 1.00  provider extractive
  [1] refund-policy.md (Refund Policy > Eligibility #0)
      "Customers may request a refund within 30 days of purchase."
```

### HTTP service

```bash
export RAG_SECURITY__API_KEYS="acme:local-development-key"
uv run rag-assistant serve --port 8000
# or: uv run python -m rag_assistant
```

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/documents \
  -H "X-API-Key: local-development-key" \
  -F "file=@data/regression/corpus/refund-policy.md;type=text/markdown"

curl -sS -X POST http://127.0.0.1:8000/v1/query \
  -H "X-API-Key: local-development-key" \
  -H "content-type: application/json" \
  -d '{"query":"How long do I have to request a refund?","include_diagnostics":true}'
```

## Running tests

```bash
python tasks.py test              # everything, with the coverage gate
python tasks.py test-unit
python tasks.py test-integration
python tasks.py test-security     # adversarial suite
uv run pytest -m e2e
```

Tests are separated by marker so a security regression is distinguishable from a
flaky integration test. The `security` suite contains the adversarial corpus:
direct injection, indirect injection through documents, role confusion,
instruction smuggling, encoded payloads, delimiter attacks, exfiltration
attempts, system-prompt extraction, path traversal, SSRF, and cross-tenant
access.

## Running the evaluation gate

```bash
uv run rag-assistant evaluate data/regression/dataset.jsonl --report var/eval.json
```

This ingests the corpus into a temporary index, runs every case through the real
pipeline, and exits non-zero if any threshold is violated. It runs in CI on every
push. See [`docs/evaluation.md`](docs/evaluation.md).

## Running security scans

```bash
python tasks.py security                                  # bandit + pip-audit
uv run bandit -c pyproject.toml -r src
uv export --no-emit-project --format requirements-txt -o req.txt && uvx pip-audit -r req.txt
docker run --rm -v "$PWD:/src" aquasec/trivy fs --severity HIGH,CRITICAL /src
```

CI additionally runs gitleaks over the working tree *and* the full git history,
CodeQL with `security-extended`, and Trivy against the built image.

## Docker

```bash
docker build -t rag-research-assistant:local .
docker run --rm -p 8000:8000 \
  -e RAG_SECURITY__API_KEYS="acme:local-development-key" \
  -e RAG_STORAGE__DATABASE_URL="sqlite+pysqlite:////app/var/rag.db" \
  rag-research-assistant:local rag_assistant
```

Multi-stage build, non-root (`uid 10001`), no build toolchain in the runtime
layer, healthcheck against `/healthz`, and no secrets baked into the image. CI
asserts the container does not run as root and scans the image with Trivy.

`docker-compose.yml` brings up the service with PostgreSQL and an OTLP collector
for the full observability path.

## Deployment

The service is a stateless container; state lives in PostgreSQL. See
[`docs/operations.md`](docs/operations.md) for the production checklist,
migration guidance, and capacity notes.

Minimum production configuration:

```bash
RAG_ENVIRONMENT=production
RAG_STORAGE__DATABASE_URL=postgresql+psycopg://user:pass@host/db
RAG_SECURITY__REQUIRE_API_KEY=true
RAG_SECURITY__API_KEYS=<from your secret manager, never from a file in the image>
RAG_EMBEDDING__BACKEND=fastembed
RAG_CHAT__BACKEND=ollama    # or openai
RAG_OBSERVABILITY__TRACING_ENABLED=true
RAG_OBSERVABILITY__OTLP_ENDPOINT=http://collector:4318/v1/traces
```

Startup fails if any of these is inconsistent — SQLite in production,
authentication disabled, private-network fetching enabled, or document content
logging turned on.

## CI/CD

| Workflow | What it does |
| --- | --- |
| `ci.yml` | ruff lint + format check, mypy strict, unit/integration/security suites in parallel, 80% coverage gate, wheel build |
| `security.yml` | gitleaks (tree + history), bandit, pip-audit against the resolved lockfile, CodeQL `security-extended`, Trivy filesystem scan. Also runs weekly. |
| `docker.yml` | Buildx image build, Trivy image scan, non-root assertion, container smoke test |

No step is `continue-on-error`. A dependency vulnerability with no upstream fix
is handled by a dated, justified entry in
[`security/audit-exceptions.md`](security/audit-exceptions.md), not by
suppressing the scanner.

## API documentation

OpenAPI is served at `/openapi.json` and Swagger UI at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness. Touches no dependency. |
| `GET` | `/readyz` | Readiness + backend warnings. 503 when not ready. |
| `POST` | `/v1/documents` | Upload and index a document (multipart). |
| `POST` | `/v1/documents/from-url` | Fetch and index by URL (disabled by default). |
| `GET` | `/v1/documents` | List documents for the tenant. |
| `GET` | `/v1/documents/{id}` | Document metadata and ingestion status. |
| `DELETE` | `/v1/documents/{id}` | Remove a document and its index entries. |
| `POST` | `/v1/query` | Ask a question. Returns answer, citations, grounding. |
| `GET` | `/v1/retrieve` | Retrieval diagnostics without generation. |
| `GET` | `/v1/audit` | Recent audit events for the tenant. |

Every failure returns the same shape:

```json
{
  "code": "unsupported_media_type",
  "message": "this media type is not in the ingestion allowlist",
  "request_id": "9f2c…",
  "detail": {"media_type": "application/zip"}
}
```

More detail and worked examples: [`docs/api.md`](docs/api.md).

## Examples

Runnable scripts in [`examples/`](examples):

| Script | Shows |
| --- | --- |
| `quickstart.py` | Ingest a directory and ask a question, entirely in process |
| `injection_demo.py` | Ingest a poisoned document and show the injection being neutralised |
| `hybrid_retrieval.py` | A query where dense retrieval fails and BM25 rescues it |

```bash
uv run python examples/quickstart.py
uv run python examples/injection_demo.py
```

## Performance considerations

- Dense search is an exact matrix product over a cached per-tenant `float32`
  matrix: ~150 MB and tens of milliseconds at 100k chunks × 384 dimensions.
  Above roughly that, move to `pgvector` with HNSW — the seam is
  `VectorStore.search`.
- The vector cache is keyed by `(tenant, embedding provider)` and invalidated by
  a generation counter on write, so a deletion cannot be answered from cache.
- BM25 reads postings per query. On PostgreSQL the `(tenant_id, term)` index
  makes this an index-only scan.
- Ingestion is synchronous. For bulk loads use the CLI, which batches embedding
  calls; for continuous high-volume ingestion, move the pipeline behind a queue.
- Query rewriting adds up to three extra retrievals, run concurrently.

## Observability

- **Logs** — JSON, one object per event, with `request_id`, `trace_id`,
  `span_id` and `tenant_id`. Document and query text are excluded by default.
- **Traces** — spans for `retrieval.pipeline`, `generation.answer` and
  `ingestion.pipeline`, plus per-stage latency in the diagnostics payload.
- **Metrics** — `rag.documents.ingested`, `rag.queries.answered`,
  `rag.security.injection_findings`, `rag.query.duration`,
  `rag.answer.grounding_score`.

The signals worth alerting on are the grounding-score distribution and the
injection-finding rate; both are leading indicators of corpus problems.

## Failure modes

| Failure | Behaviour |
| --- | --- |
| Model provider down or not installed | Extractive fallback; response carries a warning and `finish_reason=degraded_extractive` |
| Provider times out | Retried with jittered backoff, then fallback |
| Embedding dimensions changed without reindex | Dense search returns nothing rather than ranking across embedding spaces; BM25 still answers |
| No document matches | Explicit refusal, HTTP 200, `refused=true` |
| Model produces an ungrounded answer | Withheld; refusal names the grounding shortfall |
| Model fabricates a citation marker | Marker reported as unresolvable; answer refused if no citation resolves |
| Retrieved passage contains an injection | Neutralised, dropped, or annotated per policy; always audited |
| Majority of context is instruction-shaped | Answer carries a corpus-compromise warning |
| Database unreachable | `/readyz` returns 503; `/healthz` still 200 so the process is not restarted |
| Oversized upload | 413 before the body is buffered |
| PDF is a scanned image | 422 stating that OCR is not performed |

## Security considerations

- No secret is ever logged: redaction is a processor in the logging pipeline.
- API keys are compared with `hmac.compare_digest` against SHA-256 digests.
- The tenant is bound to the credential, never taken from a request header.
- 403 and 404 return identical text so document ids cannot be enumerated.
- Upstream error bodies are never forwarded — they can echo user content.
- The container runs as a non-root user with no shell.
- Report vulnerabilities per [SECURITY.md](SECURITY.md). Please do not open a
  public issue.

## Known limitations

These are real and deliberately listed:

1. **Prompt injection is mitigated, not solved.** Novel phrasings will evade
   pattern and structural detection.
2. **Grounding is measured by term coverage, not entailment.** It detects a
   sentence that discusses something the passage does not mention; it does not
   reliably detect a reversal of meaning such as a dropped "not". The evaluation
   dataset measures this residual gap.
3. **Dense search does not scale past a few hundred thousand chunks per
   tenant.** The migration path is documented; it is not implemented.
4. **Ingestion is synchronous.** A large PDF occupies the request for its
   duration.
5. **No OCR.** Scanned PDFs are rejected with an explicit error.
6. **No incremental re-embedding.** Changing the embedding model requires
   re-ingesting the corpus.
7. **`create_all` rather than migrations.** Adequate for a single-writer
   service; Alembic is the documented next step.
8. **The `hashing` embedding backend is lexical only.** It is the zero-setup
   default and `/readyz` says so.
9. **English-centric.** The stemmer, stopword list and sentence splitter assume
   English. Other languages retrieve on the dense path only.
10. **Confidence is a heuristic, not a calibrated probability.**

## Roadmap

Ordered by value, not by effort:

- `pgvector` vector store implementation behind the existing `VectorStore` seam
- Cross-encoder reranking as an optional provider
- Entailment-based grounding as an alternative scorer, measured against the
  current one on the regression dataset
- Asynchronous ingestion with a job queue and status polling
- Alembic migrations
- Streaming answers (SSE) with citation resolution after the stream completes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Also
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
