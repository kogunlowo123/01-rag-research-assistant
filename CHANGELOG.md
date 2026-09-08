# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation site generated from the repository Markdown and published to
  GitHub Pages, with a build that fails on a broken internal link
- `site` task in `tasks.py` and `Makefile` to build the site locally

### Changed
- Repository renamed to `rag-research-assistant`; badge, container-label and
  project-metadata URLs updated to the canonical name

## [0.1.0] - 2026-09-07

First public release. The service is feature-complete for its stated scope:
ingest documents, answer questions from them with verified citations, and refuse
when the answer cannot be traced to evidence.

### Added

**Ingestion**
- Parsers for PDF (with page attribution), HTML, Markdown, plain text and JSON
- Magic-byte validation; archives and executables refused regardless of the allowlist
- Content-digest deduplication, scoped per tenant
- Structure-aware chunking: heading breadcrumbs carried into the embedded text,
  sentence-aligned windows, whole-sentence overlap, hard split for oversized paragraphs
- Metadata normalisation for attacker-controlled PDF and HTML fields
- Document lifecycle states exposed over the API, with failures recorded rather than lost
- Optional URL ingestion behind an explicit SSRF policy, disabled by default

**Retrieval**
- Exact dense cosine search over a cached per-tenant matrix, with generation-based invalidation
- Okapi BM25 over a persisted inverted index
- Reciprocal rank fusion
- Deterministic query rewriting: reference resolution, identifier surface forms, decomposition
- MMR reranking with a near-duplicate ceiling
- Neighbour stitching that preserves the applied content policy
- Per-stage latency and candidate counts as diagnostics, and a `GET /v1/retrieve` endpoint

**Generation**
- Trust-separated prompt construction with per-request nonce fences
- Citation extraction, resolution and verification; fabricated markers reported
- Per-sentence grounding measurement with a configurable refusal threshold
- Pre-generation evidence-coverage gate
- Extractive fallback when the configured model is unavailable, flagged in the response

**Security**
- API-key authentication with constant-time comparison and per-key tenant binding
- Tenant isolation enforced in SQL predicates
- Per-document ACLs, deny-by-default
- Layered prompt-injection detection: Unicode normalisation, structural signals,
  intent patterns, noisy-OR aggregation, configurable annotate/neutralise/drop
- Corroboration required before the risk threshold forces a drop
- Append-only audit trail containing no document or query text
- Secret redaction as a logging-pipeline stage
- Startup invariants that refuse unsafe production configurations

**Providers**
- `EmbeddingProvider` and `ChatProvider` protocols
- fastembed (local ONNX), Ollama, OpenAI-compatible, and a deterministic hashing embedder
- Retrying transport with jittered backoff and domain error mapping

**Operations**
- Structured JSON logging with request and trace correlation, unified with stdlib records
- OpenTelemetry spans and metrics
- `/healthz` (dependency-free) and `/readyz` (checks dependencies, reports backend warnings)
- Multi-stage container image running as a non-root user
- `docker-compose.yml` with PostgreSQL, Ollama and an OTLP collector

**Evaluation**
- JSONL dataset format with retrieval and answer expectations
- Recall@k, MRR, citation precision, grounding, latency and token metrics
- Threshold gate that fails CI, with adversarial cases gated at 100%
- A 17-case regression dataset and corpus, including a poisoned document and a
  false-positive control

**Developer experience**
- `python tasks.py` cross-platform task runner, with a `Makefile` that delegates to it
- CLI: `serve`, `ingest`, `query`, `evaluate`
- Three runnable examples
- CI, security and container workflows with no `continue-on-error`

### Security

- Every threat in [THREAT-MODEL.md](THREAT-MODEL.md) with a mechanical control
  has a regression test that fails if the control is removed.

[Unreleased]: https://github.com/kogunlowo123/rag-research-assistant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kogunlowo123/rag-research-assistant/releases/tag/v0.1.0
