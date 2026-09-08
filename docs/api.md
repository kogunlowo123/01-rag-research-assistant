# API guide

The OpenAPI document is served at `/openapi.json` and rendered at `/docs`. This
page covers the parts a schema cannot express: what the fields mean, what a
refusal is, and how to debug a bad answer.

## Authentication

Present the key as `X-API-Key` or as a bearer token:

```bash
curl -H "X-API-Key: $RAG_API_KEY" ...
curl -H "Authorization: Bearer $RAG_API_KEY" ...
```

A key may encode its tenant as `tenant:secret`. The tenant comes from the
credential, never from a request header — there is no way for a caller to
select its own tenant.

Missing and incorrect keys both return `403` with the same body. That is
deliberate: a distinguishable response is an oracle.

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/healthz` | no | Liveness. Touches no dependency. |
| `GET` | `/readyz` | no | Readiness plus backend warnings. `503` when not ready. |
| `POST` | `/v1/documents` | yes | Upload and index a document (multipart). |
| `POST` | `/v1/documents/from-url` | yes | Fetch and index by URL. Disabled by default. |
| `GET` | `/v1/documents` | yes | List documents for the tenant. |
| `GET` | `/v1/documents/{id}` | yes | Document metadata and ingestion status. |
| `DELETE` | `/v1/documents/{id}` | yes | Remove a document and its index entries. |
| `POST` | `/v1/query` | yes | Ask a question. |
| `GET` | `/v1/retrieve` | yes | Retrieval diagnostics without generation. |
| `GET` | `/v1/audit` | yes | Recent audit events for the tenant. |

---

## Upload a document

```bash
curl -X POST http://127.0.0.1:8000/v1/documents \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@handbook.pdf;type=application/pdf" \
  -F "title=Employee Handbook 2026" \
  -F "acl=hr,legal"
```

`acl` is an optional comma-separated list of group identifiers. A document with
no ACL is visible tenant-wide; a document with one requires the caller to hold a
listed group.

```json
{
  "document": {
    "id": "doc_3f2a…",
    "title": "Employee Handbook 2026",
    "media_type": "application/pdf",
    "status": "indexed",
    "chunk_count": 214,
    "acl": ["hr", "legal"]
  },
  "chunks_created": 214,
  "duplicate_of": null,
  "warnings": [],
  "injection_findings": [],
  "latency_ms": {"validate_ms": 1.2, "parse_ms": 812.4, "embed_ms": 1904.0}
}
```

Notable response fields:

- **`duplicate_of`** — set when the content digest already exists for this
  tenant. `chunks_created` is then `0` and nothing was re-indexed.
- **`injection_findings`** — instruction-like text was found in the document.
  It is still indexed; the finding governs how its passages are treated at
  retrieval time. This is information, not an error.
- **`warnings`** — human-readable notes such as a hit chunk ceiling.

Failure modes:

| Status | `code` | Cause |
| --- | --- | --- |
| `413` | `document_too_large` | Body or document exceeds the limit |
| `415` | `unsupported_media_type` | Empty, archive, executable, or a type/signature mismatch |
| `422` | `ingestion_failed` | Parsed but produced nothing indexable (for example a scanned PDF) |

## Ask a question

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "X-API-Key: $RAG_API_KEY" \
  -H 'content-type: application/json' \
  -d '{
        "query": "How long do employees have to submit an expense claim?",
        "session_id": "thread-42",
        "top_k": 8,
        "include_diagnostics": true
      }'
```

```json
{
  "answer": "Expense claims must be submitted within 60 days of the expense date. [1]",
  "refused": false,
  "refusal_reason": null,
  "confidence": 0.94,
  "citations": [
    {
      "marker": "[1]",
      "document_id": "doc_3f2a…",
      "document_title": "Employee Handbook 2026",
      "chunk_id": "chk_91b7…",
      "locator": "p.14 Expenses > Submission #37",
      "quote": "Expense claims must be submitted within 60 days of the expense date.",
      "support_score": 1.0
    }
  ],
  "grounding": {
    "score": 1.0,
    "citation_coverage": 1.0,
    "total_sentences": 1,
    "supported_sentences": 1,
    "is_grounded": true,
    "unsupported_sentences": []
  },
  "warnings": [],
  "provider": "ollama",
  "model": "llama3.2:3b",
  "prompt_tokens": 1180,
  "completion_tokens": 18,
  "latency_ms": 842.1,
  "session_id": "thread-42"
}
```

### Reading the response

**`refused`** — a refusal is a **200**, not an error. Refusing is a correct
outcome, and treating it as a failure would make refusal rate invisible in your
error dashboards. Check `refused` before `answer`.

Refusals happen when: nothing was retrieved; the retrieved passages do not
discuss the question's subject; the model itself declined; no citation resolved;
or measured grounding fell below the threshold. `refusal_reason` says which.

**`grounding.score`** — the fraction of answer sentences traceable to the
passage they cite. `1.0` means every sentence checked out.

**`grounding.citation_coverage`** — the fraction of sentences carrying any
resolvable citation. A high score with low coverage means the answer is
under-cited.

**`confidence`** — grounding attenuated by citation coverage. A signal for a
human reviewer, **not a calibrated probability**.

**`support_score`** on a citation — how much of the citing sentence's content
appears in that passage.

**`warnings`** — always worth surfacing. They include truncated context,
degraded generation (the model was unavailable and the extractive fallback
answered), fabricated citation markers, and a compromised-corpus warning when
most retrieved passages contained instruction-like text.

### Sessions

Pass a stable `session_id` to make follow-up questions resolvable: "and what
about contractors?" is rewritten using the previous turns before retrieval runs.
History is bounded at 20 turns.

## Debug a bad answer

`GET /v1/retrieve` runs retrieval and stops. No model call, no tokens.

```bash
curl -G http://127.0.0.1:8000/v1/retrieve \
  -H "X-API-Key: $RAG_API_KEY" \
  --data-urlencode "q=expense claim deadline" \
  --data-urlencode "top_k=8"
```

```json
{
  "original_query": "expense claim deadline",
  "rewritten_queries": ["expense claim deadline expenses submission"],
  "dense_candidate_count": 40,
  "sparse_candidate_count": 22,
  "fused_candidate_count": 46,
  "reranked": true,
  "dropped_by_policy": 0,
  "dropped_by_authorization": 3,
  "neutralised_chunks": 1,
  "latency_ms": {
    "rewrite_ms": 0.4, "search_ms": 31.2, "fuse_ms": 0.9,
    "hydrate_ms": 6.1, "policy_ms": 2.2, "rerank_ms": 1.8, "stitch_ms": 4.0
  }
}
```

How to read it:

| Observation | Likely cause |
| --- | --- |
| `sparse_candidate_count` is 0 | No query term is in the index — check tokenisation, or the term genuinely does not appear |
| `dense_candidate_count` is 0 | Nothing embedded with the current provider — the embedding model changed without a reindex |
| `dropped_by_authorization` is high | The caller lacks the ACL groups for the matching documents |
| `dropped_by_policy` is high | The matching passages look like injection attempts |
| `search_ms` dominates | Corpus size, or a slow embedding provider |
| Both counts healthy but the answer is wrong | A generation problem, not a retrieval one |

## Audit trail

```bash
curl -H "X-API-Key: $RAG_API_KEY" http://127.0.0.1:8000/v1/audit?limit=50
```

Events carry identifiers, rule ids, scores and decisions — never document or
query text — so the trail can be retained under a longer policy than the content
it describes.

| Event | Outcomes |
| --- | --- |
| `document.ingest` | `indexed`, `duplicate`, `failed` |
| `document.delete` | `deleted` |
| `query.answer` | `answered`, `refused` |

## Errors

Every failure returns the same shape:

```json
{
  "code": "unsupported_media_type",
  "message": "this media type is not in the ingestion allowlist",
  "request_id": "9f2c1d…",
  "detail": {"media_type": "application/zip"}
}
```

Branch on `code`, never on `message`. `request_id` matches the
`X-Request-ID` response header and the `request_id` field in every log line for
that request; quote it in a bug report.

| `code` | Status |
| --- | --- |
| `validation_error` | 400 / 422 |
| `unsafe_source` | 400 |
| `forbidden`, `policy_violation` | 403 |
| `not_found` | 404 |
| `document_too_large` | 413 |
| `unsupported_media_type` | 415 |
| `ingestion_failed` | 422 |
| `internal_error` | 500 |
| `provider_error` | 502 |
| `provider_unavailable` | 503 |
| `provider_timeout` | 504 |

Error bodies never echo the submitted value, and never forward an upstream
provider's response body.

## Headers

| Header | Direction | Purpose |
| --- | --- | --- |
| `X-API-Key` | request | Authentication |
| `Authorization: Bearer` | request | Authentication, alternative |
| `X-Request-ID` | both | Correlation. Supplied values are sanitised; anything unsafe is replaced. |

Every response also carries `Content-Security-Policy`, `X-Content-Type-Options`,
`X-Frame-Options`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`,
`Cross-Origin-Resource-Policy` and `Permissions-Policy`.
