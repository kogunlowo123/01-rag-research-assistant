# Configuration reference

All configuration is environment variables prefixed `RAG_`, with nesting
expressed by a double underscore: `RAG_RETRIEVAL__TOP_K` sets
`settings.retrieval.top_k`. A `.env` file in the working directory is read if
present; see [`.env.example`](../.env.example).

Values are validated at startup. An out-of-range value fails immediately with
the field name, rather than at the first request that touches it.

---

## Environment

| Variable | Default | Values |
| --- | --- | --- |
| `RAG_ENVIRONMENT` | `local` | `local`, `test`, `staging`, `production` |

Setting `production` enables additional startup invariants. The process refuses
to start if authentication is disabled, the database is SQLite, private-network
fetching is enabled, document-content logging is on, or SQL echo is on.

## Security

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_SECURITY__REQUIRE_API_KEY` | `true` | Cannot be `false` in production |
| `RAG_SECURITY__API_KEYS` | *(empty)* | Comma-separated. `tenant:secret` binds a key to a tenant; a bare secret binds to `default` |
| `RAG_SECURITY__INJECTION_ACTION` | `neutralise` | `annotate`, `neutralise`, `drop` |
| `RAG_SECURITY__INJECTION_BLOCK_THRESHOLD` | `0.95` | Aggregate risk forcing a drop, applied only when two or more rules fired |
| `RAG_SECURITY__MAX_REQUEST_BYTES` | `2097152` | Body limit, enforced before the body is read |
| `RAG_SECURITY__MAX_QUERY_CHARS` | `4000` | Query length limit |

Generate a key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Choosing an injection action:**

| Action | Behaviour | Use when |
| --- | --- | --- |
| `annotate` | Record the finding; pass the passage through unchanged | Investigating detection quality |
| `neutralise` | Replace instruction-like spans with an explicit marker, keep the rest | Default. Preserves the passage's factual content |
| `drop` | Exclude any flagged passage entirely | The corpus is known-hostile and losing content is acceptable |

## Storage

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_STORAGE__DATABASE_URL` | `sqlite+pysqlite:///./var/rag.db` | Sync and async driver names are both accepted and translated |
| `RAG_STORAGE__POOL_SIZE` | `5` | PostgreSQL only |
| `RAG_STORAGE__POOL_TIMEOUT_SECONDS` | `10.0` | PostgreSQL only |
| `RAG_STORAGE__ECHO_SQL` | `false` | Logs every statement. Refused in production |

## Embeddings

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_EMBEDDING__BACKEND` | `hashing` | `hashing`, `fastembed`, `ollama`, `openai` |
| `RAG_EMBEDDING__MODEL` | `BAAI/bge-small-en-v1.5` | Backend-specific |
| `RAG_EMBEDDING__DIMENSIONS` | `384` | Must match the model's native width |
| `RAG_EMBEDDING__BATCH_SIZE` | `32` | Passages per provider call |
| `RAG_EMBEDDING__CACHE_DIR` | `./var/models` | Where fastembed stores weights |

| Backend | Credentials | Network | Quality |
| --- | --- | --- | --- |
| `hashing` | none | none | Lexical overlap only. Default; tests and first run |
| `fastembed` | none | one-time model download | Real transformer embeddings, CPU. **Recommended** |
| `ollama` | none | local Ollama server | Depends on the model |
| `openai` | API key | outbound HTTPS | Depends on the model |

**Changing the backend or model requires re-ingesting.** Vectors record which
provider produced them and search filters on it, so after a change dense
retrieval returns nothing rather than ranking across incompatible spaces. BM25
keeps working, so the service degrades rather than failing. See
[`operations.md`](operations.md#reindexing).

## Generation

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_CHAT__BACKEND` | `extractive` | `extractive`, `ollama`, `openai` |
| `RAG_CHAT__MODEL` | `llama3.2:3b` | Backend-specific |
| `RAG_CHAT__TEMPERATURE` | `0.0` | Deterministic by default; grounding is the goal, not variety |
| `RAG_CHAT__MAX_OUTPUT_TOKENS` | `800` | |
| `RAG_CHAT__TIMEOUT_SECONDS` | `60.0` | |
| `RAG_CHAT__MAX_RETRIES` | `2` | Transient failures only, with jittered backoff |

`extractive` answers with verbatim source sentences. It cannot synthesise across
passages and cannot rephrase, but it needs no model and cannot hallucinate. It
is also the automatic fallback when the configured model is unavailable.

## Providers

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_PROVIDERS__OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | |
| `RAG_PROVIDERS__OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any compatible endpoint: Azure, vLLM, LiteLLM, llama.cpp |
| `RAG_PROVIDERS__OPENAI_API_KEY` | *(unset)* | Required only when an `openai` backend is selected |

## Ingestion

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_INGESTION__MAX_DOCUMENT_BYTES` | `10485760` | 10 MiB |
| `RAG_INGESTION__CHUNK_TARGET_TOKENS` | `320` | Window size |
| `RAG_INGESTION__CHUNK_OVERLAP_TOKENS` | `64` | Must be smaller than the target |
| `RAG_INGESTION__MAX_CHUNKS_PER_DOCUMENT` | `5000` | Ceiling; hitting it produces a warning |
| `RAG_INGESTION__ALLOW_URL_INGESTION` | `false` | Opt-in; it is an SSRF surface |
| `RAG_INGESTION__URL_ALLOWED_SCHEMES` | `https` | |
| `RAG_INGESTION__URL_ALLOWED_HOSTS` | *(empty)* | Required when URL ingestion is on; empty refuses everything |
| `RAG_INGESTION__URL_FETCH_TIMEOUT_SECONDS` | `10.0` | |
| `RAG_INGESTION__ALLOW_PRIVATE_NETWORK_FETCH` | `false` | Test fixtures only. Refused in production |

**Tuning chunk size.** Larger windows give a model more context per passage and
retrieve less precisely; smaller windows retrieve precisely and fragment facts
that span sentences. 320 tokens with 64 overlap suits policy and reference
documents. Prose narrative benefits from larger; structured reference material
from smaller. Change it and re-run the evaluation gate rather than guessing.

## Retrieval

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_RETRIEVAL__TOP_K` | `8` | Passages passed to generation |
| `RAG_RETRIEVAL__DENSE_CANDIDATES` | `40` | Per query variant |
| `RAG_RETRIEVAL__SPARSE_CANDIDATES` | `40` | Per query variant |
| `RAG_RETRIEVAL__RRF_K` | `60` | Fusion constant; higher damps the top ranks more |
| `RAG_RETRIEVAL__ENABLE_QUERY_REWRITE` | `true` | Additive; the original is always retrieved for |
| `RAG_RETRIEVAL__ENABLE_RERANK` | `true` | MMR |
| `RAG_RETRIEVAL__MMR_LAMBDA` | `0.7` | `1.0` pure relevance, `0.0` pure diversity |
| `RAG_RETRIEVAL__CONTEXT_WINDOW_NEIGHBOURS` | `1` | Adjacent chunks stitched around each hit |

`top_k` is the main cost lever with a hosted provider: it multiplies prompt
tokens almost linearly.

## Generation policy

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_GENERATION__MAX_CONTEXT_CHARS` | `24000` | Evidence budget; passages beyond it are truncated and reported |
| `RAG_GENERATION__MIN_GROUNDING_SCORE` | `0.35` | Answers below this are refused |
| `RAG_GENERATION__MIN_EVIDENCE_COVERAGE` | `0.5` | Query subject terms that must appear in the evidence before a model is called. `0` disables |
| `RAG_GENERATION__REQUIRE_CITATIONS` | `true` | Refuse an answer with no resolvable citation |
| `RAG_GENERATION__REFUSE_WHEN_UNSUPPORTED` | `true` | Enforce the grounding threshold |
| `RAG_GENERATION__MAX_SENTENCES` | `4` | Upper bound for the extractive backend |

**Tuning the grounding threshold.** Raising it increases refusals and decreases
the chance an unsupported answer is returned. `0.35` is permissive enough for a
paraphrasing model; a corpus of short factual statements tolerates `0.6` or
higher. Measure with the evaluation gate rather than adjusting by feel.

## Observability

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_OBSERVABILITY__LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `RAG_OBSERVABILITY__LOG_FORMAT` | `json` | `json` or `console` |
| `RAG_OBSERVABILITY__SERVICE_NAME` | `rag-research-assistant` | Appears on every record and span |
| `RAG_OBSERVABILITY__TRACING_ENABLED` | `false` | |
| `RAG_OBSERVABILITY__OTLP_ENDPOINT` | *(unset)* | Optional; trace ids reach logs even without a collector |
| `RAG_OBSERVABILITY__TRACE_SAMPLE_RATIO` | `1.0` | |
| `RAG_OBSERVABILITY__LOG_DOCUMENT_CONTENT` | `false` | User data. Refused in production |

## Profiles

**Zero setup** — the defaults. No credentials, no downloads, no services.

```bash
RAG_SECURITY__API_KEYS=acme:$(python -c "import secrets;print(secrets.token_urlsafe(32))")
```

**Local, credential-free, production-quality models:**

```bash
RAG_EMBEDDING__BACKEND=fastembed
RAG_CHAT__BACKEND=ollama
RAG_CHAT__MODEL=llama3.2:3b
```

**Hosted models:**

```bash
RAG_EMBEDDING__BACKEND=openai
RAG_EMBEDDING__MODEL=text-embedding-3-small
RAG_EMBEDDING__DIMENSIONS=1536
RAG_CHAT__BACKEND=openai
RAG_CHAT__MODEL=gpt-4o-mini
RAG_PROVIDERS__OPENAI_API_KEY=<from your secret manager>
```

**Hostile corpus** — maximum caution, at the cost of losing some content:

```bash
RAG_SECURITY__INJECTION_ACTION=drop
RAG_GENERATION__MIN_GROUNDING_SCORE=0.6
RAG_GENERATION__MIN_EVIDENCE_COVERAGE=0.6
```

## Verifying a configuration

```bash
uv run python -c "from rag_assistant.config import get_settings; \
                  s = get_settings(); s.enforce_environment_invariants(); \
                  print('ok:', s.environment, s.embedding.backend, s.chat.backend)"
```

Secrets are `SecretStr`, so printing the settings object never reveals a key.
