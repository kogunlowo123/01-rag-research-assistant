# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Report it privately through GitHub Security Advisories:

1. Go to the [Security tab](https://github.com/kogunlowo123/rag-research-assistant/security/advisories)
2. Select **Report a vulnerability**

Please include: what the issue is, how to reproduce it, the impact you believe it
has, and the version or commit you tested.

**What to expect**

| Stage | Target |
| --- | --- |
| Acknowledgement | 3 working days |
| Initial assessment | 10 working days |
| Fix or documented mitigation for a confirmed high-severity issue | 30 days |

This is a portfolio project maintained by one person, not a funded product.
Those targets are honest intent, not a contractual SLA. Credit is given in the
advisory and the changelog unless you ask otherwise.

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes — the current line |
| Older | No |

Security fixes land on `main` and in the next patch release.

## Security architecture

The design is described in full in [THREAT-MODEL.md](THREAT-MODEL.md) and
[ARCHITECTURE.md](ARCHITECTURE.md). The short version:

**Retrieved document text never carries authority.** Every string reaching a
prompt is tagged with a trust level, and `UNTRUSTED` content — which is all
document content, regardless of who uploaded it — enters only the user role,
inside per-request nonce-delimited evidence blocks. No code path places document
text into a system message.

**The application exposes no tool a model could be persuaded to call.** An
injected instruction has nothing to actuate.

**An answer is refused unless it traces back to retrieved evidence.** Grounding
is measured per sentence against the passage cited; citations are resolved and
verified; fabricated markers are reported.

**The tenant is bound to the credential**, never taken from a request header, and
every database query filters on it in SQL rather than after the fact.

**Detection is a layer, not the foundation.** Prompt-injection detection reduces
exposure. The three controls above do not depend on it.

## Controls at a glance

| Area | Control |
| --- | --- |
| Authentication | API key, SHA-256 digest compared with `hmac.compare_digest`; key never logged |
| Authorisation | Tenant bound to credential; per-document ACLs, deny-by-default; 403 and 404 indistinguishable |
| Prompt injection | Unicode normalisation, structural signals, intent patterns, configurable annotate/neutralise/drop, corroboration required before dropping |
| Upload safety | Magic-byte validation, archive and executable refusal, size limits at three layers, bounded parsers |
| SSRF | Opt-in, scheme and host allowlists, post-DNS address validation, redirects refused, streaming size limit |
| Path traversal | Filenames are display labels; content addressed by digest |
| Secrets | `SecretStr` in configuration, redaction as the final logging processor and again after traceback rendering |
| Transport | Timeouts always set, jittered backoff, upstream bodies never forwarded |
| HTTP hardening | CSP, `nosniff`, `DENY` framing, `no-referrer`, COOP/CORP, body-size limit |
| Container | Multi-stage build, non-root uid 10001, no shell for the app user, no build toolchain, no secrets in the image |
| Startup | Unsafe production configurations refuse to start |

## Security assumptions

The design assumes all of the following. If one is false in your deployment, the
analysis in [THREAT-MODEL.md](THREAT-MODEL.md) does not hold.

1. **API keys are distributed and stored securely** and injected from a secret
   manager, not from a file baked into an image.
2. **The service runs behind a gateway that enforces TLS, rate limiting and
   quotas.** None of those are implemented here.
3. **The database is not directly reachable from untrusted networks**, and
   encryption at rest is handled by the database or the volume.
4. **The client rendering answers does not automatically fetch URLs found in
   them.** A rendering client that auto-loads images from answer text
   reintroduces an exfiltration channel this service otherwise closes.
5. **The configured model provider is trusted with prompt contents.** Prompts
   contain retrieved documents. Run Ollama locally, or point the
   OpenAI-compatible adapter at a self-hosted endpoint, if that is unacceptable.
6. **Operators read `/readyz`.** It reports when the service is running on the
   deterministic development backends.

## Known limitations

Stated plainly, because a security document that lists only strengths is not
useful.

1. **Prompt injection is mitigated, not solved.** Detection based on patterns
   and structure will miss novel phrasings, and natural-language paraphrase of
   an injection is an open research problem. This project makes no claim of
   complete protection.
2. **Grounding is measured by term coverage, not entailment.** It detects a
   sentence discussing something the passage does not mention; it does not
   reliably detect a reversal of meaning such as a dropped "not".
3. **No rate limiting or quotas.** A valid key can issue unlimited requests.
   Enforce this at the gateway.
4. **DNS rebinding is not fully closed.** The SSRF guard validates every
   resolved address but does not pin the address into the connection.
5. **ACL groups are not sourced from an identity provider.** They are carried on
   the principal.
6. **The audit log is append-only by convention, not cryptographically signed.**
7. **A compromised model provider sees prompt contents.** Inherent; the local
   path exists so it can be avoided.
8. **No content moderation of generated answers.** Out of scope.

## Automated security testing

Every push and pull request runs, with no `continue-on-error` anywhere:

| Check | Tool | Scope |
| --- | --- | --- |
| Secret scanning | gitleaks | Working tree **and** full git history |
| Static analysis | bandit | All first-party source |
| Static analysis | ruff `S` rules | All first-party source |
| Dependency vulnerabilities | pip-audit | Resolved lockfile, `--strict` |
| Code scanning | CodeQL | `security-extended` query pack |
| Filesystem scan | Trivy | Vulnerabilities, secrets, misconfiguration |
| Image scan | Trivy | Built container, HIGH and CRITICAL |
| Non-root assertion | Container job | Fails if the image runs as uid 0 |
| Adversarial suite | pytest `-m security` | ~90 cases; a failure is a security regression |

The security workflow also runs weekly, so a vulnerability disclosed after a
merge is found without a push.

A dependency vulnerability with no upstream fix is handled by a dated, justified
entry in [`security/audit-exceptions.md`](security/audit-exceptions.md) and a
matching identifier in `security/audit-ignores.txt` — never by suppressing the
scanner. An exception with an expired review date is treated as a failure.

## Running the scans locally

```bash
python tasks.py security                       # bandit + pip-audit
python tasks.py test-security                  # the adversarial suite
uv run bandit -c pyproject.toml -r src
docker run --rm -v "$PWD:/src" aquasec/trivy fs --severity HIGH,CRITICAL /src
```

## Hardening checklist for a deployment

```bash
RAG_ENVIRONMENT=production                     # enables the startup invariants
RAG_SECURITY__REQUIRE_API_KEY=true             # cannot be disabled in production
RAG_SECURITY__API_KEYS=<from your secret manager>
RAG_STORAGE__DATABASE_URL=postgresql+psycopg://...   # SQLite is refused
RAG_SECURITY__INJECTION_ACTION=neutralise      # or drop, for a hostile corpus
RAG_GENERATION__REFUSE_WHEN_UNSUPPORTED=true
RAG_GENERATION__REQUIRE_CITATIONS=true
RAG_INGESTION__ALLOW_URL_INGESTION=false       # unless you need it, with an allowlist
RAG_OBSERVABILITY__LOG_DOCUMENT_CONTENT=false  # refused in production anyway
```

Startup fails, loudly, if any of these is inconsistent. That is deliberate: a
misconfigured deployment should refuse to serve rather than silently drop a
control.
