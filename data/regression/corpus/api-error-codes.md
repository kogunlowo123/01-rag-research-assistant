# Platform API Error Codes

Each error returned by the platform API carries a stable machine-readable code.
Clients should branch on the code, never on the human-readable message.

## Authentication errors

ERR_1001 — The API key is missing from the request.
ERR_1002 — The API key is malformed or has been revoked.
ERR_1003 — The API key is valid but lacks the scope required for this operation.

## Rate limiting

ERR_2200 — The per-minute request quota for this key has been exhausted. The
Retry-After header carries the number of seconds to wait.
ERR_2201 — The per-day quota has been exhausted. Quotas reset at 00:00 UTC.

## Ingestion errors

ERR_4417 — The uploaded document exceeded the configured size limit. The limit
is returned in the detail object as limit_bytes. Split the document and retry.
ERR_4418 — The uploaded document's declared media type does not match its
content signature.
ERR_4419 — No parser is registered for the supplied media type.

## Upstream errors

ERR_5030 — The upstream model provider did not respond within the timeout
budget. This error is safe to retry with exponential backoff.
