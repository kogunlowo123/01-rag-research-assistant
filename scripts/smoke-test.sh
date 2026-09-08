#!/usr/bin/env bash
# Container smoke test.
#
# Starts the image, waits for liveness, then exercises the primary user path:
# authenticate, upload a document, ask a question about it, and check that the
# answer is grounded and cited. A container that starts but cannot answer is
# not a passing container.
set -euo pipefail

IMAGE="${1:?usage: smoke-test.sh <image[:tag]>}"
PORT="${SMOKE_PORT:-8123}"
NAME="rag-smoke-$$"
# A relative path: on a Windows host running Git Bash, curl is a native binary
# and cannot resolve the POSIX-style /tmp path this shell would produce.
DOC="./smoke-document-$$.md"
API_KEY="smoke-test-key-not-a-real-secret"

cleanup() {
  echo "--- container logs (tail) ---" >&2
  docker logs "${NAME}" 2>&1 | tail -40 >&2 || true
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  rm -f "${DOC}"
}
trap cleanup EXIT

echo "starting ${IMAGE}"
docker run -d --name "${NAME}" \
  -p "${PORT}:8000" \
  -e "RAG_SECURITY__API_KEYS=acme:${API_KEY}" \
  -e "RAG_OBSERVABILITY__LOG_FORMAT=json" \
  "${IMAGE}" >/dev/null

base="http://127.0.0.1:${PORT}"

echo "waiting for liveness"
for _ in $(seq 1 60); do
  if curl -fsS "${base}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "${base}/healthz" | grep -q '"status": *"ok"'
echo "liveness ok"

curl -fsS "${base}/readyz" | grep -q '"status": *"ready"'
echo "readiness ok"

echo "checking that an unauthenticated request is refused"
status="$(curl -s -o /dev/null -w '%{http_code}' "${base}/v1/documents")"
test "${status}" = "403"

echo "uploading a document"
cat > "${DOC}" <<'DOC'
# Refund Policy

## Eligibility

Customers may request a refund within 30 days of the original purchase date.
Proof of purchase is required for every refund request.

## Processing

Approved refunds are processed within 5 business days of approval.
DOC

curl -fsS -X POST "${base}/v1/documents" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@${DOC};type=text/markdown" \
  | grep -q '"status": *"indexed"'
echo "ingestion ok"

echo "asking a question"
answer="$(curl -fsS -X POST "${base}/v1/query" \
  -H "X-API-Key: ${API_KEY}" \
  -H 'content-type: application/json' \
  -d '{"query":"How many days do customers have to request a refund?"}')"

echo "${answer}"
echo "${answer}" | grep -q '"refused": *false'
echo "${answer}" | grep -q '30 days'
echo "${answer}" | grep -q '"citations": *\['
echo "${answer}" | grep -q '"document_title"'
echo "answer is grounded and cited"

echo "checking that an unanswerable question is refused rather than invented"
refusal="$(curl -fsS -X POST "${base}/v1/query" \
  -H "X-API-Key: ${API_KEY}" \
  -H 'content-type: application/json' \
  -d '{"query":"What is the company cryptocurrency treasury policy?"}')"
echo "${refusal}" | grep -q '"refused": *true'
echo "refusal ok"

echo "SMOKE TEST PASSED for ${IMAGE}"
