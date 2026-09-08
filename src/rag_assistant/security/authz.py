"""Authentication and document-level authorisation.

Two boundaries are enforced here.

**Tenant isolation** is applied in the data layer, not by filtering results
after retrieval. Every query carries a tenant id that becomes part of the SQL
predicate and the vector-index partition key, so a bug in ranking cannot leak a
document across tenants. Post-filtering would also silently shrink result sets
below ``top_k``, which is how "why did retrieval get worse?" incidents start.

**Document ACLs** are checked before a chunk can be used as evidence. A
document with an empty ACL is visible to the whole tenant; otherwise the caller
must hold at least one of the listed groups. The check is deny-by-default: an
unknown principal sees nothing.

API keys are compared with :func:`hmac.compare_digest` and are matched against
their SHA-256 digests, so the configured secret is never held in a
comparison-visible form longer than necessary.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from pydantic import SecretStr

from rag_assistant.errors import AuthorizationError


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller.

    ``key_id`` is the first eight characters of the key's digest. It is safe to
    log, is stable for a given key, and lets an operator answer "which key did
    this?" without the key ever appearing in a log.
    """

    tenant_id: str
    key_id: str
    groups: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_anonymous(self) -> bool:
        """Whether the caller authenticated with no credential."""
        return self.key_id == "anonymous"


ANONYMOUS_TENANT = "public"


def digest_key(raw: str) -> str:
    """Return the hex SHA-256 digest of an API key."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_id(raw: str) -> str:
    """Short, log-safe identifier derived from an API key."""
    return digest_key(raw)[:8]


class ApiKeyAuthenticator:
    """Constant-time API key authentication with a per-key tenant binding.

    A key may encode its tenant as ``tenant:secret``. Keys without a prefix are
    bound to the default tenant. Binding the tenant to the credential rather
    than accepting it from a request header is what makes tenant isolation an
    authentication property instead of a client-supplied hint.
    """

    def __init__(
        self,
        *,
        keys: tuple[SecretStr, ...],
        require_key: bool,
        default_tenant: str = "default",
    ) -> None:
        """Precompute digests for the configured keys."""
        self._require_key = require_key
        self._default_tenant = default_tenant
        self._by_digest: dict[str, tuple[str, str]] = {}
        for key in keys:
            raw = key.get_secret_value()
            tenant, _, secret = raw.partition(":")
            if not secret:
                tenant, secret = default_tenant, raw
            self._by_digest[digest_key(secret)] = (tenant, key_id(secret))

    @property
    def requires_key(self) -> bool:
        """Whether a credential is mandatory."""
        return self._require_key

    def authenticate(self, presented: str | None) -> Principal:
        """Resolve a presented credential to a :class:`Principal`.

        Raises :class:`AuthorizationError` when a key is required and the
        presented value does not match. The error message never distinguishes
        "no key" from "wrong key", so it cannot be used as an oracle.
        """
        if not self._require_key:
            return Principal(tenant_id=self._default_tenant, key_id="anonymous")

        if not presented:
            raise AuthorizationError("a valid API key is required")

        candidate = digest_key(presented.strip())
        for known_digest, (tenant, identifier) in self._by_digest.items():
            if hmac.compare_digest(candidate, known_digest):
                return Principal(tenant_id=tenant, key_id=identifier)
        raise AuthorizationError("a valid API key is required")


def can_read_document(principal: Principal, *, tenant_id: str, acl: tuple[str, ...]) -> bool:
    """Whether ``principal`` may read a document.

    Deny-by-default: a tenant mismatch fails regardless of ACL, and a non-empty
    ACL requires group membership.
    """
    if principal.tenant_id != tenant_id:
        return False
    if not acl:
        return True
    return bool(principal.groups & set(acl))


def require_read(principal: Principal, *, tenant_id: str, acl: tuple[str, ...]) -> None:
    """Raise unless ``principal`` may read the document.

    The message is deliberately identical to the not-found case at the API
    layer so that a caller cannot enumerate document ids by comparing 403 and
    404 responses.
    """
    if not can_read_document(principal, tenant_id=tenant_id, acl=acl):
        raise AuthorizationError("document not found or not accessible")


__all__ = [
    "ANONYMOUS_TENANT",
    "ApiKeyAuthenticator",
    "Principal",
    "can_read_document",
    "digest_key",
    "key_id",
    "require_read",
]
