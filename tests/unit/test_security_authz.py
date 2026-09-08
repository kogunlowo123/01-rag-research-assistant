"""Authentication and document authorisation."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from rag_assistant.errors import AuthorizationError
from rag_assistant.security.authz import (
    ApiKeyAuthenticator,
    Principal,
    can_read_document,
    digest_key,
    key_id,
    require_read,
)

pytestmark = pytest.mark.unit

ACME_KEY = "acme-secret-value-for-tests"
GLOBEX_KEY = "globex-secret-value-for-tests"


@pytest.fixture
def authenticator() -> ApiKeyAuthenticator:
    return ApiKeyAuthenticator(
        keys=(SecretStr(f"acme:{ACME_KEY}"), SecretStr(f"globex:{GLOBEX_KEY}")),
        require_key=True,
    )


class TestApiKeyAuthenticator:
    def test_valid_key_resolves_to_its_bound_tenant(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        assert authenticator.authenticate(ACME_KEY).tenant_id == "acme"
        assert authenticator.authenticate(GLOBEX_KEY).tenant_id == "globex"

    def test_tenant_comes_from_the_credential_not_the_caller(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        """There is no API surface by which a caller can choose its own tenant."""
        principal = authenticator.authenticate(ACME_KEY)
        assert principal.tenant_id == "acme"
        assert authenticator.authenticate(GLOBEX_KEY).tenant_id != principal.tenant_id

    @pytest.mark.parametrize("presented", [None, "", "   ", "wrong-key", ACME_KEY + "x"])
    def test_missing_or_wrong_keys_are_refused(
        self, authenticator: ApiKeyAuthenticator, presented: str | None
    ) -> None:
        with pytest.raises(AuthorizationError):
            authenticator.authenticate(presented)

    def test_refusal_message_is_identical_for_missing_and_wrong_keys(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        """The error must not act as an oracle distinguishing the two cases."""
        with pytest.raises(AuthorizationError) as absent:
            authenticator.authenticate(None)
        with pytest.raises(AuthorizationError) as wrong:
            authenticator.authenticate("not-the-key")
        assert str(absent.value) == str(wrong.value)

    def test_surrounding_whitespace_is_tolerated(self, authenticator: ApiKeyAuthenticator) -> None:
        assert authenticator.authenticate(f"  {ACME_KEY}  ").tenant_id == "acme"

    def test_key_without_a_prefix_binds_to_the_default_tenant(self) -> None:
        auth = ApiKeyAuthenticator(keys=(SecretStr("bare-key-value"),), require_key=True)
        assert auth.authenticate("bare-key-value").tenant_id == "default"

    def test_disabled_authentication_yields_an_anonymous_principal(self) -> None:
        auth = ApiKeyAuthenticator(keys=(), require_key=False)
        principal = auth.authenticate(None)
        assert principal.is_anonymous
        assert principal.tenant_id == "default"

    def test_key_id_is_short_stable_and_not_the_key(self) -> None:
        identifier = authenticator_key_id = key_id(ACME_KEY)
        assert len(identifier) == 8
        assert identifier == key_id(ACME_KEY)
        assert ACME_KEY not in identifier
        assert digest_key(ACME_KEY).startswith(authenticator_key_id)

    def test_principal_never_carries_the_raw_secret(
        self, authenticator: ApiKeyAuthenticator
    ) -> None:
        principal = authenticator.authenticate(ACME_KEY)
        assert ACME_KEY not in repr(principal)


class TestDocumentAuthorization:
    ACME = Principal(tenant_id="acme", key_id="aaaaaaaa", groups=frozenset({"finance"}))

    def test_same_tenant_and_empty_acl_is_readable(self) -> None:
        assert can_read_document(self.ACME, tenant_id="acme", acl=())

    def test_other_tenant_is_never_readable(self) -> None:
        assert not can_read_document(self.ACME, tenant_id="globex", acl=())

    def test_other_tenant_is_refused_even_with_a_matching_group(self) -> None:
        assert not can_read_document(self.ACME, tenant_id="globex", acl=("finance",))

    def test_matching_group_grants_access(self) -> None:
        assert can_read_document(self.ACME, tenant_id="acme", acl=("finance", "legal"))

    def test_non_matching_group_denies_access(self) -> None:
        assert not can_read_document(self.ACME, tenant_id="acme", acl=("legal",))

    def test_principal_without_groups_is_denied_a_restricted_document(self) -> None:
        bare = Principal(tenant_id="acme", key_id="bbbbbbbb")
        assert not can_read_document(bare, tenant_id="acme", acl=("finance",))

    def test_require_read_raises_with_a_non_enumerating_message(self) -> None:
        with pytest.raises(AuthorizationError) as raised:
            require_read(self.ACME, tenant_id="globex", acl=())
        assert "not found or not accessible" in str(raised.value)

    def test_require_read_is_silent_when_permitted(self) -> None:
        require_read(self.ACME, tenant_id="acme", acl=("finance",))
