"""Source validation: filenames, media types and the SSRF policy."""

from __future__ import annotations

import socket

import pytest

from rag_assistant.errors import (
    DocumentTooLargeError,
    UnsafeSourceError,
    UnsupportedMediaTypeError,
)
from rag_assistant.security import sources
from rag_assistant.security.sources import (
    safe_filename,
    sniff_media_type,
    validate_upload,
    validate_url,
)

pytestmark = pytest.mark.unit

ALLOWED = frozenset({"text/plain", "text/markdown", "text/html", "application/pdf"})


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("report.pdf", "report.pdf"),
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\config", "config"),
            ("/absolute/path/notes.md", "notes.md"),
            ("C:\\Users\\bob\\notes.md", "notes.md"),
            ("notes.md:$DATA", "notes.md"),
            ("", "document"),
            ("...", "document"),
            ("   ", "document"),
        ],
    )
    def test_traversal_and_stream_suffixes_are_stripped(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected

    def test_reserved_windows_device_names_are_defused(self) -> None:
        assert safe_filename("CON.txt") != "CON.txt"
        assert safe_filename("nul") != "nul"

    def test_control_and_bidi_characters_are_replaced(self) -> None:
        result = safe_filename("inv\u202eoice\u200bfdp.exe")
        assert "\u202e" not in result
        assert "\u200b" not in result

    def test_long_names_are_truncated_but_keep_their_extension(self) -> None:
        result = safe_filename("a" * 400 + ".pdf")
        assert len(result) <= 120
        assert result.endswith(".pdf")

    def test_result_never_contains_a_separator(self) -> None:
        for raw in ("a/b/c.txt", "a\\b\\c.txt", "..%2f..%2fetc"):
            assert "/" not in safe_filename(raw)
            assert "\\" not in safe_filename(raw)


class TestMediaTypeSniffing:
    def test_pdf_signature_is_recognised(self) -> None:
        assert sniff_media_type(b"%PDF-1.7\nrest", "application/pdf") == "application/pdf"

    def test_declared_pdf_without_signature_is_refused(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            sniff_media_type(b"not a pdf at all", "application/pdf")

    def test_zip_is_always_refused_even_when_declared_as_text(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            sniff_media_type(b"PK\x03\x04rest", "text/plain")

    def test_executable_is_always_refused(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            sniff_media_type(b"\x7fELF\x02\x01\x01", "text/plain")

    def test_declaration_contradicting_the_signature_is_refused(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            sniff_media_type(b"%PDF-1.7", "text/html")

    def test_signature_free_text_keeps_its_declaration(self) -> None:
        assert sniff_media_type(b"# Heading", "text/markdown") == "text/markdown"


class TestValidateUpload:
    def test_empty_upload_is_refused(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            validate_upload(
                b"", declared_media_type="text/plain", allowed_media_types=ALLOWED, max_bytes=100
            )

    def test_oversized_upload_is_refused(self) -> None:
        with pytest.raises(DocumentTooLargeError):
            validate_upload(
                b"x" * 200,
                declared_media_type="text/plain",
                allowed_media_types=ALLOWED,
                max_bytes=100,
            )

    def test_media_type_outside_the_allowlist_is_refused(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            validate_upload(
                b"body",
                declared_media_type="application/x-tar",
                allowed_media_types=ALLOWED,
                max_bytes=1000,
            )

    def test_charset_parameter_is_ignored_when_matching(self) -> None:
        assert (
            validate_upload(
                b"hello",
                declared_media_type="text/plain; charset=utf-8",
                allowed_media_types=ALLOWED,
                max_bytes=1000,
            )
            == "text/plain"
        )


class TestValidateUrl:
    ALLOWED_HOSTS = frozenset({"docs.example.com"})
    SCHEMES = frozenset({"https"})

    def _validate(self, url: str, **kwargs: object) -> str:
        return validate_url(
            url,
            allowed_schemes=self.SCHEMES,
            allowed_hosts=self.ALLOWED_HOSTS,
            **kwargs,  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize(
        "url",
        [
            "http://docs.example.com/a",  # scheme not allowed
            "file:///etc/passwd",
            "gopher://docs.example.com/",
            "https://evil.example.com/a",  # host not allowed
            "https://user:pass@docs.example.com/a",  # embedded credentials
            "https://docs.example.com:8080/a",  # non-standard port
            "https:///no-host",
        ],
    )
    def test_policy_violations_are_refused(self, url: str) -> None:
        with pytest.raises(UnsafeSourceError):
            self._validate(url, allow_private_network=True)

    def test_allowlisted_https_url_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["93.184.216.34"])
        assert self._validate("https://docs.example.com/paper.pdf").startswith("https://")

    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # cloud metadata
            "0.0.0.0",
            "::1",
            "fd00::1",
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
        ],
    )
    def test_private_and_metadata_addresses_are_refused_after_resolution(
        self, address: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: [address])
        with pytest.raises(UnsafeSourceError):
            self._validate("https://docs.example.com/a")

    def test_a_single_private_record_among_public_ones_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["93.184.216.34", "127.0.0.1"])
        with pytest.raises(UnsafeSourceError):
            self._validate("https://docs.example.com/a")

    def test_unresolvable_host_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> list[str]:
            raise socket.gaierror("name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        with pytest.raises(UnsafeSourceError):
            self._validate("https://docs.example.com/a")

    def test_empty_allowlist_refuses_everything(self) -> None:
        with pytest.raises(UnsafeSourceError):
            validate_url(
                "https://docs.example.com/a",
                allowed_schemes=self.SCHEMES,
                allowed_hosts=frozenset(),
                allow_private_network=True,
            )

    def test_error_message_does_not_reveal_the_resolved_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sources, "resolve_host", lambda host: ["169.254.169.254"])
        with pytest.raises(UnsafeSourceError) as raised:
            self._validate("https://docs.example.com/a")
        assert "169.254" not in str(raised.value)
