"""Normalisation must defeat the standard evasion techniques."""

from __future__ import annotations

import pytest

from rag_assistant.security.normalization import (
    normalize,
    strip_control_characters,
)

pytestmark = pytest.mark.unit

ZERO_WIDTH = "\u200b"
SOFT_HYPHEN = chr(0x00AD)
RTL_OVERRIDE = chr(0x202E)
TAG_A = "\U000e0041"


def test_zero_width_characters_are_removed_and_counted() -> None:
    result = normalize(f"ig{ZERO_WIDTH}no{ZERO_WIDTH}re me")
    assert result.text == "ignore me"
    assert result.stats["invisible_removed"] == 2


def test_soft_hyphen_and_bidi_override_are_removed() -> None:
    result = normalize(f"dis{SOFT_HYPHEN}regard{RTL_OVERRIDE} this")
    assert result.text == "disregard this"
    assert result.stats["invisible_removed"] == 2


def test_unicode_tag_characters_are_removed_separately() -> None:
    result = normalize(f"hello{TAG_A}world")
    assert result.text == "helloworld"
    assert result.stats["tag_chars_removed"] == 1


def test_cyrillic_homoglyphs_fold_to_latin() -> None:
    # "ignоre" with a Cyrillic small letter o.
    result = normalize("ignоre")
    assert result.text == "ignore"
    assert result.stats["confusables_folded"] == 1


def test_fullwidth_characters_fold_via_nfkc() -> None:
    assert normalize("ＩＧＮＯＲＥ").text == "ignore"


def test_newlines_survive_but_horizontal_runs_collapse() -> None:
    result = normalize("first   line\n\nsecond\tline")
    assert result.text == "first line\n\nsecond line"


def test_offsets_map_a_normalised_span_back_to_the_original() -> None:
    original = f"please ig{ZERO_WIDTH}nore the rule"
    result = normalize(original)
    start = result.text.index("ignore")
    end = start + len("ignore")
    origin_start, origin_end = result.original_span(start, end)
    # The original span is one character longer because of the zero-width joiner.
    assert original[origin_start:origin_end].replace(ZERO_WIDTH, "") == "ignore"


def test_original_span_on_empty_text_is_safe() -> None:
    assert normalize("").original_span(0, 5) == (0, 0)


def test_strip_control_characters_preserves_case_and_script() -> None:
    assert strip_control_characters(f"Café{ZERO_WIDTH}\nBar") == "Café\nBar"


def test_strip_control_characters_removes_tag_block() -> None:
    assert strip_control_characters(f"a{TAG_A}b") == "ab"


@pytest.mark.parametrize(
    "text",
    [
        "The transformer architecture uses multi-head attention.",
        "Ignore outliers above three sigma when fitting.",
        "Section 4 lists the safety guidelines for shutdown.",
    ],
)
def test_ordinary_prose_is_unchanged_apart_from_case(text: str) -> None:
    assert normalize(text).text == text.lower()
