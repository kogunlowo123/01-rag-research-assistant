"""Text normalisation applied before any security decision is made.

Detection that runs on raw bytes is trivially defeated. ``i g n o r e``,
``ｉｇｎｏｒｅ``, ``ig\u200bnore`` and ``ignore`` are the same instruction to a
language model and four different strings to a regular expression, so every
detector in this package runs against normalised text.

The normalised form is used *for detection only*. The text a user sees, and the
text stored as the document body, keeps its original characters — normalisation
would otherwise silently corrupt legitimate documents in non-Latin scripts.
:func:`normalize` therefore returns the mapping back to original offsets so a
finding can be reported against, and redacted from, the real text.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

#: Characters with no visible rendering that can split a keyword in two.
#: Written as explicit code points: pasting the characters themselves into
#: source makes the module unreviewable and survives copy-paste poorly.
_INVISIBLE: Final[frozenset[str]] = frozenset(
    chr(code)
    for code in (
        0x00AD,  # soft hyphen
        0x034F,  # combining grapheme joiner
        0x061C,  # arabic letter mark
        0x115F,
        0x1160,  # hangul filler
        0x17B4,
        0x17B5,  # khmer inherent vowels
        0x180E,  # mongolian vowel separator
        0x200B,
        0x200C,
        0x200D,
        0x200E,
        0x200F,  # zero width / directional marks
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,  # bidi embedding and override
        0x2060,
        0x2061,
        0x2062,
        0x2063,
        0x2064,  # word joiner, invisible operators
        0x2066,
        0x2067,
        0x2068,
        0x2069,  # bidi isolates
        0x206A,
        0x206B,
        0x206C,
        0x206D,
        0x206E,
        0x206F,  # deprecated format chars
        0x3164,  # hangul filler
        0xFEFF,  # zero width no-break space
        0xFFA0,  # halfwidth hangul filler
    )
)

#: Unicode Tags block. Renders as nothing in most clients but is tokenised by
#: many models, which makes it a channel for hidden instructions.
_TAG_BLOCK: Final[range] = range(0xE0000, 0xE0080)

#: Horizontal whitespace that collapses to a single space. Newlines are kept
#: because paragraph structure is a legitimate signal for chunking.
_HORIZONTAL_SPACE: Final[frozenset[str]] = frozenset(
    chr(code)
    for code in (
        0x0009,  # tab
        0x000B,  # vertical tab
        0x000C,  # form feed
        0x0020,  # space
        0x00A0,  # no-break space
        0x1680,  # ogham space mark
        *range(0x2000, 0x200B),  # en quad through hair space
        0x205F,  # medium mathematical space
        0x3000,  # ideographic space
    )
)

#: Confusable characters mapped to their Latin lookalike. NFKC does not fold
#: these, and they are the standard way to write "ignore" so a keyword filter
#: does not see it.
_CONFUSABLES: Final[dict[str, str]] = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "і": "i",
    "ј": "j",
    "һ": "h",
    "Α": "a",
    "Β": "b",
    "Ε": "e",
    "Ζ": "z",
    "Η": "h",
    "Ι": "i",
    "Κ": "k",
    "Μ": "m",
    "Ν": "n",
    "Ο": "o",
    "Ρ": "p",
    "Τ": "t",
    "Χ": "x",
    "ο": "o",
    "α": "a",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Normalised text plus the offset map back to the original string."""

    text: str
    #: ``offsets[i]`` is the index in the original string that produced
    #: ``text[i]``. Lets a match on normalised text be redacted from the original.
    offsets: tuple[int, ...]
    #: Counts of each transformation, used as a risk signal in its own right:
    #: a passage containing 40 zero-width characters is suspicious regardless of
    #: what it says.
    stats: dict[str, int]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a span in the normalised text back to the original string."""
        if not self.offsets:
            return (0, 0)
        clamped_start = max(0, min(start, len(self.offsets) - 1))
        clamped_end = max(0, min(end, len(self.offsets)))
        origin_start = self.offsets[clamped_start]
        origin_end = self.offsets[clamped_end - 1] + 1 if clamped_end > 0 else origin_start
        return (origin_start, max(origin_start, origin_end))


def normalize(text: str) -> NormalizedText:
    """Fold text into a canonical lowercase form suitable for detection."""
    stats = {"invisible_removed": 0, "tag_chars_removed": 0, "confusables_folded": 0}

    stripped_chars: list[str] = []
    stripped_offsets: list[int] = []
    for index, char in enumerate(text):
        code = ord(char)
        if char in _INVISIBLE:
            stats["invisible_removed"] += 1
            continue
        if code in _TAG_BLOCK:
            stats["tag_chars_removed"] += 1
            continue
        if unicodedata.category(char) in {"Cc", "Cf"} and char not in "\n\r\t":
            stats["invisible_removed"] += 1
            continue
        folded = _CONFUSABLES.get(char)
        if folded is not None:
            stats["confusables_folded"] += 1
            stripped_chars.append(folded)
            stripped_offsets.append(index)
            continue
        stripped_chars.append(char)
        stripped_offsets.append(index)

    # NFKC is applied per character so the offset map survives it. A character
    # that decomposes into several keeps pointing at its single origin index.
    expanded_chars: list[str] = []
    expanded_offsets: list[int] = []
    for char, origin in zip(stripped_chars, stripped_offsets, strict=True):
        for produced in unicodedata.normalize("NFKC", char).lower():
            expanded_chars.append(produced)
            expanded_offsets.append(origin)

    # Collapse whitespace runs to a single space, keeping the first origin.
    collapsed_chars: list[str] = []
    collapsed_offsets: list[int] = []
    previous_was_space = False
    for char, origin in zip(expanded_chars, expanded_offsets, strict=True):
        is_space = char in _HORIZONTAL_SPACE
        if is_space:
            if previous_was_space:
                continue
            collapsed_chars.append(" ")
            collapsed_offsets.append(origin)
            previous_was_space = True
            continue
        previous_was_space = False
        collapsed_chars.append(char)
        collapsed_offsets.append(origin)

    return NormalizedText(
        text="".join(collapsed_chars),
        offsets=tuple(collapsed_offsets),
        stats=stats,
    )


def strip_control_characters(text: str) -> str:
    """Remove control and invisible characters from text destined for storage.

    Unlike :func:`normalize` this preserves case, script and word forms; it only
    removes characters that have no legitimate place in a document body and that
    exist in an attacker's payload precisely because they are invisible.
    """
    keep_always = frozenset(("\n", "\r", "\t"))

    def keep(char: str) -> bool:
        if char in keep_always:
            return True
        if char in _INVISIBLE or ord(char) in _TAG_BLOCK:
            return False
        return unicodedata.category(char) not in {"Cc", "Cf"}

    return "".join(char for char in text if keep(char))


__all__ = ["NormalizedText", "normalize", "strip_control_characters"]
