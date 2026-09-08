"""Structure-aware chunking.

Fixed-width chunking is the most common reason a RAG system retrieves the right
document and the wrong passage: it cuts sentences in half, strips the heading
that gave a paragraph its meaning, and merges two unrelated sections into one
embedding.

This chunker works top-down instead:

1. Split on headings, keeping the heading breadcrumb for every span below it.
   The breadcrumb is prepended to the chunk text, so an embedding of "the limit
   is 30 days" also carries "Refunds > Eligibility".
2. Split each section into paragraphs, then sentences, and pack them into
   windows near the target size without crossing a sentence boundary.
3. Overlap consecutive windows by whole sentences so a fact that straddles a
   boundary appears intact in at least one chunk.
4. Split any single paragraph that is larger than the window on its own — a
   minified HTML body or a table dumped as one line — on word boundaries, since
   dropping it would silently lose content.

Every chunk records its ordinal so neighbours can be stitched back at retrieval
time, and its page so citations can name one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from rag_assistant.domain.models import Chunk, ParsedDocument

#: Characters per token. Empirically close for English across BPE tokenisers.
#: An estimate is deliberate: a real tokeniser would tie chunking to one model
#: family and make chunk boundaries change when the model changes.
_CHARS_PER_TOKEN: Final[float] = 4.0

_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_PARAGRAPH_SPLIT: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class Section:
    """A span of the document under one heading breadcrumb."""

    path: tuple[str, ...]
    text: str
    start_offset: int


def split_sections(text: str) -> list[Section]:
    """Split a document into sections keyed by their heading breadcrumb.

    Documents with no headings yield a single unnamed section, which keeps the
    downstream code free of special cases.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [Section(path=(), text=text, start_offset=0)]

    sections: list[Section] = []
    if preamble := text[: matches[0].start()].strip():
        sections.append(Section(path=(), text=preamble, start_offset=0))

    breadcrumb: list[str] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        del breadcrumb[level - 1 :]
        breadcrumb.append(title)

        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append(Section(path=tuple(breadcrumb), text=body, start_offset=body_start))
    return sections


def _split_units(section_text: str) -> list[str]:
    """Break a section into the smallest units that will not be split further."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(section_text):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        units.extend(part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip())
    return units


def _hard_split(unit: str, max_chars: int) -> list[str]:
    """Split an oversized unit on word boundaries."""
    words = _WHITESPACE.split(unit)
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > max_chars:
            parts.append(" ".join(current))
            current, length = [word], len(word)
            continue
        current.append(word)
        length += addition
    if current:
        parts.append(" ".join(current))
    return parts


class Chunker:
    """Packs a parsed document into overlapping, structure-aware chunks."""

    def __init__(
        self,
        *,
        target_tokens: int = 320,
        overlap_tokens: int = 64,
        max_chunks: int = 5000,
    ) -> None:
        """Configure window size, overlap and the per-document chunk ceiling."""
        if overlap_tokens >= target_tokens:
            msg = "overlap_tokens must be smaller than target_tokens"
            raise ValueError(msg)
        self._target_chars = int(target_tokens * _CHARS_PER_TOKEN)
        self._overlap_chars = int(overlap_tokens * _CHARS_PER_TOKEN)
        self._max_chunks = max_chunks

    def chunk(self, parsed: ParsedDocument, *, document_id: str, tenant_id: str) -> list[Chunk]:
        """Produce the ordered chunk list for a parsed document."""
        chunks: list[Chunk] = []
        ordinal = 0

        for section in split_sections(parsed.text):
            for window, offset in self._windows(section):
                if ordinal >= self._max_chunks:
                    return chunks
                # ``text`` is the passage as the document wrote it. The heading
                # breadcrumb is available through ``Chunk.indexing_text`` and is
                # what gets embedded; keeping it out of ``text`` is what keeps
                # citation quotes free of synthesised prefixes.
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        ordinal=ordinal,
                        text=window,
                        token_estimate=estimate_tokens(window) + len(section.path) * 4,
                        section_path=section.path,
                        page=parsed.page_for_offset(section.start_offset + offset),
                    )
                )
                ordinal += 1
        return chunks

    def _windows(self, section: Section) -> list[tuple[str, int]]:
        """Pack a section's units into overlapping windows.

        Returns each window with its character offset inside the section, which
        is what makes page attribution possible for multi-page sections.
        """
        units = _split_units(section.text)
        if not units:
            return []

        # Offsets are located by scanning forward, so a repeated sentence maps to
        # its own occurrence rather than always to the first one.
        offsets: list[int] = []
        cursor = 0
        for unit in units:
            found = section.text.find(unit, cursor)
            if found == -1:
                found = cursor
            offsets.append(found)
            cursor = found + len(unit)

        windows: list[tuple[str, int]] = []
        current: list[str] = []
        current_offset = offsets[0]
        length = 0
        index = 0

        while index < len(units):
            unit = units[index]
            if len(unit) > self._target_chars and not current:
                for part in _hard_split(unit, self._target_chars):
                    windows.append((part, offsets[index]))
                index += 1
                continue

            addition = len(unit) + (2 if current else 0)
            if current and length + addition > self._target_chars:
                windows.append((" ".join(current), current_offset))
                carry, carry_length = self._carry(current)
                current = carry
                length = carry_length
                current_offset = offsets[max(0, index - len(carry))]
                continue

            if not current:
                current_offset = offsets[index]
            current.append(unit)
            length += addition
            index += 1

        if current:
            windows.append((" ".join(current), current_offset))
        return windows

    def _carry(self, units: list[str]) -> tuple[list[str], int]:
        """Select trailing units to repeat at the start of the next window."""
        if self._overlap_chars <= 0:
            return [], 0
        carry: list[str] = []
        length = 0
        for unit in reversed(units):
            addition = len(unit) + (2 if carry else 0)
            if length + addition > self._overlap_chars:
                break
            carry.insert(0, unit)
            length += addition
        # Never carry the whole window: that would make no forward progress and
        # loop forever on a section of exactly one oversized sentence.
        if len(carry) >= len(units):
            carry = carry[1:]
            length = sum(len(unit) + 2 for unit in carry)
        return carry, length


__all__ = ["Chunker", "Section", "estimate_tokens", "split_sections"]
