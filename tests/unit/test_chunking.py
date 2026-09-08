"""Structure-aware chunking."""

from __future__ import annotations

import pytest

from rag_assistant.domain.models import Chunk, ParsedDocument
from rag_assistant.ingestion.chunking import Chunker, estimate_tokens, split_sections

pytestmark = pytest.mark.unit

STRUCTURED = """# Handbook

Intro paragraph before any subheading.

## Refunds

Customers may request a refund within 30 days. Proof of purchase is required.

### Digital goods

Digital goods are refundable within 14 days.

## Shipping

Standard shipping takes 3 to 7 days.
"""


class TestSplitSections:
    def test_headingless_document_yields_one_unnamed_section(self) -> None:
        sections = split_sections("Just a paragraph of text.")
        assert len(sections) == 1
        assert sections[0].path == ()

    def test_breadcrumb_nests_and_unwinds_with_heading_level(self) -> None:
        paths = [section.path for section in split_sections(STRUCTURED)]
        assert ("Handbook",) in paths
        assert ("Handbook", "Refunds") in paths
        assert ("Handbook", "Refunds", "Digital goods") in paths
        assert ("Handbook", "Shipping") in paths

    def test_preamble_before_the_first_heading_is_kept(self) -> None:
        sections = split_sections("Preface text.\n\n# Title\n\nBody.")
        assert sections[0].path == ()
        assert "Preface" in sections[0].text

    def test_empty_sections_are_dropped(self) -> None:
        sections = split_sections("# A\n\n## B\n\nOnly B has content.")
        assert all(section.text.strip() for section in sections)


class TestChunker:
    def _chunk(self, text: str, **kwargs: int) -> list[Chunk]:
        chunker = Chunker(**{"target_tokens": 60, "overlap_tokens": 12, **kwargs})
        return chunker.chunk(
            ParsedDocument(text=text, media_type="text/markdown"),
            document_id="doc_1",
            tenant_id="acme",
        )

    def test_chunks_carry_their_section_path(self) -> None:
        chunks = self._chunk(STRUCTURED)
        assert any(chunk.section_path == ("Handbook", "Refunds") for chunk in chunks)

    def test_chunk_text_excludes_the_breadcrumb_but_indexing_text_includes_it(self) -> None:
        chunks = self._chunk(STRUCTURED)
        refund = next(c for c in chunks if c.section_path == ("Handbook", "Refunds"))
        assert not refund.text.startswith("Handbook")
        assert refund.indexing_text.startswith("Handbook > Refunds")
        assert refund.text in refund.indexing_text

    def test_ordinals_are_contiguous_and_ordered(self) -> None:
        chunks = self._chunk(STRUCTURED)
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))

    def test_a_paragraph_larger_than_the_window_is_split_not_dropped(self) -> None:
        chunks = self._chunk("word " * 800, target_tokens=40, overlap_tokens=8)
        assert len(chunks) > 1
        assert sum(chunk.text.count("word") for chunk in chunks) >= 800

    def test_consecutive_windows_overlap(self) -> None:
        text = " ".join(f"Sentence number {i} states an independent fact." for i in range(40))
        chunks = self._chunk(text, target_tokens=40, overlap_tokens=16)
        assert len(chunks) > 2
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert first_words & second_words

    def test_zero_overlap_is_permitted(self) -> None:
        chunks = self._chunk(STRUCTURED, target_tokens=60, overlap_tokens=0)
        assert chunks

    def test_overlap_at_or_above_target_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap_tokens"):
            Chunker(target_tokens=40, overlap_tokens=40)

    def test_max_chunks_is_enforced(self) -> None:
        chunks = self._chunk("word " * 5000, target_tokens=32, overlap_tokens=4, max_chunks=3)
        assert len(chunks) <= 3

    def test_empty_document_produces_no_chunks(self) -> None:
        assert self._chunk("   ") == []

    def test_page_is_attributed_when_the_parser_supplied_offsets(self) -> None:
        body = "First page content here.\n\n" + "Second page content here.\n\n"
        parsed = ParsedDocument(
            text=body,
            media_type="application/pdf",
            page_offsets=((0, 1), (len(body) // 2, 2)),
        )
        chunks = Chunker(target_tokens=10, overlap_tokens=2).chunk(
            parsed, document_id="d", tenant_id="t"
        )
        assert {chunk.page for chunk in chunks} <= {1, 2}
        assert any(chunk.page == 1 for chunk in chunks)

    def test_locator_reflects_page_and_section(self) -> None:
        chunks = self._chunk(STRUCTURED)
        refund = next(c for c in chunks if c.section_path == ("Handbook", "Refunds"))
        assert "Handbook > Refunds" in refund.locator
        assert refund.locator.endswith(f"#{refund.ordinal}")


def test_token_estimate_is_positive_and_scales_with_length() -> None:
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
