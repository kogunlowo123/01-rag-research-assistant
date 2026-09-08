"""Document parsers.

Each parser turns raw bytes into a :class:`~rag_assistant.domain.models.ParsedDocument`:
plain text, sanitised metadata, and — where the format carries it — a map from
character offset to page number so citations can name a page.

Parsers are the first code to touch attacker-controlled bytes, so they are
written defensively:

* HTML is parsed with the standard library rather than a heavyweight
  dependency, and ``script``, ``style``, ``noscript`` and HTML comments are
  discarded rather than flattened into the text. Flattening them is how a
  hidden ``<div style="display:none">`` instruction reaches a model.
* PDF metadata is bounded and normalised; a ``/Title`` field can otherwise carry
  kilobytes of attacker text straight into a prompt.
* JSON is walked with an explicit depth and node budget so a deeply nested
  document cannot exhaust the stack.
"""

from __future__ import annotations

import io
import json
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Final

from rag_assistant.domain.models import ParsedDocument, redact_metadata
from rag_assistant.errors import IngestionError
from rag_assistant.security.normalization import strip_control_characters

_MAX_JSON_DEPTH: Final[int] = 12
_MAX_JSON_NODES: Final[int] = 50_000
_BLANK_RUN: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
_TRAILING_SPACE: Final[re.Pattern[str]] = re.compile(r"[ \t]+\n")

#: Elements whose text content is never part of the readable document.
_SKIPPED_ELEMENTS: Final[frozenset[str]] = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "object"}
)
#: Elements that imply a line break in the extracted text.
_BLOCK_ELEMENTS: Final[frozenset[str]] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "dd",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    }
)


def _tidy(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    cleaned = strip_control_characters(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _TRAILING_SPACE.sub("\n", cleaned)
    return _BLANK_RUN.sub("\n\n", cleaned).strip()


class _TextExtractor(HTMLParser):
    """Extracts readable text and the document title from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str = ""
        self.meta: dict[str, str] = {}
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track skipped regions, headings, breaks and metadata."""
        if tag in _SKIPPED_ELEMENTS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attributes = {key.lower(): (value or "") for key, value in attrs}
            name = attributes.get("name") or attributes.get("property")
            if name and (content := attributes.get("content")):
                self.meta[name.lower()] = content
        if tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n" + "#" * int(tag[1]) + " ")

    def handle_endtag(self, tag: str) -> None:
        """Leave skipped regions and close block elements."""
        if tag in _SKIPPED_ELEMENTS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect text outside skipped regions."""
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        self.parts.append(data)

    def handle_comment(self, data: str) -> None:
        """Discard comments. Hidden instructions are commonly placed there."""
        return

    def error(self, message: str) -> None:  # pragma: no cover - removed in Python 3.10+
        """Tolerate malformed markup; required by the legacy ParserBase interface."""
        return


def parse_html(payload: bytes) -> ParsedDocument:
    """Extract readable text, title and meta tags from an HTML document."""
    extractor = _TextExtractor()
    try:
        extractor.feed(payload.decode("utf-8", errors="replace"))
        extractor.close()
    except (ValueError, AssertionError) as exc:
        raise IngestionError("the HTML document could not be parsed") from exc

    metadata: dict[str, Any] = dict(extractor.meta)
    if extractor.title.strip():
        metadata["title"] = extractor.title.strip()
    return ParsedDocument(
        text=_tidy(unescape("".join(extractor.parts))),
        media_type="text/html",
        metadata=redact_metadata(metadata),
    )


def parse_pdf(payload: bytes) -> ParsedDocument:
    """Extract text and page offsets from a PDF."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - pypdf is a hard dependency
        raise IngestionError("PDF support is unavailable") from exc

    try:
        reader = PdfReader(io.BytesIO(payload), strict=False)
        if reader.is_encrypted:
            # An empty-password decrypt succeeds for PDFs that are "encrypted"
            # only to set permissions, which is common for published papers.
            try:
                reader.decrypt("")
            except (PdfReadError, NotImplementedError) as exc:
                raise IngestionError("the PDF is password protected") from exc
        pages = reader.pages
        chunks: list[str] = []
        offsets: list[tuple[int, int]] = []
        position = 0
        for number, page in enumerate(pages, start=1):
            text = page.extract_text() or ""
            offsets.append((position, number))
            chunks.append(text)
            position += len(text) + 2
        raw_metadata = dict(reader.metadata or {})
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("the PDF could not be parsed") from exc

    body = _tidy("\n\n".join(chunks))
    if not body:
        raise IngestionError(
            "no extractable text was found in the PDF; scanned documents require OCR, "
            "which this service does not perform"
        )

    metadata = redact_metadata({str(k).lstrip("/"): v for k, v in raw_metadata.items()})
    metadata["page_count"] = str(len(chunks))
    return ParsedDocument(
        text=body,
        media_type="application/pdf",
        metadata=metadata,
        page_offsets=tuple(offsets),
    )


def _flatten_json(node: Any, path: str, out: list[str], depth: int, budget: list[int]) -> None:
    if depth > _MAX_JSON_DEPTH or budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_json(value, f"{path}.{key}" if path else str(key), out, depth + 1, budget)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _flatten_json(value, f"{path}[{index}]", out, depth + 1, budget)
    elif node is not None:
        out.append(f"{path}: {node}" if path else str(node))


def _exceeds_depth(node: Any, limit: int, depth: int = 0) -> bool:
    """Whether any value in ``node`` lies deeper than ``limit`` levels."""
    if depth > limit:
        return True
    if isinstance(node, dict):
        return any(_exceeds_depth(value, limit, depth + 1) for value in node.values())
    if isinstance(node, list):
        return any(_exceeds_depth(value, limit, depth + 1) for value in node)
    return False


def parse_json(payload: bytes) -> ParsedDocument:
    """Flatten a JSON document into ``path: value`` lines.

    Retrieval over JSON works far better on flattened paths than on pretty
    printed braces: the path carries the semantics that a brace does not.
    """
    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise IngestionError("the JSON document could not be parsed") from exc

    lines: list[str] = []
    _flatten_json(data, "", lines, 0, [_MAX_JSON_NODES])
    if not lines:
        # Distinguish "nothing to extract" from "structure too deep to walk":
        # the two call for different responses from whoever produced the file.
        if _exceeds_depth(data, _MAX_JSON_DEPTH):
            raise IngestionError(
                "the JSON document nests deeper than the supported limit of "
                f"{_MAX_JSON_DEPTH} levels"
            )
        raise IngestionError("the JSON document contained no extractable values")
    return ParsedDocument(text=_tidy("\n".join(lines)), media_type="application/json")


def parse_text(payload: bytes, media_type: str = "text/plain") -> ParsedDocument:
    """Decode a plain text or Markdown document."""
    text = _tidy(payload.decode("utf-8", errors="replace"))
    if not text:
        raise IngestionError("the document contained no text")

    metadata: dict[str, Any] = {}
    if media_type == "text/markdown":
        for line in text.splitlines():
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break
    return ParsedDocument(text=text, media_type=media_type, metadata=redact_metadata(metadata))


#: Media type to parser. Adding a format means adding one entry here and one
#: allowlist entry in configuration; no dispatch logic changes.
PARSERS: Final[dict[str, Any]] = {
    "text/plain": lambda payload: parse_text(payload, "text/plain"),
    "text/markdown": lambda payload: parse_text(payload, "text/markdown"),
    "text/html": parse_html,
    "application/pdf": parse_pdf,
    "application/json": parse_json,
}


def parse(payload: bytes, media_type: str) -> ParsedDocument:
    """Parse ``payload`` according to its media type."""
    parser = PARSERS.get(media_type)
    if parser is None:
        raise IngestionError(
            "no parser is registered for this media type",
            detail={"media_type": media_type},
        )
    parsed: ParsedDocument = parser(payload)
    return parsed


__all__ = ["PARSERS", "parse", "parse_html", "parse_json", "parse_pdf", "parse_text"]
