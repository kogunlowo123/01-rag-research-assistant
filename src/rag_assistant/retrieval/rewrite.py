"""Query rewriting.

Follow-up questions are the common failure case in conversational retrieval:
"what about refunds?" retrieves nothing useful because the subject lives in the
previous turn. Rewriting resolves the reference before retrieval runs.

This is deliberately deterministic rather than model-driven. A model call here
would add latency to every query, would introduce a second place the system can
hallucinate, and would make retrieval regression tests non-reproducible. The
rules below cover the cases that actually occur:

* **Reference expansion** — a short query containing a dangling pronoun or a
  bare "what about X" is expanded with the noun phrases of the previous turn.
* **Acronym pairing** — ``ERR_4417`` and ``error 4417`` are both emitted so the
  sparse retriever can match either surface form.
* **Decomposition** — a query joined by "and" that asks two things produces one
  sub-query per clause, and the retriever unions their results.

Every rewrite is additive: the original query is always retrieved for as well,
so a bad rewrite can only add noise, never remove the correct result.
"""

from __future__ import annotations

import re
from typing import Final

from rag_assistant.indexing.text import STOPWORDS, terms

#: Pronouns and determiners whose referent is outside the current query.
_DANGLING: Final[frozenset[str]] = frozenset(
    {"it", "its", "this", "that", "these", "those", "they", "them", "their", "one", "ones"}
)
_FOLLOWUP_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:and\s+)?(?:what|how)\s+about\b|^\s*(?:and|also|then)\b", re.IGNORECASE
)
_CONJUNCTION_SPLIT: Final[re.Pattern[str]] = re.compile(
    r"\s+(?:and|as\s+well\s+as)\s+(?=(?:what|how|why|when|where|who|which|does|do|is|are|can)\b)",
    re.IGNORECASE,
)
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"\b([A-Za-z]{2,10})[_-]?(\d{2,8})\b")
_ACRONYM: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{2,8})\b")

_MAX_REWRITES: Final[int] = 4
_SHORT_QUERY_TERMS: Final[int] = 6


def _keyphrases(text: str, limit: int = 6) -> list[str]:
    """Content words from a previous turn, in order of appearance."""
    seen: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text):
        lowered = word.lower()
        if lowered in STOPWORDS or lowered in _DANGLING or lowered in seen:
            continue
        seen.append(lowered)
        if len(seen) >= limit:
            break
    return seen


def needs_context(query: str) -> bool:
    """Whether the query cannot stand alone without the conversation."""
    tokens = re.findall(r"[a-z0-9']+", query.lower())
    if not tokens:
        return False
    if _FOLLOWUP_PREFIX.search(query):
        return True
    has_dangling = any(token in _DANGLING for token in tokens)
    return has_dangling and len(terms(query)) <= _SHORT_QUERY_TERMS


def expand_identifiers(query: str) -> list[str]:
    """Emit alternative surface forms for identifiers and acronyms.

    ``ERR_4417`` is indexed as the terms ``err`` and ``4417``; a user typing
    ``error 4417`` produces ``error`` and ``4417``. Emitting both forms lets the
    sparse retriever bridge them without a synonym dictionary.
    """
    variants: list[str] = []
    for match in _IDENTIFIER.finditer(query):
        prefix, digits = match.group(1), match.group(2)
        joined = f"{prefix}{digits}"
        spaced = f"{prefix} {digits}"
        variants.extend(
            variant for variant in (joined, spaced) if variant.lower() != match.group(0).lower()
        )
    for match in _ACRONYM.finditer(query):
        letters = match.group(1)
        spaced = " ".join(letters)
        if spaced not in variants:
            variants.append(spaced)
    return variants


def decompose(query: str) -> list[str]:
    """Split a compound question into its clauses."""
    parts = [part.strip(" ?.") for part in _CONJUNCTION_SPLIT.split(query) if part.strip(" ?.")]
    return parts if len(parts) > 1 else []


def rewrite(query: str, *, history: str = "", enabled: bool = True) -> list[str]:
    """Return the queries to retrieve for, original first.

    The result is always non-empty and always starts with the original query.
    """
    original = query.strip()
    if not enabled or not original:
        return [original] if original else []

    rewrites: list[str] = [original]

    if history and needs_context(original):
        phrases = _keyphrases(history)
        if phrases:
            rewrites.append(f"{original} {' '.join(phrases)}".strip())

    rewrites.extend(f"{original} {variant}" for variant in expand_identifiers(original))
    rewrites.extend(decompose(original))

    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in rewrites:
        key = " ".join(terms(candidate)) or candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
        if len(deduplicated) >= _MAX_REWRITES:
            break
    return deduplicated


__all__ = ["decompose", "expand_identifiers", "needs_context", "rewrite"]
