"""Term extraction shared by indexing and querying.

Indexing and search must tokenise identically or the inverted index silently
stops matching, so both call :func:`terms` and neither has its own copy of the
rules. A term is lowercased, stripped of possessives, suffix-normalised by a
small deterministic stemmer, and dropped if it is a stopword or a single
character.

The stemmer is intentionally a light suffix stripper rather than a full Porter
implementation: it handles the plural and tense variation that dominates
retrieval misses, is trivially auditable, and does not turn "policies" into a
token no human would recognise in a diagnostics payload.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_MAX_TERM_LENGTH: Final[int] = 40
#: Single characters carry no retrieval signal and inflate the postings table.
_MIN_TERM_LENGTH: Final[int] = 2

STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)

_SUFFIXES: Final[tuple[tuple[str, str, int], ...]] = (
    # (suffix, replacement, minimum stem length required)
    ("ies", "y", 4),
    ("sses", "ss", 5),
    ("ches", "ch", 5),
    ("shes", "sh", 5),
    ("xes", "x", 4),
    ("ing", "", 5),
    ("edly", "", 6),
    ("ed", "", 4),
    ("ly", "", 4),
    ("ness", "", 5),
    ("ment", "", 6),
    ("tion", "te", 6),
    ("s", "", 3),
)


def stem(word: str) -> str:
    """Apply a single, deterministic suffix reduction.

    Only one rule fires per word. Cascading rules produce surprising stems and
    make the index harder to reason about during a retrieval incident.
    """
    for suffix, replacement, minimum in _SUFFIXES:
        if word.endswith(suffix) and len(word) >= minimum:
            candidate = word[: -len(suffix)] + replacement
            if len(candidate) >= _MIN_TERM_LENGTH:
                return candidate
    return word


def terms(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Extract the index terms of a piece of text, in order."""
    result: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        word = raw.removesuffix("'s").strip("-'")
        if len(word) < _MIN_TERM_LENGTH or len(word) > _MAX_TERM_LENGTH:
            continue
        if not keep_stopwords and word in STOPWORDS:
            continue
        result.append(stem(word))
    return result


def term_frequencies(text: str) -> dict[str, int]:
    """Term frequency map used to build inverted-index postings."""
    return dict(Counter(terms(text)))


__all__ = ["STOPWORDS", "stem", "term_frequencies", "terms"]
