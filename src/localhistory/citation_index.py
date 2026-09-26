"""Shared citation-sentence detection used by indexing and search fallback."""

import re
from collections.abc import Iterator


CITATION_HEURISTIC_VERSION = "1"
CITATION_MARKERS = (
    r"\b(?:press|publisher|publishing|university press|journal|"
    r"vol\.?|volume|pp?\.?|pages|isbn|doi|retrieved|accessed)\b"
)
CITATION_MARKER_PATTERN = re.compile(CITATION_MARKERS, re.IGNORECASE)
CITATION_YEAR_PATTERN = re.compile(r"\b(?:18|19|20)\d{2}\b")
BIBLIOGRAPHIC_SHAPE_PATTERN = re.compile(
    r":[^.!?]{0,120},\s*(?:18|19|20)\d{2}\b"
)
NEXT_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def is_citation_sentence(sentence: str) -> bool:
    """Return whether a sentence has enough signals to resemble a citation."""
    has_bibliographic_shape = bool(
        BIBLIOGRAPHIC_SHAPE_PATTERN.search(sentence)
    )
    if has_bibliographic_shape:
        return True
    if not CITATION_YEAR_PATTERN.search(sentence):
        return False

    marker_count = 0
    for _ in CITATION_MARKER_PATTERN.finditer(sentence):
        marker_count += 1
        if marker_count >= 2:
            return True
    return False


def previous_sentence_start(text: str, position: int) -> int:
    """Find the preceding sentence boundary without scanning the whole text."""
    cursor = position - 1
    while cursor >= 0:
        if text[cursor] == "\n":
            return cursor + 1
        if text[cursor].isspace():
            boundary_end = cursor + 1
            while cursor >= 0 and text[cursor].isspace():
                if text[cursor] == "\n":
                    return cursor + 1
                cursor -= 1
            if cursor >= 0 and text[cursor] in ".!?":
                return boundary_end
            continue
        cursor -= 1
    return 0


def iter_citation_sentences(text: object) -> Iterator[str]:
    """Yield citation-like sentences by jumping directly to year mentions."""
    if text is None:
        return
    text = str(text)
    processed_until = 0
    for match in CITATION_YEAR_PATTERN.finditer(text):
        if match.start() < processed_until:
            continue

        sentence_start = previous_sentence_start(text, match.start())
        next_boundary = NEXT_SENTENCE_BOUNDARY_PATTERN.search(text, match.end())
        if next_boundary is None:
            sentence_end = len(text)
            processed_until = len(text)
        else:
            sentence_end = next_boundary.start()
            processed_until = next_boundary.end()
        sentence = text[sentence_start:sentence_end]
        if is_citation_sentence(sentence):
            yield sentence
