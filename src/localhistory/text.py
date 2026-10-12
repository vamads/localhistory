"""Small, dependency-free cleanup helpers for article text shown to users."""

import html
import re


# mwparserfromhell intentionally keeps the contents of image arguments when
# strip_code() removes the surrounding [[File:...]] node.  These are the
# option/caption remnants that should never be displayed as article prose.
IMAGE_REMNANT_LINE = re.compile(
    r"(?im)^\s*(?:thumb|right|left|center|frame|frameless|border|upright(?:\s*=\s*[^|\n]+)?)(?:\|.*)?\s*$"
)


def clean_article_text(value: object) -> str:
    """Return readable plain text from the stored article text."""
    if value is None:
        return ""
    text = str(value)
    text = IMAGE_REMNANT_LINE.sub("", text)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<ref\b[^>]*/\s*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"'{2,5}", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
