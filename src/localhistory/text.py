"""Small, dependency-free cleanup helpers for article text shown to users."""

import html
import re


# mwparserfromhell intentionally keeps the contents of image arguments when
# strip_code() removes the surrounding [[File:...]] node.  These are the
# option/caption remnants that should never be displayed as article prose.
IMAGE_REMNANT_LINE = re.compile(
    r"(?im)^\s*"
    r"(?:(?:\d+(?:\.\d+)?px|x\d+(?:\.\d+)?px)\s*\|\s*)?"
    r"(?:thumb|right|left|center|frame|frameless|border|upright(?:\s*=\s*[^|\n]+)?)"
    r"(?:\|.*)?\s*$"
)

# A few malformed/partially stripped image nodes retain only their dimensions.
IMAGE_DIMENSION_LINE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)?px|x\d+(?:\.\d+)?px)\s*\|.*$"
)

HTML_COMMENT = re.compile(r"(?is)<!--.*?-->")
WIKI_TEMPLATE = re.compile(r"(?is)\{\{.*?\}\}")


def clean_article_text(value: object) -> str:
    """Return readable plain text from the stored article text."""
    if value is None:
        return ""
    text = str(value)
    text = HTML_COMMENT.sub("", text)
    text = IMAGE_REMNANT_LINE.sub("", text)
    text = IMAGE_DIMENSION_LINE.sub("", text)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<ref\b[^>]*/\s*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    # Citation and metadata templates are not useful in the readable article
    # text.  This also removes common leftovers such as {{cite web|...}}.
    text = WIKI_TEMPLATE.sub("", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"(?m)^\s*[|!][^\n]*$", "", text)
    text = re.sub(r"(?m)^\s*\{\|.*$|^\s*\|\}.*$", "", text)
    text = re.sub(r"(?m)^\s*={2,6}\s*(.*?)\s*={2,6}\s*$", r"\1", text)
    text = re.sub(r"(?m)^\s*[*#;:]\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def article_lead(first_paragraph: object, full_text: object) -> str:
    """Return a usable lead, falling back when the stored lead is markup-only."""
    lead = clean_article_text(first_paragraph)
    if lead:
        return lead
    cleaned_full_text = clean_article_text(full_text)
    return cleaned_full_text.split("\n\n", 1)[0].strip()
