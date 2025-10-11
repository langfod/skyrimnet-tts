"""Minimal text cleaners for XTTS - only functions actually used"""

import re
from unicodedata import normalize

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(_whitespace_re, " ", text).strip()


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""
    text = normalize("NFC", text)
    return text
