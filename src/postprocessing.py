"""Inference-only location cleanup and word-boundary experiments."""

import unicodedata

from src.metrics import hard_processing
from src.preprocessing import clean_locations_v3

EDGE_CHARS = ' \t\r\n,;:!?…|•"«»“”„'
CONNECTORS = "'’ʼ-‐‑"

VARIANTS = {
    "v3": {},
    "v4_empty": {"empty": True},
    "v4_edges": {"trim": True},
    "v4_words": {"expand": True},
    "v4_combined": {"expand": True, "trim": True, "empty": True},
}


def word_char(text, i):
    ch = text[i]
    return (
        ch.isalnum()
        or unicodedata.category(ch).startswith("M")
        or (
            ch in CONNECTORS
            and 0 < i < len(text) - 1
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        )
    )


def restore_word(text, start, end):
    if start > 0 and start < end:
        if word_char(text, start - 1) and word_char(text, start):
            while start > 0 and word_char(text, start - 1):
                start -= 1

    if 0 < end < len(text):
        if word_char(text, end - 1) and word_char(text, end):
            while end < len(text) and word_char(text, end):
                end += 1

    return start, end


def postprocess(text, spans, expand=False, trim=False, empty=False):
    locations = []
    for start, end in spans:
        if expand and sum(ch.isalpha() for ch in text[start:end]) > 4:
            start, end = restore_word(text, start, end)
        location = text[start:end]
        if trim:
            location = location.strip(EDGE_CHARS)
        locations.append(location)

    locations = clean_locations_v3(locations)
    if empty:
        locations = [location for location in locations if hard_processing(location) != ""]
    return locations
