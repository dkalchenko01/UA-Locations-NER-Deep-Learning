"""Postprocessing of predicted location strings for UA Locations NER.

This module does NOT alter BIO tags, offsets, model weights, or the official
competition metric. Use the same version for validation and Kaggle inference
when comparing a postprocessing strategy.

Versions (historical notebook naming):
    raw              No changes (original BiLSTM baseline).
    strip            Strip surrounding whitespace and remove empty predictions.
    clean_v2         Strip + remove Unicode 'So' characters (includes many emoji).
    clean_v3         clean_v2 + discard predictions without a letter or digit.
    remove_non_alnum Teammate's original filter, applied without prior cleaning.

Note: 'So' is a Unicode category, NOT an exhaustive emoji detector. It also
includes non-emoji symbols and excludes some components of emoji sequences.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Sequence

POSTPROCESSING_VERSIONS = (
    "raw",
    "strip",
    "clean_v2",
    "clean_v3",
    "remove_non_alnum",
)


def _validated_locations(locations: Sequence[str]) -> list[str]:
    """Return a copy, rejecting a single string and non-string list entries."""
    if isinstance(locations, (str, bytes)):
        raise TypeError("Expected a sequence of location strings, not a string")
    result = list(locations)
    if not all(isinstance(location, str) for location in result):
        raise TypeError("Each location must be a string")
    return result


def clean_locations(locations: Sequence[str]) -> list[str]:
    """Original notebook `strip` strategy."""
    cleaned = [location.strip() for location in _validated_locations(locations)]
    return [location for location in cleaned if location]


def clean_locations_v2(locations: Sequence[str]) -> list[str]:
    """Original notebook `clean_v2`: drop category So, strip, remove empties."""
    cleaned = []
    for location in _validated_locations(locations):
        location = "".join(
            char for char in location
            if unicodedata.category(char) != "So"
        ).strip()
        if location:
            cleaned.append(location)
    return cleaned


def remove_non_alnum(locations: Sequence[str]) -> list[str]:
    """Keep only strings containing at least one alphanumeric character.

    Does not modify the retained strings (matching the teammate's function).
    """
    return [
        location for location in _validated_locations(locations)
        if any(char.isalnum() for char in location)
    ]


def clean_locations_v3(locations: Sequence[str]) -> list[str]:
    """Original notebook `clean_v3`: clean_v2 then remove_non_alnum."""
    return remove_non_alnum(clean_locations_v2(locations))


def postprocess_locations(
    locations: Sequence[str],
    version: str = "raw",
) -> list[str]:
    """Apply an explicitly chosen postprocessing version to location strings."""
    if version == "raw":
        return _validated_locations(locations)
    if version == "strip":
        return clean_locations(locations)
    if version == "clean_v2":
        return clean_locations_v2(locations)
    if version == "clean_v3":
        return clean_locations_v3(locations)
    if version == "remove_non_alnum":
        return remove_non_alnum(locations)
    raise ValueError(
        f"Unknown postprocessing version: {version!r}. "
        f"Choose one of {POSTPROCESSING_VERSIONS}."
    )


def parse_locations(value: str | Sequence[str]) -> list[str]:
    """Read a locations list or a JSON-encoded locations list (e.g. CSV cell)."""
    if isinstance(value, str):
        value = json.loads(value)
    return _validated_locations(value)


def serialize_locations(
    locations: str | Sequence[str],
    version: str = "raw",
) -> str:
    """Serialize a Kaggle CSV cell as UTF-8-friendly JSON after processing."""
    return json.dumps(
        postprocess_locations(parse_locations(locations), version=version),
        ensure_ascii=False,
    )
