"""The referent bucket a row carries (``metrics.referent_bucket``): car,
building or other, read off the referring expression."""

from __future__ import annotations

import re


_CAR_PAT = re.compile(
    r"\b(car|cars|vehicle|truck|van|bus|taxi|suv|sedan|lorry)\b",
    re.IGNORECASE,
)
_BUILDING_PAT = re.compile(
    r"\b(building|buildings|tower|church|cathedral|hall|museum|hotel|shop|store|library)\b",
    re.IGNORECASE,
)


def classify_referent(description: str) -> str:
    """Return 'car', 'building', or 'other' based on the referring expression."""
    if _CAR_PAT.search(description):
        return "car"
    if _BUILDING_PAT.search(description):
        return "building"
    return "other"
