"""Aggregation + slicing of CityNav metrics for the stepwise runner.

Wraps benchmarks.citynav.eval so the runner only imports from one place,
and adds the car-vs-building subset SR that is the headline ablation for
the runtime-graph contribution.
"""

from __future__ import annotations

import re

from benchmarks.citynav.eval import aggregate as _aggregate


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


def aggregate(per_episode: list[dict]) -> dict:
    """Macro-average + per-subset aggregates."""
    overall = _aggregate(per_episode)
    buckets: dict[str, list[dict]] = {"car": [], "building": [], "other": []}
    for row in per_episode:
        bucket = row.get("referent_bucket", "other")
        buckets.setdefault(bucket, []).append(row)
    by_referent: dict[str, dict] = {}
    for name, rows in buckets.items():
        if rows:
            sub = _aggregate(rows)
            by_referent[name] = sub
    overall["by_referent"] = by_referent
    return overall
