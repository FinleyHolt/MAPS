"""The run context the mission is flown against.

``MapsCompiler`` carries what every episode in a run shares and nothing
else: the discovered reasoner endpoint (``reasoner_url`` /
``endpoint_spec`` / ``api_key``), the active sensor description, the
workdir artifacts land under, and ``_resolve_seed_base`` (the per-episode
seed). It compiles nothing. Constructed once per process by
``benchmarks.citynav.run_benchmark.dispatch`` and by
``benchmarks.flight.mission``, then handed to every ``SearchLoop``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from maps.llm.endpoint_spec import spec
from maps.search.parse_compile.perception_director import (
    WATCH_ROLES,
    canon_concept,
)


_NULL_CONCEPT_TOKENS = frozenset({"none", "null", "n/a", "na"})


def _extract_reasoner_concepts(plan: dict) -> list[tuple[str, str]]:
    """Role-tagged SAM concept specs for one reasoner.

    Reads ``sam_concepts`` from the given dict (a finalized record, or any
    dict carrying the key). Each entry is
    ``{"class": <phrase>, "role": "target"|"anchor"}``.
    Returns a list of ``(class_phrase, role)`` tuples. The class phrase is
    case-folded + whitespace-collapsed before being fed to SAM.

    Tolerates the bare ``str`` entry shape (treated as ``role="target"``)
    since the model emits both. Empty strings and placeholder
    tokens ``"none"`` / ``"null"`` / ``"n/a"`` are dropped. Entries
    are deduplicated by ``(class, role)``; first-occurrence order is
    preserved.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in plan.get("sam_concepts") or []:
        if isinstance(entry, dict):
            raw_class = entry.get("class")
            raw_role = entry.get("role") or "target"
        elif isinstance(entry, str):
            raw_class = entry
            raw_role = "target"
        else:
            continue
        if not isinstance(raw_class, str):
            continue
        s = canon_concept(raw_class)
        if not s or s in _NULL_CONCEPT_TOKENS:
            continue
        role = str(raw_role).strip().lower()
        if role not in WATCH_ROLES:
            role = "target"
        key = (s, role)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def concept_union(
    specs: list[tuple[str, str]],
) -> tuple[list[str], list[dict]]:
    """The SAM class union of ``(class, role)`` specs (as
    ``_extract_reasoner_concepts`` returns them): ``union`` is the classes in
    first-seen order, the set SAM segments (SAM reads no roles);
    ``union_specs`` is ``[{"class": c, "roles": [...]}]`` in the same order,
    each class's roles merged in first-seen order."""
    roles: dict[str, list[str]] = {}
    for cls, role in specs:
        merged = roles.setdefault(cls, [])
        if role not in merged:
            merged.append(role)
    return list(roles), [{"class": c, "roles": r} for c, r in roles.items()]


class MapsCompiler:
    """The run context: the discovered endpoint, the API key, the sensor
    description and the artifact workdir."""

    def __init__(
        self, sensor_description: str = "", workdir: Path | None = None,
    ) -> None:
        # The server-discovered endpoint authority (installed once at the
        # process entry point), threaded explicitly down the hot path into
        # the Reasoner: the model name, the window-derived budgets and the
        # endpoint set all come from the live server.
        self.endpoint_spec = spec()
        self.reasoner_url = self.endpoint_spec.base_url
        # "EMPTY" is the no-auth sentinel a local vLLM accepts.
        self.api_key = os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY")
        self.sensor_description = sensor_description or ""
        if workdir is None:
            env_workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip()
            workdir = Path(env_workdir) if env_workdir else None
        self.workdir: Path | None = workdir

    def _resolve_seed_base(self) -> int:
        """Resolve the base seed: ``SEARCH_SEED_BASE`` env if set, else a
        wall-clock nanosecond stamp. Per-attempt offsets (retry) and the
        per-reasoner ``+i`` fan-out are applied on top of this."""
        env_seed_base = os.environ.get("SEARCH_SEED_BASE")
        if env_seed_base is not None:
            return int(env_seed_base) & 0xFFFFFFFF
        return time.time_ns() & 0xFFFFFFFF

