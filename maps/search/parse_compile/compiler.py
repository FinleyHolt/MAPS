"""The run context the mission is flown against.

``MapsCompiler`` carries what every episode in a run shares and nothing
else: the discovered reasoner endpoint (``reasoner_url`` /
``endpoint_spec`` / ``api_key``), the active sensor description, the
workdir artifacts land under, the look gate, and ``_resolve_seed_base``
(the per-episode seed, derived from the instruction so a rerun repeats).

It compiles nothing. The name predates the mission cutover, when this
class ran a pre-flight compile; that pipeline is deleted and the class is
now a carrier. Constructed once per process by
``benchmarks.citynav.run_benchmark.dispatch`` and by
``benchmarks.flight.mission``, then handed to every ``SearchLoop``.

Artifact writes: when the constructor is given a ``workdir`` (or
``SEARCH_COMPILE_WORKDIR`` is set), per-turn artifacts land under it as
soon as they exist rather than buffering to end-of-turn, so a dashboard
reading the tree sees a run progress.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from maps.search.io.atomic import atomic_write_json
from maps.llm.endpoint_spec import spec
from maps.llm.net_meter import stage as net_meter_stage
from maps.search.analyze_scene.tools import (
    ToolContext,
)
from maps.search.parse_compile.concept_union import ConceptUnionResult
from maps.search.parse_compile.perception_director import WATCH_ROLES
from maps.search.protocol import CompilerInput

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept union helpers (cross-reasoner SAM dedup)
# ---------------------------------------------------------------------------


_NULL_CONCEPT_TOKENS = frozenset({"none", "null", "n/a", "na"})


def _is_null_token(s: str) -> bool:
    return s.strip().lower() in _NULL_CONCEPT_TOKENS


def _normalize_concept_phrase(s: str) -> str:
    """Lowercase, strip, and collapse internal whitespace.

    The case-fold normalization the union builder later assumes:
    case variants of the same phrase map to one canonical form.
    Collapsing internal whitespace handles double-spaced modifiers.
    """
    return " ".join((s or "").strip().lower().split())


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
        if not isinstance(raw_class, str) or _is_null_token(raw_class):
            continue
        s = _normalize_concept_phrase(raw_class)
        if not s:
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


# ---------------------------------------------------------------------------
# Per-reasoner ToolContext
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Empty-result helpers
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class MapsCompiler:
    """Pre-flight compiler driven by the persistent Reasoner.

    Each call to :meth:`compile` runs the reasoner's round 0 (the mission
    packet, then the uniform tool loop over the priors), string-dedups
    the union of its committed sam_concepts (no pre-flight SAM call, no
    ObjectGraph mutation; SAM segmentation and ingest are flight-time,
    per tour stop), and builds the initial tour from the committed
    requested_viewpoints.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        workdir: Path | None = None,
        sensor_description: str = "",
    ) -> None:
        # The server-discovered endpoint authority (installed once at the
        # process entry point). Read ONCE here and threaded explicitly down
        # the hot path into the Reasoner: the model name, the
        # window-derived budgets and the endpoint set all come from the live
        # server, so there is no client-side knob left to mis-set.
        self.endpoint_spec = spec()
        self.model = self.endpoint_spec.model_id
        self.reasoner_url = self.endpoint_spec.base_url
        # ``api_key`` defaults to the "EMPTY" sentinel (no-auth local vLLM).
        # Treat that sentinel (and ""/None) as "unset" so the env-resolved key
        # (SEARCH_REASONER_API_KEY -- e.g. an OpenRouter key) wins; a real explicit
        # arg still takes precedence. Plain ``api_key or env_key`` is WRONG
        # here: "EMPTY" is truthy, so it short-circuits and DISCARDS the env key,
        # and every OpenRouter reasoner call then 401s "Missing Authentication
        # header".
        env_key = os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY")
        self.api_key = api_key if (api_key and api_key != "EMPTY") else env_key
        self.sensor_description = sensor_description or ""
        if workdir is None:
            env_workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip()
            workdir = Path(env_workdir) if env_workdir else None
        self.workdir: Path | None = workdir


    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def _resolve_seed_base(self) -> int:
        """Resolve the base seed: ``SEARCH_SEED_BASE`` env if set, else a
        wall-clock nanosecond stamp. Per-attempt offsets (retry) and the
        per-reasoner ``+i`` fan-out are applied on top of this."""
        env_seed_base = os.environ.get("SEARCH_SEED_BASE")
        if env_seed_base is not None:
            return int(env_seed_base) & 0xFFFFFFFF
        return time.time_ns() & 0xFFFFFFFF

    # ------------------------------------------------------------------
    # Phase helpers (called from compile())
    # ------------------------------------------------------------------

