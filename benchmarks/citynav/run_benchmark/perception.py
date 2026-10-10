"""Perception factory: the sam production tier and the
gt_perfect diagnostic tier.
"""

from __future__ import annotations

from benchmarks.citynav.perception_adapter import citynav_to_gt_perfect_objects
from benchmarks.citynav.run_benchmark.config import RunConfig
from pathlib import Path
from maps.perception.base import PerceptionAdapter
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient
import json
import logging

log = logging.getLogger("run_benchmark")


def _build_concept_provider(cfg: RunConfig) -> ReasonerUnionConceptProvider:
    """Build the per-frame SAM concept provider.

    The v1 stack drives SAM 3.1's class set from the canonicalized
    concept union of every active reasoner thread. The runner pushes
    that union into the provider via ``set_concepts`` after each compile
    turn (see the ``result["sam_concepts_union"]`` wiring in the step
    loop). The provider returns an empty list before the first compile
    finishes; ``SamDetectClient.detect`` short-circuits on
    empty concepts so that frame produces no detections, which is the
    correct behavior during the cold-start window.
    """
    del cfg
    return ReasonerUnionConceptProvider()


def _load_gt_color_table(cfg: RunConfig, map_name: str) -> dict:
    """The offline GT color table for ``map_name`` ({object_id: {color,...}}).

    Reads ``cfg.gt_perfect['color_table_path']`` (JSON keyed
    {map_name: {object_id: {...}}}) and returns this map's sub-table, or ``{}``
    when the path is unset / missing / the map is absent. NEVER raises — a
    missing table degrades GT-PERFECT to masks-only (no color), which the
    parity gate's color-present assert catches before any fleet."""
    path = (cfg.gt_perfect or {}).get("color_table_path")
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        log.warning("gt_perfect: color_table_path %s missing -> masks-only", path)
        return {}
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        log.warning("gt_perfect: unreadable color table %s (%s) -> masks-only", path, exc)
        return {}
    table = raw.get(map_name) or {}
    log.info("gt_perfect: loaded %d GT colors for %s", len(table), map_name)
    return table


def make_perception(
    cfg: RunConfig, objects_for_map: list, map_name: str = ""
) -> PerceptionAdapter | None:
    if cfg.perception == "gt_perfect":
        # DIAGNOSTIC perfect sensor (non-headline, config-gated).
        # Deterministic, offline, no network dependency.
        from maps.perception.synthetic import GTPerfectPerception

        # Match the baseline arm's prior-percept color calibration (E4): the
        # render-and-look enrichment of in-view priors calls _set_color_attr,
        # which reads the module-global palette. gt_perfect's OWN car color is
        # the injected GT table (palette-independent), but matching the prior
        # palette keeps the arms identical except the perception tier.
        from maps.perception.sam.detect_client import set_color_palette

        set_color_palette("citynav")
        color_table = _load_gt_color_table(cfg, map_name)
        gt_objects = citynav_to_gt_perfect_objects(objects_for_map, color_table)
        knobs = {k: v for k, v in (cfg.gt_perfect or {}).items() if k != "color_table_path"}
        return GTPerfectPerception(objects=gt_objects, **knobs)
    if cfg.perception == "sam":
        adapter = SamDetectClient(
            concept_provider=_build_concept_provider(cfg),
            **cfg.sam,
        )
        # Activate this camera's color calibration module-globally so BOTH the
        # detection path and the prior-percept enrichment path (which calls
        # _set_color_attr directly) share it. None keeps the env/default palette.
        from maps.perception.sam.detect_client import set_color_palette

        palette = adapter.color_palette
        # The recalibrated CityNav palette is the DEFAULT for the nadir-ortho
        # ortho backend. The ideal-sRGB "default" anchors are too
        # bright/saturated for the dark, low-saturation aerial imagery (red->brown,
        # dark->brown; +10.8pp color-family when calibrated, 313c5e32), so a
        # config that OMITS color_palette must NOT silently fall back to it.
        # the canonical config sets it explicitly; a config that omits it used to
        # silently run the broken default.
        # An explicit color_palette in the config still wins (e.g. "default" for
        # an isolating A/B arm).
        if not palette:
            palette = "citynav"
        set_color_palette(palette)
        return adapter
    raise ValueError(f"Unknown perception: {cfg.perception!r}")
