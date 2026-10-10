"""The start altitude IS the crop window's observation ceiling.

``obs_span_m = 2 * (start_alt - ground)`` — the view-square side the
mission hands every crop request (see
``maps/search/render/look_render.py``, and
``maps/search/tests/test_look_obs_span_ceiling.py`` for what the
renderer does with it). This file pins the number itself, because it was
wrong in the tree: a comment recorded a 25.8 m median span, which is
``2 * (cfg.altitude_m - ground)`` — the inert non-ortho fallback, not the
per-episode band sample the runner actually flies. On the CityNav splits
the real ceiling is 112-291 m and never binds (the pixel-budget
ceiling, 100 m on a 0.1 m/px ortho, is what clamps the window).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from benchmarks.citynav import paths
from benchmarks.citynav.adapter import map_name_for_row
from benchmarks.citynav.run_benchmark import load_config
from benchmarks.citynav.run_benchmark.episode_runner import start_altitude_m
from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.search.render.look_render import look_window_bounds_m

# Every CityNav block is a 0.1 m/px ortho.
_, _BUDGET_M = look_window_bounds_m(0.1)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_FILE = Path(paths.SPLITS_DIR) / paths.SPLIT_FILES["val_seen"]
needs_split = pytest.mark.skipif(
    not SPLIT_FILE.exists(), reason=f"no CityNav split at {SPLIT_FILE}"
)


def _cfg():
    return load_config(str(REPO_ROOT / "configs" / "citynav.yaml"))


def _span(cfg, map_name: str, episode_id: str) -> float:
    alt, ground = start_altitude_m(cfg, map_name, episode_id)
    return 2.0 * (alt - ground)


def test_the_start_altitude_does_not_read_the_inert_fallback():
    # THE regression this file exists for. ``cfg.altitude_m`` is the
    # non-ortho fallback; the ortho start pose is the seeded band sample.
    # Moving the fallback must not move the span by one metre — and at
    # the fallback's own value the span would be a quarter of the truth.
    cfg = _cfg()
    mn = "birmingham_block_1"
    eid = f"{mn}::0#0"
    before = _span(cfg, mn, eid)
    cfg.altitude_m = 50.0
    assert _span(cfg, mn, eid) == before, cfg.altitude_m
    cfg.altitude_m = 999.0
    assert _span(cfg, mn, eid) == before, cfg.altitude_m
    # VACUITY FLOOR: the fallback is a DIFFERENT number, so "unchanged"
    # is a real assertion and not two names for one value.
    ground = float(GROUND_LEVEL[mn])
    assert abs(2.0 * (50.0 - ground) - before) > 100.0, before


def test_the_smoke_episode_span_is_the_one_the_flight_reported():
    # birmingham_block_1::0#0 flown end to end (train_seen smoke,
    # 2026-08-05): four flights, every one at 141.3 m over a 10.03 m
    # ground datum, obs_span_m 262.6 m on the crop request. The seeded
    # sample and the ground datum together are what produce it.
    alt, ground = start_altitude_m(_cfg(), "birmingham_block_1",
                                   "birmingham_block_1::0#0")
    assert round(alt, 2) == 141.33, alt
    assert round(ground, 2) == 10.03, ground
    assert round(2.0 * (alt - ground), 1) == 262.6


@needs_split
def test_the_observation_ceiling_never_binds_on_the_citynav_splits():
    # Every episode of a real split: the ceiling sits ABOVE the window
    # budget, so the budget is what clamps and the observation extent
    # never does. If this fails, the crop window narrowed and the
    # look_render / reasoner_loop notes that say otherwise are stale.
    cfg = _cfg()
    rows = json.loads(SPLIT_FILE.read_text())
    spans = []
    maps = set()
    for i, row in enumerate(rows):
        mn = map_name_for_row(row)
        maps.add(mn)
        for d_i, _ in enumerate(row.get("descriptions") or [None]):
            spans.append(_span(cfg, mn, f"{mn}::{i}#{d_i}"))
    # VACUITY FLOOR: a split that failed to parse would leave an empty
    # sweep, and "no span was below the budget" would pass on nothing.
    assert len(maps) >= 20, sorted(maps)
    assert len(spans) >= 2000, len(spans)
    assert min(spans) > _BUDGET_M, (min(spans), _BUDGET_M)
    assert 150.0 < statistics.median(spans) < 250.0, statistics.median(spans)
