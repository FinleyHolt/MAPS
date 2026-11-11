"""The start altitude IS the crop window's observation ceiling.

``obs_span_m = 2 * (start_alt - ground)`` is the view-square side the
mission hands every crop request (see
``maps/render/look_render.py``, and
``maps/render/tests/test_look_obs_span_ceiling.py`` for what the
renderer does with it). On the CityNav test split it never binds: the
pixel-budget ceiling, 100 m on a 0.1 m/px ortho, is what clamps the window.
Upstream's start z puts 4 val_seen and 60 train_seen episodes low enough
that the observation extent clamps instead.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from benchmarks.citynav import paths
from benchmarks.citynav.adapter import map_name_for_row
from benchmarks.run_record.config import load_config
from benchmarks.citynav.run_benchmark.episode_runner import start_altitude_m
from maps.render.look_render import look_window_bounds_m

# Every CityNav block is a 0.1 m/px ortho.
_, _BUDGET_M = look_window_bounds_m(0.1)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_FILE = Path(paths.SPLITS_DIR) / paths.SPLIT_FILES["test_unseen"]
needs_split = pytest.mark.skipif(
    not SPLIT_FILE.exists(), reason=f"no CityNav split at {SPLIT_FILE}"
)


def _cfg():
    return load_config(str(REPO_ROOT / "configs" / "citynav.yaml"))


def _span(cfg, map_name: str, start_z: float) -> float:
    alt, ground = start_altitude_m(cfg, map_name, start_z)
    return 2.0 * (alt - ground)


@needs_split
def test_the_observation_ceiling_never_binds_on_the_citynav_test_split():
    # Every episode of the test split: the ceiling sits ABOVE the window
    # budget, so the budget is what clamps and the observation extent
    # never does. If this fails, the crop window narrowed and the
    # look_render / mission loop notes that say otherwise are stale.
    cfg = _cfg()
    rows = json.loads(SPLIT_FILE.read_text())
    spans = []
    maps = set()
    for i, row in enumerate(rows):
        mn = map_name_for_row(row)
        maps.add(mn)
        for d_i, _ in enumerate(row.get("descriptions") or [None]):
            spans.append(_span(cfg, mn, float(row["trajectory"][0][2])))
    # VACUITY FLOOR: a split that failed to parse would leave an empty
    # sweep, and "no span was below the budget" would pass on nothing.
    assert len(maps) >= 6, sorted(maps)
    assert len(spans) >= 2000, len(spans)
    assert min(spans) > _BUDGET_M, (min(spans), _BUDGET_M)
    assert 150.0 < statistics.median(spans) < 250.0, statistics.median(spans)
