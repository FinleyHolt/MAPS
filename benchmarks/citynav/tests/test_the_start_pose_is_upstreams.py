"""An episode starts at the altitude upstream CityNav starts it at.

Upstream reads the start pose off the split row's ``trajectory[0]``
(``gsamllavanav/dataset/mturk_trajectory.py``). The start altitude is the
row's z held to the flight envelope, at least a metre above the block's
ground and at most the config's ceiling; ``start_altitude_agl_m`` replaces it
with a height above ground.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.citynav.adapter import parse_episode
from benchmarks.citynav.run_benchmark import load_config
from benchmarks.citynav.run_benchmark.episode_runner import start_altitude_m
from maps.perception.ortho.ground_level import GROUND_LEVEL

MAP = "birmingham_block_1"
GROUND = float(GROUND_LEVEL[MAP])
REPO_ROOT = Path(__file__).resolve().parents[3]


def _row(pose: list) -> dict:
    return {"area": "birmingham", "block": 1, "descriptions": ["the church"],
            "ann_ids": [0], "target_positions": [[10.0, 20.0, 0.0]],
            "trajectory": [pose]}


def test_an_episode_starts_at_its_rows_z_held_to_the_flight_envelope():
    cfg = load_config(str(REPO_ROOT / "configs" / "citynav.yaml"))
    z = parse_episode(_row([0.0, 0.0, 137.25, 0.0, 0.0]), row_index=0).start_z
    assert start_altitude_m(cfg, MAP, z) == (137.25, GROUND)
    assert start_altitude_m(cfg, MAP, GROUND - 30.0) == (GROUND + 1.0, GROUND)
    assert start_altitude_m(cfg, MAP, cfg.altitude_cap_m + 50.0) == (cfg.altitude_cap_m, GROUND)
    cfg.altitude_m = 50.0  # the cruise fallback is not the start pose
    assert start_altitude_m(cfg, MAP, 137.25)[0] == 137.25


def test_a_height_above_ground_replaces_the_rows_z():
    cfg = load_config(str(REPO_ROOT / "configs" / "calcity.yaml"))
    assert start_altitude_m(cfg, MAP, 125.0) == (GROUND + cfg.start_altitude_agl_m, GROUND)
