"""An episode starts where, how high and facing the way upstream CityNav
starts it.

Upstream reads the start pose off the split row's ``trajectory[0]``
(``gsamllavanav/dataset/mturk_trajectory.py``): ``[x, y, z, yaw, pitch]``,
or ``[x, y, z, dx, dy, dz]`` whose yaw is ``atan2(dy, dx)``; yaw is radians
counter-clockwise from east. The start altitude is the row's z held to the
flight envelope, at least a metre above the block's ground and at most the
config's ceiling; ``start_altitude_agl_m`` replaces it with a height above
ground. The heading is the executor's, in this tree's convention (degrees
clockwise from north); a row's ``[x, y, z]`` start carries none and faces
north.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("pose, heading", [
    ([0.0, 0.0, 120.0, 0.0, 0.0], 90.0),  # east
    ([0.0, 0.0, 120.0, math.pi / 2, -0.3], 0.0),  # north
    ([0.0, 0.0, 120.0, math.pi, 0.0], 270.0),  # west
    ([0.0, 0.0, 120.0, 0.0, -1.0, 0.0], 180.0),  # a direction vector, due south
    ([0.0, 0.0, 120.0], 0.0),  # no heading: north
])
def test_the_row_start_yaw_is_read_as_upstream_reads_it(pose, heading):
    assert parse_episode(_row(pose), row_index=0).start_heading_deg == pytest.approx(heading)


def test_a_start_pose_of_another_length_is_refused():
    with pytest.raises(ValueError, match="trajectory"):
        parse_episode(_row([0.0, 0.0, 120.0, 0.0]), row_index=0)


def test_an_episode_starts_at_its_rows_z_held_to_the_flight_envelope():
    cfg = load_config(str(REPO_ROOT / "configs" / "citynav.yaml"))
    z = parse_episode(_row([0.0, 0.0, 137.25, 0.0, 0.0]), row_index=0).start_z
    assert start_altitude_m(cfg, MAP, z) == (137.25, GROUND)
    assert start_altitude_m(cfg, MAP, GROUND - 30.0) == (GROUND + 1.0, GROUND)
    assert start_altitude_m(cfg, MAP, cfg.altitude_cap_m + 50.0) == (cfg.altitude_cap_m, GROUND)


def test_a_height_above_ground_replaces_the_rows_z():
    cfg = load_config(str(REPO_ROOT / "configs" / "calcity.yaml"))
    assert start_altitude_m(cfg, MAP, 125.0) == (GROUND + cfg.start_altitude_agl_m, GROUND)
