"""A program never redefines a handle."""
from __future__ import annotations


import pytest

from maps.search.analyze_scene.geom_program import (
    ProgramError,
    run_program,
)

# a small runtime pool + a point anchor (shed) and a road (centerline)
_GRID = [
    {"fid": "c00", "xy": [10.0, 10.0], "class": "car", "color": "red"},
    {"fid": "c01", "xy": [15.0, 10.4], "class": "car", "color": "blue"},
    {"fid": "c02", "xy": [20.0, 10.8], "class": "car", "color": "white"},
    {"fid": "c10", "xy": [10.0, 22.0], "class": "car", "color": "red"},
    {"fid": "c11", "xy": [15.0, 22.4], "class": "car", "color": "blue"},
    {"fid": "c12", "xy": [20.0, 22.8], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "road_1", "name": "Main Street", "xy": [50.0, 0.0], "class": "road",
     "centerline": [[0.0, 0.0], [100.0, 0.0]]},
    {"fid": "shed_1", "name": "Shed", "xy": [15.0, -30.0], "class": "building"},
]


def _run(ops):
    return run_program([dict(c) for c in _GRID],
                       [dict(a) for a in _ANCHORS], {"ops": ops})["results"]


# --- One handle, one op --------------------------------------------------------


def test_order_free_duplicate_handle_errors():
    with pytest.raises(ProgramError) as e:
        _run([{"ref": "r0", "op": "find", "args": {"query": "car"}},
              {"ref": "r0", "op": "find", "args": {"query": "road"}}])
    assert "two ops" in str(e.value)
