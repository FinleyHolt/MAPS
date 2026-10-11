"""A window ``rN[a:b]`` serves that span of a handle's rows, and fails loud
when it selects none of a non-empty set."""
from __future__ import annotations

import pytest

from maps.search.analyze_scene.geom_program import ProgramError, run_program

_CANDS = [
    {"fid": "d0", "xy": [30.0, 50.0], "class": "car"},
    {"fid": "a1", "xy": [20.0, 50.0], "class": "car"},
    {"fid": "b1", "xy": [40.0, 50.0], "class": "car"},
]


def _window(spec):
    return run_program([dict(c) for c in _CANDS], [], {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": f"r0{spec}", "key": "xy", "op": "truthy"}},
    ]})["results"]["r1"]


def test_a_window_serves_its_span():
    assert [r["fid"] for r in _window("[1:3]")] == ["a1", "b1"]


def test_a_window_past_the_end_is_loud():
    with pytest.raises(ProgramError, match="selected 0 of 3"):
        _window("[5:8]")
