"""The chart of a line ref: naming 'distance to <road fid>' charts the rows
along the road's centerline -- 'along <road>' (arc metres from its end0) and
the unsigned lateral 'distance to <road>' -- and names no side the engine
would have to guess."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program

_GRID = [
    {"fid": "c00", "xy": [10.0, 10.0], "class": "car"},
    {"fid": "c02", "xy": [20.0, 10.8], "class": "car"},
]
_ROAD = {"fid": "road_1", "name": "Main Street", "xy": [50.0, 0.0],
         "class": "road", "centerline": [[0.0, 0.0], [100.0, 0.0]]}


def test_a_line_ref_charts_along_its_centerline_and_names_no_side():
    out = run_program([dict(c) for c in _GRID], [dict(_ROAD)], {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0",
                  "emphasis": [{"key": "distance to road_1", "w": -1}]}},
    ]})
    rows = {r["fid"]: r for r in out["results"]["r1"]}
    r = rows["c00"]
    assert "left of road_1" not in r and "right of road_1" not in r, r
    assert r["distance to road_1"] == 10.0, r
    assert abs(rows["c02"]["along road_1"] - 20.0) < 1.0
