from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import (
    _TABLE_LEGEND,
    compact_results,
    run_program,
)

_CANDS = [
    {"fid": "d0", "xy": [10.0, 0.0], "class": "car", "search_text": "car"},
    {"fid": "d1", "xy": [40.0, 0.0], "class": "car", "search_text": "car"},
]
_ANCHORS = [
    {"fid": "z_solo", "name": "Manhattan House", "xy": [0.0, 0.0],
     "class": "building"},
]


def test_legend_served_once_per_response_under_the_first_grid():
    out = run_program(
        [dict(c) for c in _CANDS], [dict(a) for a in _ANCHORS],
        {"ops": [
            {"ref": "r0", "op": "find",
             "args": {"query": "car", "provenance": "runtime"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0", "refs": ["z_solo"]}},
            {"ref": "r2", "op": "score_by",
             "args": {"items": "r1",
                      "emphasis": [{"key": "distance to z_solo",
                                    "w": -1}]}},
        ]})
    envs = compact_results(out["results"])
    rowed = [k for k, e in envs.items()
             if isinstance(e, dict) and e.get("rows")]
    assert len(rowed) >= 2, envs          # vacuity floor: >=2 grids rendered
    with_legend = [k for k, e in envs.items()
                   if isinstance(e, dict) and "legend" in e]
    assert with_legend == [rowed[0]], (with_legend, rowed)
    assert envs[rowed[0]]["legend"] == _TABLE_LEGEND
    # A census-only response (an unordered browse) carries none.
    browse = run_program(
        [dict(c) for c in _CANDS], [dict(a) for a in _ANCHORS],
        {"ops": [{"ref": "r0", "op": "find",
                  "args": {"provenance": "runtime"}}]})
    benvs = compact_results(browse["results"])
    assert "census" in benvs["r0"] and "rows" not in benvs["r0"], benvs
    assert not any("legend" in e for e in benvs.values()
                   if isinstance(e, dict)), benvs


def test_legend_claims_are_the_executors_own_facts():
    # bearings: 0 = north (+y), 90 = east (+x), clockwise — the executor's
    # own compass (G.bearing), quoted verbatim by the legend.
    assert G.bearing((0, 0), (0, 1)) == 0.0
    assert G.bearing((0, 0), (1, 0)) == 90.0
    assert "0 = north" in _TABLE_LEGEND
    assert "90 = east" in _TABLE_LEGEND
    assert "clockwise" in _TABLE_LEGEND
    # distance 0.0 = touching: two masks sharing an edge serve 0.0.
    assert "0.0 = touching" in _TABLE_LEGEND
    out = run_program(
        [{"fid": "d0", "xy": [40.0, 0.0], "class": "car",
          "mask": [[[38.0, -1.0], [42.0, -1.0], [42.0, 1.0], [38.0, 1.0], [38.0, -1.0]]]}],
        [{"fid": "b1", "xy": [44.0, 0.0], "class": "building",
          "mask": [[[42.0, -1.0], [46.0, -1.0], [46.0, 1.0], [42.0, 1.0], [42.0, -1.0]]]}],
        {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0",
                      "refs": [{"ref": "b1", "as": "wall"}]}}]})
    assert out["results"]["r1"][0]["distance to wall"] == 0.0


def test_legend_quotes_no_line_chart_column():
    # compare attaches no chart of a line, so the legend explains none.
    assert "along" not in _TABLE_LEGEND
    assert "LINE" not in _TABLE_LEGEND
