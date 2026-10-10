"""THE one-line table legend ("Beyond
columns" item 1: the axis-sign question alone derailed ~4 episodes per pack
in two packs). Served ONCE per response, under the FIRST envelope that
renders rows -- and that per-response copy is LOAD-BEARING: a
once-per-mission countdown regressed SR -0.042 (pysurface4, efb4a7a3),
breaking exactly the axis-sign and xy-is-a-recorded-point inferences in
legend-less tables. Every clause is the executor's own fact, re-derived
here so legend drift from the engine fails loudly.
"""
from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import (
    _TABLE_LEGEND,
    build_surface_from_graph,
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
    # 'along' reads from the end0_xy end, and end0_xy IS the centerline's
    # first vertex (test_universal_relations' law, re-pinned here because
    # the legend quotes the column by name).
    assert "end0_xy" in _TABLE_LEGEND

    class _N:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, k):
            return None

    road = _N(name="Main Street", provenance="prebaked",
              world_xy=[50.0, 0.0],
              centerline=[[0.0, 0.0], [60.0, 0.0], [100.0, 4.0]],
              tags={"highway": "residential"})
    _, anchors = build_surface_from_graph([("road_1", road)], (0.0, 0.0), 2.0)
    row = {a["fid"]: a for a in anchors}["road_1"]
    cl = row["centerline"]
    assert row["end0_xy"] == [round(cl[0][0], 2), round(cl[0][1], 2)], row


def test_legend_speaks_no_banned_vocabulary():
    # The commensurate sweep's own word lists over the one new constant
    # (its dynamic drive covers the live path; this pins the constant).
    from maps.search.tests.test_served_surface_commensurate import (
        _violations,
    )
    assert _violations(_TABLE_LEGEND) == [], _violations(_TABLE_LEGEND)
