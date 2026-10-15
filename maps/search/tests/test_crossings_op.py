"""The crossing referent: ONE row per distinct crossing, loud at zero."""
from __future__ import annotations

import glob
import json
import math
import os

import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import ProgramError, run_program

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# THE bake the run would use: ``SEARCH_CITYNAV_TILES`` wins, else the shipped default --
# the rule ``benchmarks.citynav.paths`` applies, read off the env directly because
# ``maps`` never imports ``benchmarks``.
_TILES = (os.environ.get("SEARCH_CITYNAV_TILES")
          or os.path.join(_REPO, "tiles", "citynav_2m_osmv3"))


def _road(fid, poly, name=None):
    # A row's geometry is its ``mask`` -- the exact chains it was given, one
    # per part. A way is open; nothing here declares it is a road.
    return {"fid": fid, "name": name or fid, "xy": list(poly[0]),
            "mask": [[list(p) for p in poly]], "tags": {"highway": "x"},
            "search_text": ""}


def _run(anchors, a, b):
    prog = {"ops": [{"ref": "r0", "op": "crossings",
                     "args": {"refs": [a, b]}}]}
    return run_program([], anchors, prog)["results"]["r0"]


# --- the primitive ----------------------------------------------------------

def test_a_mid_span_crossing_is_found_with_no_vertex_near_it():
    # Two long segments crossing at their middles: every endpoint is 40-50 m
    # from the other line, so an endpoint-distance-only reading sees nothing.
    pts, near = G.polyline_crossings([[[0, 0], [100, 0]]],
                                     [[[50, -40], [50, 40]]])
    assert near == 0.0
    assert [[round(v, 6) for v in p] for p in pts] == [[50.0, 0.0]]


def test_every_crossing_of_a_zigzag_is_its_own_point():
    # Four genuine crossings, 20 m apart: the answer is FOUR points, and it
    # is emphatically not their centroid (which sits at 40, 0 — on the line
    # but at none of the crossings).
    zig = [[0, -10], [20, 10], [40, -10], [60, 10], [80, -10]]
    pts, near = G.polyline_crossings([zig], [[[0, 0], [80, 0]]])
    assert near == 0.0
    assert [[round(v, 6) for v in p] for p in pts] == [
        [10.0, 0.0], [30.0, 0.0], [50.0, 0.0], [70.0, 0.0]]
    centre = [sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0]
    # the ONE answer a centroid would have given sits 10 m from the nearest
    # real crossing and ON none of them
    assert min(math.dist(centre, p) for p in pts) >= 10.0


def test_a_t_junction_meets_without_crossing():
    pts, near = G.polyline_crossings([[[0, 0], [100, 0]]],
                                     [[[50, 0], [50, 40]]])
    assert near == 0.0 and pts == [[50.0, 0.0]]


def test_one_road_split_into_two_ways_is_still_one_crossing():
    # OSM splits ways AT junctions, so the road that crosses is routinely two
    # rows both ENDING on the junction node. A bundle per side is why that
    # reads as one crossing rather than two coincident ones.
    pts, _ = G.polyline_crossings([[[0, 0], [50, 0]], [[50, 0], [100, 0]]],
                                  [[[50, -30], [50, 30]]])
    assert pts == [[50.0, 0.0]]


def test_features_that_never_come_close_report_their_gap():
    pts, near = G.polyline_crossings([[[0, 0], [10, 0]]], [[[0, 50], [10, 50]]])
    assert pts == [] and near == 50.0


def test_a_point_feature_meets_a_line_at_its_own_position():
    pts, near = G.polyline_crossings([[[5, 0]]], [[[0, 0], [10, 0]]])
    assert near == 0.0 and pts == [[5.0, 0.0]]


def test_near_is_not_touching():
    import numpy as np

    def ring(r):
        return [[r * math.cos(t), r * math.sin(t)]
                for t in np.linspace(0, 2 * math.pi, 60)]

    pts, near = G.polyline_crossings([ring(40.0)], [ring(41.6)])
    assert pts == [] and 1.5 < near < 1.7, (pts, near)

    lo = [[float(x), 0.0] for x in range(0, 200, 4)]
    hi = [[float(x), 1.5] for x in range(0, 200, 4)]
    pts, near = G.polyline_crossings([lo], [hi])
    assert pts == [] and near == 1.5, (pts, near)

    # And a road that shares the junction node touches at exactly 0.0.
    pts, near = G.polyline_crossings([lo], [[[100.0, -20.0], [100.0, 0.0]]])
    assert near == 0.0 and pts == [[100.0, 0.0]]


def test_a_non_finite_vertex_does_not_report_an_infinite_gap():
    pts, near = G.polyline_crossings(
        [[[0, 0], [float("nan"), 0], [10, 0]]], [[[0, 50], [10, 50]]])
    assert pts == [] and near == 50.0        # not "inf m" in the refusal


def test_nothing_crosses_itself():
    # Every segment touches its own neighbours, so a row on BOTH sides hands
    # back the polyline's own corners as "crossings" — a confident answer
    # with no error. Reachable by naming one road twice and, the real case,
    # by two name phrases fuzzy-grounding onto the same row.
    anchors = [_road("prior_1", [[0, -10], [20, 10], [40, -10]], "Mill Lane"),
               _road("prior_2", [[0, 0], [40, 0]], "Silver Street")]
    with pytest.raises(ProgramError, match="nothing crosses itself"):
        _run(anchors, "prior_1", "prior_1")
    with pytest.raises(ProgramError, match="the same feature in part"):
        _run(anchors, ["prior_1", "prior_2"], "prior_2")


# --- the op contract --------------------------------------------------------

def test_each_crossing_is_a_row_with_its_own_fid_and_its_own_point():
    anchors = [_road("prior_1", [[0, -10], [20, 10], [40, -10], [60, 10]],
                     "Zig Road"),
               _road("prior_2", [[0, 0], [60, 0]], "Straight Street")]
    rows = _run(anchors, "prior_1", "prior_2")
    assert len(rows) == 3, rows
    assert len({r["fid"] for r in rows}) == 3      # individually selectable
    assert len({tuple(r["xy"]) for r in rows}) == 3
    for r in rows:
        # the fid IS the point: the tree's uniform fid-or-point address, so
        # the row flies / stages / maps with no new plumbing
        from maps.util.xy_token import parse_xy_token
        assert parse_xy_token(r["fid"]) == tuple(r["xy"])
        assert r["name"] == "Zig Road x Straight Street"
    assert "3 distinct crossings" in " ".join(rows.notes)
    assert "at none of them" in " ".join(rows.notes)


def test_zero_crossings_refuse_loudly_and_never_bind():
    anchors = [_road("prior_1", [[0, 0], [100, 0]], "Wellington Road"),
               _road("prior_2", [[0, 84], [100, 84]], "Grosvenor Road")]
    with pytest.raises(ProgramError) as e:
        _run(anchors, "prior_1", "prior_2")
    msg = str(e.value)
    assert "Wellington Road and Grosvenor Road do not meet" in msg
    assert "prior_1" in msg and "prior_2" in msg
    assert "come no closer than 84.0 m" in msg      # the measured fact
    assert "is not their intersection" in msg       # no near-ish fallback


def test_a_side_with_no_geometry_refuses_rather_than_guessing():
    anchors = [_road("prior_1", [[0, 0], [100, 0]]),
               {"fid": "prior_2", "name": "Nothing", "xy": None,
                "centerline": None, "tags": {}, "search_text": ""}]
    with pytest.raises(ProgramError, match="names no geometry"):
        _run(anchors, "prior_1", "prior_2")


def test_a_road_named_by_several_rows_passes_as_one_bundle():
    # OSM splits a way ON the junction node, so both halves END there.
    anchors = [_road("prior_1", [[0, 0], [50, 0]], "Split Road"),
               _road("prior_2", [[50, 0], [100, 0]], "Split Road"),
               _road("prior_3", [[50, -30], [50, 30]], "Cross Street")]
    rows = _run(anchors, ["prior_1", "prior_2"], "prior_3")
    assert len(rows) == 1
    assert rows[0]["name"] == "Split Road x Cross Street"


def test_the_crossings_chain_like_any_other_rows():
    # The narrowing the language does ("the nearest one.") is score_by over
    # THESE rows, so they have to survive compare + score_by unchanged.
    anchors = [_road("prior_1", [[0, -10], [20, 10], [40, -10], [60, 10]],
                     "Zig Road"),
               _road("prior_2", [[0, 0], [60, 0]], "Straight Street"),
               {"fid": "prior_3", "name": "Depot", "xy": [50.0, 0.0],
                "centerline": None, "tags": {}, "search_text": ""}]
    prog = {"ops": [
        {"ref": "r0", "op": "crossings",
         "args": {"refs": ["prior_1", "prior_2"]}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_3"]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "distance to prior_3", "w": -1}]}},
    ]}
    rows = run_program([], anchors, prog)["results"]["r2"]
    assert len(rows) == 3
    d = [r["distance to prior_3"] for r in rows]
    assert d == sorted(d) and d[0] < d[-1]   # a real ordering over crossings


def test_a_crossing_row_anchors_geometry_as_a_point():
    anchors = [_road("prior_1", [[0, 0], [100, 0]], "Long Road"),
               _road("prior_2", [[50, -30], [50, 30]], "Cross Street"),
               {"fid": "prior_3", "name": "Depot", "xy": [50.0, 30.0],
                "centerline": None, "tags": {}, "search_text": ""}]
    prog = {"ops": [
        {"ref": "r0", "op": "crossings",
         "args": {"refs": ["prior_1", "prior_2"]}},
        {"ref": "r1", "op": "find", "args": {}},
        {"ref": "r2", "op": "compare",
         "args": {"items": "r1", "refs": [{"ref": ["r0"], "as": "corner"}]}},
    ]}
    rows = run_program([], anchors, prog)["results"]["r2"]
    by = {r["fid"]: r for r in rows}
    assert math.isclose(by["prior_3"]["distance to corner"], 30.0, abs_tol=0.5)


# --- the same op, walked over a real bake -----------------------------------

def _blocks():
    return sorted(p[:-len("_meta.json")]
                  for p in glob.glob(os.path.join(_TILES, "*_meta.json")))


def _named_ways(stem):
    """Every named feature of one block as its EXACT chains, by name."""
    nodes = json.loads(open(stem + "_nodes.json").read())
    out: dict = {}
    for _fid, n in sorted(nodes.items()):
        wc = n.get("world_contour")
        if not (n.get("name") and (n.get("tags") or {}).get("highway") and wc):
            continue
        out.setdefault(n["name"], []).append([[float(x), float(y)]
                                              for x, y in wc])
    return out


@pytest.mark.skipif(not os.path.isdir(_TILES), reason="tile bake absent")
def test_every_crossing_the_bake_emits_is_on_both_things_it_names():
    """A crossing is CONTACT, so the point it reports is ON both features -- not near
    them, on them."""
    import itertools
    worst, n, pairs = 0.0, 0, 0
    for stem in _blocks():
        by_name = _named_ways(stem)
        pairs += len(by_name)
        for na, nb in itertools.combinations(sorted(by_name), 2):
            pa, pb = by_name[na], by_name[nb]
            pts, _ = G.polyline_crossings(pa, pb)
            for p in pts:
                n += 1
                worst = max(worst, max(
                    G.gap_witness(G.mask_geom([[list(p)]]),
                                  G.mask_geom(side))[0]
                    for side in (pa, pb)))
    # VACUITY FLOORS.
    assert pairs >= 100, (
        f"only {pairs} named road features carry exact geometry -- re-bake "
        f"{_TILES} (benchmarks/citynav/data_prep/bake_tiles.sh)")
    assert n >= 250, n
    # The ceiling is the storage precision (2 dp), not a touch band: a
    # crossing computed mid-segment is emitted rounded, which can move it by
    # at most sqrt(2)/2 cm. Anything above that is a point off the roads.
    assert worst <= 0.01, worst


@pytest.mark.skipif(not os.path.isdir(_TILES), reason="tile bake absent")
def test_the_bake_really_does_carry_roads_that_cross_more_than_once():
    """The multi-crossing requirement is not hypothetical, and this is the floor that
    says so: if a bake ever served only single crossings, the test above would still
    pass while the case they exist for went untested."""
    import itertools
    multi = pairs = 0
    for stem in _blocks():
        by_name = _named_ways(stem)
        pairs += len(by_name)
        for na, nb in itertools.combinations(sorted(by_name), 2):
            pts, _ = G.polyline_crossings(by_name[na], by_name[nb])
            if len(pts) > 1:
                multi += 1
                # every crossing of a multi-crossing pair is a DISTINCT place
                for p, q in itertools.combinations(pts, 2):
                    assert p != q, (stem, na, nb, p)
    assert pairs >= 100, (
        f"only {pairs} named road features carry exact geometry -- re-bake "
        f"{_TILES} (benchmarks/citynav/data_prep/bake_tiles.sh)")
    assert multi >= 8, multi


def test_rounding_never_serves_two_rows_the_same_address():
    """Two crossings under a metre apart collapse to one address."""
    # A hairpin crossing the road twice, at x=50.0 and x=50.4: two exact
    # intersections 0.4 m apart, one whole-metre address.
    hairpin = [[49.8, -10], [50.2, 10], [50.6, -10]]
    anchors = [_road("prior_1", [[0, 0], [100, 0]], "Long Road"),
               _road("prior_2", hairpin, "Hairpin Way")]
    pts, _near = G.polyline_crossings([[[0, 0], [100, 0]]], [hairpin])
    assert len(pts) == 2, pts          # the exact geometry really does cross twice
    rows = _run(anchors, "prior_1", "prior_2")
    assert len({r["fid"] for r in rows}) == len(rows)
    assert rows[0]["fid"] == "[50, 0]"


def test_a_road_ends_are_rows_you_can_go_to():
    anchors = [dict(_road("prior_1", [[0, -100], [0, 100]], "Walker Court"),
                    centerline=[[0, -100], [0, 100]])]
    prog = {"ops": [{"ref": "r0", "op": "ends", "args": {"refs": ["prior_1"]}}]}
    rows = run_program([], anchors, prog)["results"]["r0"]
    assert [r["fid"] for r in rows] == ["end_0", "end_1"]
    assert [r["xy"] for r in rows] == [[0.0, -100.0], [0.0, 100.0]]


def test_a_footprint_has_no_ends_and_says_so():
    import pytest

    from maps.search.analyze_scene.geom_program import ProgramError
    anchors = [{"fid": "prior_9", "xy": [0.0, 0.0], "name": "A Shed",
                "mask": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                "centerline": None, "cells": [], "search_text": "a shed"}]
    with pytest.raises(ProgramError, match="prior_9.*no line geometry"):
        run_program([], anchors,
                    {"ops": [{"ref": "r0", "op": "ends",
                              "args": {"refs": ["prior_9"]}}]})


def test_a_sibling_segment_that_crosses_is_named():
    import pytest

    from maps.search.analyze_scene.geom_program import ProgramError
    anchors = [_road("prior_1", [[0, 0], [100, 0]], "Airway Boulevard"),
               _road("prior_2", [[0, 84], [100, 84]], "Lime Avenue"),
               _road("prior_3", [[50, -10], [50, 10]], "Lime Avenue")]
    with pytest.raises(ProgramError) as e:
        _run(anchors, "prior_1", "prior_2")
    msg = str(e.value)
    assert "do not meet HERE" in msg, msg
    assert "prior_3" in msg and "touch at [50, 0]" in msg, msg


def test_a_crossing_set_anchors_a_distance_column():
    anchors = [_road("prior_1", [[0, 0], [100, 0]], "A Road"),
               _road("prior_2", [[50, -50], [50, 50]], "B Road")]
    cands = [{"fid": "d0", "xy": [60.0, 10.0], "class": "house",
              "search_text": "house"}]
    prog = {"ops": [
        {"ref": "r0", "op": "crossings",
         "args": {"refs": ["prior_1", "prior_2"]}},
        {"ref": "r1", "op": "find",
         "args": {"query": "house"}},
        {"ref": "r2", "op": "compare",
         "args": {"items": "r1", "refs": [{"ref": ["r0"], "as": "inter"}]}},
    ]}
    out = run_program(cands, anchors, prog)["results"]["r2"]
    rows = list(out)
    assert rows, out
    assert any("distance to inter" in r for r in rows), rows[0]


def test_a_look_list_crops_each_element():
    """look([a, b]) asks for one crop token PER element."""
    from maps.search.tests.fakes import acts_of_block
    for fids in (["prior_1", "prior_2"],
                 ["prior_1", "prior_2", "prior_3"]):
        assert acts_of_block(f"look({fids!r})\n") == [
            ("crops", None, fids, None)]
    # ...and a genuine [x, y] point still rides as ONE point token.
    assert acts_of_block("look([100, 200])\n") == [
        ("crops", None, "[100, 200]", None)]
    # A NAME in the list grounds as it does alone, to its best match.
    from maps.search.tests.fakes import calls_of_block
    assert calls_of_block('look(["Long Road", "prior_2"])\n') == [
        ("ground", {"Long Road": "crops"}),
        ("crops", None, ["<Long Road>", "prior_2"], None)]
