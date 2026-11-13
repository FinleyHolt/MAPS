"""The crossing referent: ONE row per distinct crossing, loud at zero."""

from __future__ import annotations

import glob
import json
import math
import os

import pytest

from maps.query import spatial_math as G
from maps.query.op_io import ProgramError
from maps.query.program import run_program
from maps.world_state.prior_graph import build_object_graph

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# THE bake the run would use: ``SEARCH_CITYNAV_TILES`` wins, else the shipped default --
# the rule ``benchmarks.citynav.paths`` applies, read off the env directly because
# ``maps`` never imports ``benchmarks``.
_TILES = os.environ.get("SEARCH_CITYNAV_TILES") or os.path.join(_REPO, "tiles", "citynav_2m_osmv3")


def _road(fid, poly, name=None):
    # A row's geometry is its ``mask`` -- the exact chains it was given, one
    # per part. A way is open; nothing here declares it is a road.
    return {
        "fid": fid,
        "name": name or fid,
        "xy": list(poly[0]),
        "mask": [[list(p) for p in poly]],
        "tags": {"highway": "x"},
        "search_text": "",
    }


def _run(anchors, a, b):
    prog = {"ops": [{"ref": "r0", "op": "crossings", "args": {"refs": [a, b]}}]}
    return run_program([], anchors, prog)["results"]["r0"]


# --- the primitive ----------------------------------------------------------


def test_a_mid_span_crossing_is_found_with_no_vertex_near_it():
    # Two long segments crossing at their middles: every endpoint is 40-50 m
    # from the other line, so an endpoint-distance-only reading sees nothing.
    pts, near = G.polyline_crossings([[[0, 0], [100, 0]]], [[[50, -40], [50, 40]]])
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
        [10.0, 0.0],
        [30.0, 0.0],
        [50.0, 0.0],
        [70.0, 0.0],
    ]
    centre = [sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0]
    # the ONE answer a centroid would have given sits 10 m from the nearest
    # real crossing and ON none of them
    assert min(math.dist(centre, p) for p in pts) >= 10.0


def test_a_t_junction_meets_without_crossing():
    pts, near = G.polyline_crossings([[[0, 0], [100, 0]]], [[[50, 0], [50, 40]]])
    assert near == 0.0 and pts == [[50.0, 0.0]]


def test_one_road_split_into_two_ways_is_still_one_crossing():
    # OSM splits ways AT junctions, so the road that crosses is routinely two
    # rows both ENDING on the junction node. A bundle per side is why that
    # reads as one crossing rather than two coincident ones.
    pts, _ = G.polyline_crossings([[[0, 0], [50, 0]], [[50, 0], [100, 0]]], [[[50, -30], [50, 30]]])
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
        return [[r * math.cos(t), r * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 60)]

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
    pts, near = G.polyline_crossings([[[0, 0], [float("nan"), 0], [10, 0]]], [[[0, 50], [10, 50]]])
    assert pts == [] and near == 50.0  # not "inf m" in the refusal


def test_nothing_crosses_itself():
    # Every segment touches its own neighbours, so a row on BOTH sides hands
    # back the polyline's own corners as "crossings" — a confident answer
    # with no error. Reachable by naming one road twice and, the real case,
    # by two name phrases fuzzy-grounding onto the same row.
    anchors = [
        _road("prior_1", [[0, -10], [20, 10], [40, -10]], "Mill Lane"),
        _road("prior_2", [[0, 0], [40, 0]], "Silver Street"),
    ]
    with pytest.raises(ProgramError, match="nothing crosses itself"):
        _run(anchors, "prior_1", "prior_1")
    with pytest.raises(ProgramError, match="the same feature in part"):
        _run(anchors, ["prior_1", "prior_2"], "prior_2")


# --- the op contract --------------------------------------------------------


def test_each_crossing_is_a_row_with_its_own_fid_and_its_own_point():
    anchors = [
        _road("prior_1", [[0, -10], [20, 10], [40, -10], [60, 10]], "Zig Road"),
        _road("prior_2", [[0, 0], [60, 0]], "Straight Street"),
    ]
    rows = _run(anchors, "prior_1", "prior_2")
    assert len(rows) == 3, rows
    assert len({r["fid"] for r in rows}) == 3  # individually selectable
    assert len({tuple(r["xy"]) for r in rows}) == 3
    for r in rows:
        # the fid IS the point: the tree's uniform fid-or-point address, so
        # the row flies / stages / maps with no new plumbing
        from maps.xy_token import parse_xy_token

        assert parse_xy_token(r["fid"]) == tuple(r["xy"])
        assert r["name"] == "Zig Road x Straight Street"
    assert "3 distinct crossings" in " ".join(rows.notes)
    assert "at none of them" in " ".join(rows.notes)


def test_a_crossing_of_a_mixed_set_is_named_by_the_rows_it_is_on():
    # A set whose rows carry different names (or none) labels each crossing
    # by the row it lies on, never by the one name the set happens to hold.
    unnamed = {**_road("prior_4", [[100, -30], [100, 30]]), "name": None}
    anchors = [
        _road("prior_1", [[50, -30], [50, 30]], "Mill Lane"),
        unnamed,
        _road("prior_2", [[150, -30], [150, 30]], "Oak Street"),
        _road("prior_3", [[0, 0], [200, 0]], "Cross Street"),
    ]
    rows = _run(anchors, ["prior_1", "prior_4"], "prior_3")
    assert [r["name"] for r in rows] == ["Mill Lane x Cross Street", "prior_4 x Cross Street"], rows
    rows = _run(anchors, ["prior_1", "prior_2"], "prior_3")
    assert [r["name"] for r in rows] == ["Mill Lane x Cross Street", "Oak Street x Cross Street"], (
        rows
    )
    assert "2 distinct crossings of Mill Lane / Oak Street and Cross Street" in rows.notes[0]


def test_zero_crossings_refuse_loudly_and_never_bind():
    anchors = [
        _road("prior_1", [[0, 0], [100, 0]], "Wellington Road"),
        _road("prior_2", [[0, 84], [100, 84]], "Grosvenor Road"),
    ]
    with pytest.raises(ProgramError) as e:
        _run(anchors, "prior_1", "prior_2")
    msg = str(e.value)
    assert "Wellington Road and Grosvenor Road do not meet" in msg
    assert "prior_1" in msg and "prior_2" in msg
    assert "come no closer than 84.0 m" in msg  # the measured fact
    assert "is not their intersection" in msg  # no near-ish fallback


def test_a_side_with_no_geometry_refuses_rather_than_guessing():
    anchors = [
        _road("prior_1", [[0, 0], [100, 0]]),
        {
            "fid": "prior_2",
            "name": "Nothing",
            "xy": None,
            "centerline": None,
            "tags": {},
            "search_text": "",
        },
    ]
    with pytest.raises(ProgramError, match="names no geometry"):
        _run(anchors, "prior_1", "prior_2")


def test_a_road_named_by_several_rows_passes_as_one_bundle():
    # OSM splits a way ON the junction node, so both halves END there.
    anchors = [
        _road("prior_1", [[0, 0], [50, 0]], "Split Road"),
        _road("prior_2", [[50, 0], [100, 0]], "Split Road"),
        _road("prior_3", [[50, -30], [50, 30]], "Cross Street"),
    ]
    rows = _run(anchors, ["prior_1", "prior_2"], "prior_3")
    assert len(rows) == 1
    assert rows[0]["name"] == "Split Road x Cross Street"


def test_the_crossings_chain_like_any_other_rows():
    # The narrowing the language does ("the nearest one.") is score_by over
    # THESE rows, so they have to survive compare + score_by unchanged.
    anchors = [
        _road("prior_1", [[0, -10], [20, 10], [40, -10], [60, 10]], "Zig Road"),
        _road("prior_2", [[0, 0], [60, 0]], "Straight Street"),
        {
            "fid": "prior_3",
            "name": "Depot",
            "xy": [50.0, 0.0],
            "centerline": None,
            "tags": {},
            "search_text": "",
        },
    ]
    prog = {
        "ops": [
            {"ref": "r0", "op": "crossings", "args": {"refs": ["prior_1", "prior_2"]}},
            {"ref": "r1", "op": "compare", "args": {"items": "r0", "refs": ["prior_3"]}},
            {
                "ref": "r2",
                "op": "score_by",
                "args": {"items": "r1", "emphasis": [{"key": "distance to prior_3", "w": -1}]},
            },
        ]
    }
    rows = run_program([], anchors, prog)["results"]["r2"]
    assert len(rows) == 3
    d = [r["distance to prior_3"] for r in rows]
    assert d == sorted(d) and d[0] < d[-1]  # a real ordering over crossings


def test_a_crossing_row_anchors_geometry_as_a_point():
    anchors = [
        _road("prior_1", [[0, 0], [100, 0]], "Long Road"),
        _road("prior_2", [[50, -30], [50, 30]], "Cross Street"),
        {
            "fid": "prior_3",
            "name": "Depot",
            "xy": [50.0, 30.0],
            "centerline": None,
            "tags": {},
            "search_text": "",
        },
    ]
    prog = {
        "ops": [
            {"ref": "r0", "op": "crossings", "args": {"refs": ["prior_1", "prior_2"]}},
            {"ref": "r1", "op": "find", "args": {}},
            {
                "ref": "r2",
                "op": "compare",
                "args": {"items": "r1", "refs": [{"ref": ["r0"], "as": "corner"}]},
            },
        ]
    }
    rows = run_program([], anchors, prog)["results"]["r2"]
    by = {r["fid"]: r for r in rows}
    assert math.isclose(by["prior_3"]["distance to corner"], 30.0, abs_tol=0.5)


# --- the same op, walked over a real bake -----------------------------------


def _blocks():
    return sorted(p[: -len("_meta.json")] for p in glob.glob(os.path.join(_TILES, "*_meta.json")))


def _named_ways(stem):
    """Every named feature of one block as its EXACT chains, by name."""
    meta = json.loads(open(stem + "_meta.json").read())
    nodes = build_object_graph(
        json.loads(open(stem + "_features.json").read()), cell_size_m=float(meta["cell_size_m"])
    ).nodes
    out: dict = {}
    for _fid, n in sorted(nodes.items()):
        wc = n.world_contour
        if not (n.name and (n.tags or {}).get("highway") and wc):
            continue
        out.setdefault(n.name, []).append([[float(x), float(y)] for x, y in wc])
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
                worst = max(
                    worst,
                    max(
                        G.gap_witness(G.mask_geom([[list(p)]]), G.mask_geom(side))[0]
                        for side in (pa, pb)
                    ),
                )
    # VACUITY FLOORS.
    assert pairs >= 100, (
        f"only {pairs} named road features carry exact geometry -- re-bake "
        f"{_TILES} (benchmarks/citynav/data_prep/bake_tiles.sh)"
    )
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
        f"{_TILES} (benchmarks/citynav/data_prep/bake_tiles.sh)"
    )
    assert multi >= 8, multi


def test_rounding_never_serves_two_rows_the_same_address():
    """Two crossings under a metre apart collapse to one address."""
    # A hairpin crossing the road twice, at x=50.0 and x=50.4: two exact
    # intersections 0.4 m apart, one whole-metre address.
    hairpin = [[49.8, -10], [50.2, 10], [50.6, -10]]
    anchors = [
        _road("prior_1", [[0, 0], [100, 0]], "Long Road"),
        _road("prior_2", hairpin, "Hairpin Way"),
    ]
    pts, _near = G.polyline_crossings([[[0, 0], [100, 0]]], [hairpin])
    assert len(pts) == 2, pts  # the exact geometry really does cross twice
    rows = _run(anchors, "prior_1", "prior_2")
    assert len({r["fid"] for r in rows}) == len(rows)
    assert rows[0]["fid"] == "[50, 0]"


def test_a_road_ends_are_rows_you_can_go_to():
    anchors = [
        dict(
            _road("prior_1", [[0, -100], [0, 100]], "Walker Court"),
            centerline=[[0, -100], [0, 100]],
        )
    ]
    prog = {"ops": [{"ref": "r0", "op": "ends", "args": {"refs": ["prior_1"]}}]}
    rows = run_program([], anchors, prog)["results"]["r0"]
    assert [r["fid"] for r in rows] == ["end_0", "end_1"]
    assert [r["xy"] for r in rows] == [[0.0, -100.0], [0.0, 100.0]]


def test_a_footprint_has_no_ends_and_says_so():
    import pytest

    from maps.query.op_io import ProgramError

    anchors = [
        {
            "fid": "prior_9",
            "xy": [0.0, 0.0],
            "name": "A Shed",
            "mask": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            "centerline": None,
            "cells": [],
            "search_text": "a shed",
        }
    ]
    with pytest.raises(ProgramError, match=r"prior_9 \(A Shed\) is a footprint"):
        run_program(
            [], anchors, {"ops": [{"ref": "r0", "op": "ends", "args": {"refs": ["prior_9"]}}]}
        )


def test_a_footprint_with_a_mask_axis_is_still_refused():
    # Every baked row carries its mask's principal axis as a centerline; a
    # building's axis ends are no road's ends.
    shed = {
        "fid": "prior_9",
        "xy": [5.0, 5.0],
        "name": "A Shed",
        "mask": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        "centerline": [[0.0, 5.0], [10.0, 5.0]],
        "cells": [],
        "search_text": "a shed",
    }
    road = dict(
        _road("prior_1", [[0, -100], [0, 100]], "Walker Court"),
        centerline=[[0, -100], [0, 100]],
    )
    prog = {"ops": [{"ref": "r0", "op": "ends", "args": {"refs": ["prior_9"]}}]}
    with pytest.raises(
        ProgramError, match=r"prior_9 \(A Shed\) is a footprint, and a footprint has no ends"
    ):
        run_program([], [shed, road], prog)
    # A set holding both reads the road's ends.
    prog["ops"][0]["args"]["refs"] = [["prior_9", "prior_1"]]
    rows = run_program([], [shed, road], prog)["results"]["r0"]
    assert [r["name"] for r in rows] == ["Walker Court end0", "Walker Court end1"]
    assert "1 footprint skipped (prior_9 (A Shed)): a footprint has no ends" in rows.notes[-1]


@pytest.mark.parametrize(
    "ref",
    ["[960, 960]", {"ref": "[960, 960]", "as": "x[0]"}, {"ref": [960.0, 960.0], "as": "x[0]"}],
)
def test_a_crossing_is_a_point_with_no_ends(ref):
    prog = {"ops": [{"ref": "r0", "op": "ends", "args": {"refs": [ref]}}]}
    with pytest.raises(
        ProgramError, match=r"is a point \(a crossing, a road.s end or an \[x, y\]\)"
    ):
        run_program([], [], prog)


def test_a_sibling_segment_that_crosses_is_named():
    import pytest

    from maps.query.op_io import ProgramError

    anchors = [
        _road("prior_1", [[0, 0], [100, 0]], "Airway Boulevard"),
        _road("prior_2", [[0, 84], [100, 84]], "Lime Avenue"),
        _road("prior_3", [[50, -10], [50, 10]], "Lime Avenue"),
    ]
    with pytest.raises(ProgramError) as e:
        _run(anchors, "prior_1", "prior_2")
    msg = str(e.value)
    assert "do not meet HERE" in msg, msg
    assert "prior_3" in msg and "touch at [50, 0]" in msg, msg


def test_a_crossing_set_anchors_a_distance_column():
    anchors = [
        _road("prior_1", [[0, 0], [100, 0]], "A Road"),
        _road("prior_2", [[50, -50], [50, 50]], "B Road"),
    ]
    cands = [{"fid": "d0", "xy": [60.0, 10.0], "class": "house", "search_text": "house"}]
    prog = {
        "ops": [
            {"ref": "r0", "op": "crossings", "args": {"refs": ["prior_1", "prior_2"]}},
            {"ref": "r1", "op": "find", "args": {"query": "house"}},
            {
                "ref": "r2",
                "op": "compare",
                "args": {"items": "r1", "refs": [{"ref": ["r0"], "as": "inter"}]},
            },
        ]
    }
    out = run_program(cands, anchors, prog)["results"]["r2"]
    rows = list(out)
    assert rows, out
    assert any("distance to inter" in r for r in rows), rows[0]


def test_a_look_list_crops_each_element():
    """look([a, b]) asks for one crop token PER element."""
    from maps.mission.tests.fakes import acts_of_block

    for fids in (["prior_1", "prior_2"], ["prior_1", "prior_2", "prior_3"]):
        assert acts_of_block(f"look({fids!r})\n") == [("crops", None, fids, None)]
    # ...and a genuine [x, y] point still rides as ONE point token.
    assert acts_of_block("look([100, 200])\n") == [("crops", None, "[100, 200]", None)]
    # A NAME in the list grounds as it does alone, to its best match.
    from maps.mission.tests.fakes import calls_of_block

    assert calls_of_block('look(["Long Road", "prior_2"])\n') == [
        ("ground", {"Long Road": "crops"}),
        ("crops", None, ["<Long Road>", "prior_2"], None),
    ]


def test_a_look_list_refuses_a_set_and_names_its_rows():
    """look([cars, "car_2"]) holds a whole set in a list: refused, naming the
    spelling that crops its rows."""
    from maps.mission.tests.fakes import record_block

    calls, out = record_block('cars = find("car")\nlook([cars, "car_2"])\n')
    assert not [c for c in calls if c[0] == "crops"], calls
    assert "look([cars[0], ...])" in str(out["error"]), out


@pytest.mark.parametrize(
    "block, name",
    [
        ('cars = find("car")\nlook([cars[0:2], "car_2"])\n', "cars[0:2]"),
        ('cars = find("car")\ntop = cars[0:2]\nlook([top, "car_2"])\n', "top"),
    ],
)
def test_a_look_list_refuses_a_slice_naming_it_as_spelled(block, name):
    """A slice in a look list is refused under the model's own spelling --
    its bound name, else the slice -- before any window is analyzed."""
    from maps.mission.tests.fakes import record_block

    calls, out = record_block(block)
    assert [c[0] for c in calls].count("analyze") == 1, calls  # the find
    assert not [c for c in calls if c[0] == "crops"], calls
    assert f"look([{name}[0], ...])" in str(out["error"]), out


@pytest.mark.parametrize(
    "op, query, says",
    [
        ("crossings", "car", "'roads' names no geometry"),
        ("crossings", "qqq", "'roads' holds 0 rows here; there is nothing to cross."),
        ("ends", "qqq", "'roads' holds 0 rows here; there is nothing to take the ends of."),
    ],
)
def test_a_refused_ref_is_quoted_by_the_models_label(op, query, says):
    # A set with no geometry (or no rows) is refused under the label the
    # model knows it by, never as the executor's {ref, as} element.
    roads = {"ref": ["r0"], "as": "roads"}
    prog = {
        "ops": [
            {"ref": "r0", "op": "find", "args": {"query": query}},
            {"ref": "r1", "op": op, "args": {"refs": [roads, roads]}},
        ]
    }
    with pytest.raises(ProgramError) as e:
        run_program([{"fid": "c1", "class": "car", "search_text": "car"}], [], prog)
    assert str(e.value).startswith(says), e.value


@pytest.mark.parametrize("op", ["crossings", "ends"])
def test_a_fid_off_the_surface_is_refused_as_naming_no_row(op):
    # A string that is neither a surface fid nor a handle names nothing:
    # the op's own refusal answers, never the unknown-ref error.
    anchors = [dict(_road("prior_1", [[0, -100], [0, 100]]), centerline=[[0, -100], [0, 100]])]
    prog = {"ops": [{"ref": "r0", "op": op, "args": {"refs": ["prior_999", "prior_1"]}}]}
    with pytest.raises(ProgramError) as e:
        run_program([], anchors, prog)
    assert str(e.value).startswith("'prior_999' names no"), e.value
    assert str(e.value).endswith(". Say a fid off a served row."), e.value


def _ends_block(code):
    from maps.mission.tests.fakes import ROADS, drive, priors, run, tool_results
    from maps.query.surface import build_surface_from_graph

    cands, anchors = build_surface_from_graph({**priors(), **ROADS}.items(), (0.0, 0.0), 1.0)
    _r, fr, _f, _d = drive(
        [[run('e = ends_of("Long Road")\n' + code)], "done"], cands=cands + anchors
    )
    return tool_results(fr)[0]


@pytest.mark.parametrize("code", ["show_map(around=e[1])\n", "show_map(around=end_1)\n"])
def test_a_computed_row_is_a_place_a_map_centres_on_and_badges(code):
    out = _ends_block(code)
    assert out["map"]["maps"] == 1 and "skipped" not in out["map"], out


def test_a_typed_terminus_is_a_place_to_stand():
    out = _ends_block('stand(at=end_1, facing="north")\n')
    assert "error" not in out and out["stance"]["staged"][0]["at_xy"] == [120.0, 100.0], out


def test_a_crossing_is_named_by_its_row_whichever_side_holds_it():
    # With Bent Lane's row first, this crossing computes to x 25.92, where
    # the bundle has 25.93; the row it is on still names it.
    anchors = [
        _road("prior_1", [[57, 80], [74, 96], [23, 62], [13, 15]], "Long Road"),
        _road("prior_2", [[49, 32], [10, 86], [75, 1]], "Bent Lane"),
        _road("prior_3", [[81.8, 95.9], [24.5, 95.2], [46.8, 5.0], [1.5, 9.6]], "Hook Way"),
        _road("prior_4", [[99.3, 83.1], [96.6, 93.0]], "Stub Court"),
    ]
    rows = _run(anchors, "prior_1", ["prior_2", "prior_3", "prior_4"])
    assert all(" / " not in r["name"] for r in rows), rows


_POINT = "(a crossing, a road's end or an [x, y]), and a point has no ends -- ends_of reads a road."


@pytest.mark.parametrize(
    "code, said",
    [
        ('x = intersection("Long Road", "Cross Street")\nends_of(x)\n', "'x' is a point "),
        ("ends_of(e)\n", "'e' holds points "),
        ("ends_of(e[0])\n", "'end_0' is a point "),
        ("ends_of([960, 960])\n", "'[960, 960]' is a point "),
        ('ends_of("Long Road x Cross Street")\n', "'Long Road x Cross Street' is a point "),
    ],
)
def test_a_point_or_a_set_of_points_is_refused_as_having_no_ends(code, said):
    err = _ends_block(code)["error"]
    assert said + _POINT in err, err


def test_a_crossing_in_a_mixed_set_is_skipped_with_a_note():
    out = _ends_block('ends_of(find("Long Road") + intersection("Long Road", "Cross Street"))\n')
    (env,) = [v for k, v in out["result"]["results"].items() if k != "e"]
    assert env["len"] == 2, env
    assert "1 point skipped ([100, 100]): a point has no ends." in (env["hint"]), env


def _env_of(out, key):
    return out["result"]["results"][key]


def test_two_roads_ends_carry_fids_unique_to_the_mission():
    out = _ends_block('c = ends_of("Cross Street")\ne\nc\n')
    assert _env_of(out, "e")["fids"] == ["end_0", "end_1"], out
    assert _env_of(out, "c")["fids"] == ["end_2", "end_3"], out


@pytest.mark.parametrize(
    "code, fids",
    [
        # a later call over the same road names the same rows
        ('c = ends_of("Cross Street")\nd = ends_of(find("Long Road"))\nd\n', ["end_0", "end_1"]),
        # a stance moves the surface, so c re-derives, and its rows keep their fids
        ('c = ends_of("Cross Street")\nstand(at=[0, 0])\nc\n', ["end_2", "end_3"]),
    ],
)
def test_a_road_ends_keep_their_fids_across_calls_and_replays(code, fids):
    out = _ends_block(code)
    assert list(out["result"]["results"].values())[-1]["fids"] == fids, out


def test_a_typed_terminus_resolves_to_its_own_road_in_every_call():
    from maps.mission.tests.fakes import ROADS, drive, priors, run, tool_results
    from maps.query.surface import build_surface_from_graph

    cands, anchors = build_surface_from_graph({**priors(), **ROADS}.items(), (0.0, 0.0), 1.0)
    code = (
        'e = ends_of("Long Road")\nc = ends_of("Cross Street")\n'
        'print(distance("end_0", [80, 100]), distance("end_2", [100, 80]))\ngo_to("end_2")\n'
    )
    _r, fr, flown, _d = drive([[run(code)], "done"], cands=cands + anchors)
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert out["printed"] == "0.0 0.0", out
    assert out["flown"]["flown"][0]["target"] == "[100, 80]", out


@pytest.mark.parametrize(
    "code, said",
    [
        ('ends_of("[100, 100]")\n', "'[100, 100]' is a point "),
        ("ends_of([100, 100])\n", "'[100, 100]' is a point "),
    ],
)
def test_a_point_is_quoted_as_its_xy_token(code, said):
    err = _ends_block(code)["error"]
    assert said + _POINT in err, err


def test_a_crossing_refusal_quotes_a_point_as_its_xy_token():
    err = _ends_block('intersection([100, 100], "Cross Street")\n')["error"]
    assert err.startswith("'[100, 100]' names no geometry"), err


def test_a_crossing_name_with_a_side_naming_no_row_names_no_row():
    err = _ends_block('ends_of("Long Road x Nowhere")\n')["error"]
    assert err.startswith("'Long Road x Nowhere' names no row on this surface."), err


def test_a_written_list_with_a_point_skips_the_point():
    out = _ends_block('ends_of([find("Long Road")[0], [100, 100]])\n')
    assert "error" not in out, out
    (env,) = [v for k, v in out["result"]["results"].items() if k != "e"]
    assert env["len"] == 2 and "1 point skipped ([100, 100]): a point has no ends." in env["hint"]


def test_a_written_list_of_points_is_refused_as_written():
    err = _ends_block("ends_of([[100, 100], [80, 100]])\n")["error"]
    assert err.startswith("'[[100, 100], [80, 100]]' holds points " + _POINT), err
