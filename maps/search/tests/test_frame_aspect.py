"""The compare ``frame`` aspect: view COORDINATES (SV-Frame).

The frame turns the cue's view words into coordinates: vx_/vy_ signed view
meters in a declared or viewer frame, along_/off_ on a polyline chart. The
arrangement machinery (rows/ordinals/flanks/groups) is DELETED — zero
filter/score ops in 1,013 audited episodes; an ordinal cue is a sort +
index read (score_by on the axis column, then rN[k]).

Sign-locks pinned here: the G1 polyline handedness (+off = right of travel
= the CLOCKWISE side of the tangent in the y-up world frame), the v2
point-ref rotation signs, and the receipt-content facts including the
weak-side premortem disclosure.
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    ProgramError,
    _BroadRows,
    _EmptyFindMiss,
    run_program,
)

# A 2-row x 3-col grid of cars north of a road: rows run along x, near-road
# row at y~10, far row at y~22 (the y jitter is a small linear ramp).
_GRID = [
    {"fid": "c00", "xy": [10.0, 10.0], "class": "car", "color": "red"},
    {"fid": "c01", "xy": [15.0, 10.4], "class": "car", "color": "blue"},
    {"fid": "c02", "xy": [20.0, 10.8], "class": "car", "color": "white"},
    {"fid": "c10", "xy": [10.0, 22.0], "class": "car", "color": "red"},
    {"fid": "c11", "xy": [15.0, 22.4], "class": "car", "color": "blue"},
    {"fid": "c12", "xy": [20.0, 22.8], "class": "car", "color": "white"},
]
_ANCHORS = [
    # a genuinely LINEAR road (centerline, no boundary) SOUTH of the grid
    {"fid": "road_1", "name": "Main Street", "xy": [50.0, 0.0], "class": "road",
     "centerline": [[0.0, 0.0], [100.0, 0.0]]},
    # a point landmark south of the grid (the R-point declared-view reference)
    {"fid": "shed_1", "name": "Shed", "xy": [15.0, -30.0], "class": "building"},
]

# The retired arrangement vocabulary: NO framed row may carry any of these.
_RETIRED_COLS = ("row_lo", "row_hi", "row_size", "col_lo", "col_hi",
                 "ord_lr", "ord_rl", "nb_left", "nb_right", "grp", "grp_size")


def _frame_grid(frame, cands=None, anchors=None):
    out = run_program([dict(c) for c in (cands or _GRID)],
                      [dict(a) for a in (anchors or _ANCHORS)], {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": {"items": "$r0", "frame": frame}},
    ]})
    res = out["results"]["r1"]
    return {r["fid"]: r for r in res}, res


# --------------------------------------------------------------------------- #
# R-point declared view: v2 rotation signs                                     #
# --------------------------------------------------------------------------- #


def test_point_reference_vx_vy_signs():
    # side:'bottom' puts the shed at the BOTTOM of the view: +vy points away
    # from it (far row positive), +vx = cw(up) = east-ward here. The v2
    # proper-rotation sign-lock, re-pinned.
    rows, _ = _frame_grid({"reference": "shed_1", "side": "bottom"})
    for fid in ("c00", "c01", "c02"):        # near-shed row -> below center
        assert rows[fid]["behind shed_1"] > 0, rows[fid]
    for fid in ("c10", "c11", "c12"):        # far row -> above center
        assert rows[fid]["ahead of shed_1"] > 0, rows[fid]
    for fid in ("c00", "c10"):               # left column
        assert rows[fid]["left of shed_1"] > 2, rows[fid]
    for fid in ("c02", "c12"):               # right column
        assert rows[fid]["right of shed_1"] > 2, rows[fid]


def test_frame_attaches_no_arrangement_columns():
    # The deleted family stays deleted: a framed pool carries the view
    # coordinates and NONE of the retired arrangement names, whichever form
    # framed it.
    for frame in ({"reference": "shed_1", "side": "bottom"},
                  {"reference": "auto"},
                  {"along": "road_1"}):
        rows, _ = _frame_grid(frame)
        for r in rows.values():
            assert not any(k in r for k in _RETIRED_COLS), (frame, r)
            assert any(k.startswith(("left of ", "right of ", "ahead of ",
                                     "behind ", "along ")) for k in r), (frame, r)


# --------------------------------------------------------------------------- #
# viewer frame {from, toward}: look rays, between, compass, ref projection    #
# --------------------------------------------------------------------------- #


_GATES = [
    {"fid": "pa", "name": "Gate A", "xy": [0.0, 0.0], "class": "building"},
    {"fid": "pb", "name": "Gate B", "xy": [40.0, 0.0], "class": "building"},
]
_VCANDS = [
    {"fid": "mid", "xy": [20.0, 1.0], "class": "car"},     # midway A->B
    {"fid": "off", "xy": [20.0, 30.0], "class": "car"},    # far off-axis
    {"fid": "beyond", "xy": [60.0, 2.0], "class": "car"},  # past B
]


def _viewer(frame, refs=None):
    args = {"items": "$r0", "frame": frame}
    if refs:
        args["refs"] = refs
    out = run_program([dict(c) for c in _VCANDS], [dict(a) for a in _GATES],
                      {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": args},
    ]})
    res = out["results"]["r1"]
    return {r["fid"]: r for r in res}, res


def test_viewer_between_is_a_column_read():
    rows, res = _viewer({"from": "pa", "toward": "pb"})
    def side_mag(r):
        return r.get("left of pa_pb", r.get("right of pa_pb", 0.0))
    # look = A->B: 'between' = small side offset AND 'ahead of' in span.
    assert side_mag(rows["mid"]) <= 2.0, rows["mid"]
    assert 0.0 <= rows["mid"]["ahead of pa_pb"] <= 40.0, rows["mid"]
    assert side_mag(rows["off"]) > 20.0, rows["off"]
    assert rows["beyond"]["ahead of pa_pb"] > 20.0, rows["beyond"]



def test_viewer_left_of_ref_is_a_column_comparison():
    # viewer at pa looking at pb (look = +x world): viewer-left is -vx side.
    # 'left of mid' -> candidates with vx < the mid ref's own ref_vx.
    rows, res = _viewer({"from": "pa", "toward": "pb"}, refs=["mid"])
    hdr = res.refs[0]
    assert ("left of view center" in hdr) or ("right of view center" in hdr), hdr
    # Looking east (+x), right = cw(forward) = south (-y), so world +y
    # (north) is viewer-LEFT: 'off' at y=30 sits far on the LEFT side.
    assert rows["off"]["left of pa_pb"] > hdr.get("left of view center", 0.0), (
        rows["off"], hdr)


def test_viewer_compass_toward_north():
    rows, res = _viewer({"toward": "north"}, refs=["pa"])
    # north = +y world: 'off' (y=30) is NORTH of 'mid' (y=1) -> more ahead.
    def ahead(r):
        return r.get("ahead of north", -r.get("behind north", 0.0))
    assert ahead(rows["off"]) > ahead(rows["mid"]), (rows["mid"], rows["off"])
    hdr = res.refs[0]
    assert ("ahead of view center" in hdr) or ("behind view center" in hdr), hdr


def test_viewer_class_blind_slots():
    # a CAR fills 'from' exactly like a building did (masks are masks).
    rows, _ = _viewer({"from": "mid", "toward": "pb"})
    assert all(any(k.startswith(("left of ", "right of ")) for k in r)
               for r in rows.values())


def test_scale_m_is_a_loud_retirement():
    # scale_m tuned the deleted derived-scale arrangement; it errors loud.
    try:
        _frame_grid({"reference": "shed_1", "side": "bottom", "scale_m": 20.0})
        raise AssertionError("scale_m did not error")
    except ProgramError as e:
        assert "retired" in str(e), str(e)


# --------------------------------------------------------------------------- #
# polyline chart: G1 handedness sign-locks, arc-true along                     #
# --------------------------------------------------------------------------- #


def test_polyline_handedness_sign_lock_both_directions_both_sides():
    # G1: off_<f> is signed with + = right of travel = the CLOCKWISE side
    # of the tangent in the y-up world (cross(tangent, to_point) <= 0; the
    # _frame_viewer convention: right = cw(forward)). Traveling +x (east),
    # right = cw(east) = south, so a +y (north) point is LEFT (-off);
    # reverse the polyline (travel west, right = north) and the same point
    # reads RIGHT (+off); mirrored for -y.
    north_side = [{"fid": "n", "xy": [50.0, 10.0], "class": "car"},
                  {"fid": "n2", "xy": [60.0, 10.0], "class": "car"}]
    south_side = [{"fid": "s", "xy": [50.0, -10.0], "class": "car"},
                  {"fid": "s2", "xy": [60.0, -10.0], "class": "car"}]
    fwd = [[0.0, 0.0], [100.0, 0.0]]
    rev = [[100.0, 0.0], [0.0, 0.0]]
    for cands, poly, sign in ((north_side, fwd, -1), (north_side, rev, 1),
                              (south_side, fwd, 1), (south_side, rev, -1)):
        rows, _ = _frame_grid({"along": poly}, cands=cands, anchors=[])
        want = "right of line" if sign > 0 else "left of line"
        other = "left of line" if sign > 0 else "right of line"
        for r in rows.values():
            assert r.get(want, 0) > 0 and other not in r, (poly, r)
            assert r.get("distance to line", 0) > 0, r


def test_polyline_chart_is_arc_true_on_a_u_bend():
    # A U-shaped polyline: the arc order differs from any chord order (the
    # v2 straight-axis read would misorder E before B). along_ follows the
    # ARC — the chord appears nowhere in position math.
    poly = [[0.0, 0.0], [40.0, 0.0], [40.0, 10.0], [0.0, 10.0]]
    cands = [
        {"fid": "A", "xy": [10.0, -2.0], "class": "car"},   # arc ~10
        {"fid": "B", "xy": [30.0, -2.0], "class": "car"},   # arc ~30
        {"fid": "C", "xy": [42.0, 5.0], "class": "car"},    # arc ~45
        {"fid": "D", "xy": [30.0, 12.0], "class": "car"},   # arc ~60
        {"fid": "E", "xy": [10.0, 12.0], "class": "car"},   # arc ~80
    ]
    rows, _ = _frame_grid({"along": poly}, cands=cands, anchors=[])
    alongk = "along line"
    assert rows["A"][alongk] < rows["B"][alongk] < rows["C"][alongk] \
        < rows["D"][alongk] < rows["E"][alongk], rows


def test_polyline_only_reference_with_side_is_loud_valid_forms_first():
    # road_1 realises ONLY a centerline (no areal footprint): a declared
    # side about it is undefined. The rejection LEADS with the valid call
    # forms (the no-side ALONG chart; the areal/point declared view; the
    # viewer form) before stating why side was not read.
    try:
        _frame_grid({"reference": "road_1", "side": "bottom"})
        raise AssertionError("side with a polyline-only reference did not error")
    except ProgramError as e:
        msg = str(e)
        assert msg.startswith("valid forms"), msg
        assert "along" in msg and "toward" in msg, msg
        assert "side" in msg, msg


def test_side_on_an_areal_reference_is_the_declared_view():
    # An AREAL reference (a mask-carrying lot -- buildings/roads with
    # footprints behave identically) given WITH a side is the declared view
    # about its mask's TRUE area centroid, even when it also carries a
    # centerline: the CALL SHAPE (explicit side) dispatches, the data kind
    # only picks the point.
    lot = [{"fid": "lot_9", "name": "South Lot", "xy": [14.0, -31.0],
            "class": "parking",
            "centerline": [[5.0, -30.0], [25.0, -30.0]],
            "mask": [[[5.0, -35.0], [25.0, -35.0],
                      [25.0, -25.0], [5.0, -25.0], [5.0, -35.0]]]}]
    rows, res = _frame_grid({"reference": "lot_9", "side": "bottom"},
                            anchors=_ANCHORS + lot)
    # same view as the shed point ref at the mask centroid (15, -30):
    for fid in ("c00", "c01", "c02"):        # near row -> below center
        assert rows[fid]["behind lot_9"] > 0, rows[fid]
    for fid in ("c10", "c11", "c12"):        # far row -> above center
        assert rows[fid]["ahead of lot_9"] > 0, rows[fid]
    # a declared-view read, not the ALONG chart:
    assert not any(k.startswith("along ") for k in rows["c00"]), rows["c00"]


def test_polyline_ref_echoes_ref_along_ref_off():
    # every point-like compare ref projects into the SAME chart.
    out = run_program([dict(c) for c in _GRID], [dict(a) for a in _ANCHORS],
                      {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0", "refs": ["shed_1"],
                  "frame": {"along": "road_1"}}},
    ]})
    hdr = out["results"]["r1"].refs[0]
    assert "along" in hdr, hdr
    assert ("left of the line" in hdr) or ("right of the line" in hdr), hdr


def test_polyline_coincident_vertices_are_loud():
    try:
        _frame_grid({"reference": [[10.0, 10.0], [10.0, 10.0], [20.0, 10.0]]},
                    anchors=[])
        raise AssertionError("coincident polyline vertices did not error")
    except ProgramError as e:
        assert "coincide" in str(e), str(e)


# --------------------------------------------------------------------------- #
# receipts are stripped: the frame serves columns, no receipt note            #
# --------------------------------------------------------------------------- #


def test_no_frame_receipt_note_is_served():
    _rows, res = _frame_grid({"reference": "shed_1", "side": "bottom"})
    notes = [n for n in (getattr(res, "notes", []) or [])
             if isinstance(n, str)]
    assert not any(n.startswith("frame receipt") for n in notes), notes


# --------------------------------------------------------------------------- #
# deleted ops / aspects redirect loudly to the ONE replacement                 #
# --------------------------------------------------------------------------- #

_CANDS_MIN = [{"fid": "d0", "xy": [1.0, 1.0], "class": "car"}]
_ANCH_MIN = [{"fid": "b_a", "name": "Earl Street", "xy": [9.0, 9.0],
              "class": "building"}]


def _prog_error(ops):
    try:
        run_program([dict(c) for c in _CANDS_MIN],
                    [dict(a) for a in _ANCH_MIN], {"ops": ops})
    except ProgramError as e:
        return str(e)
    raise AssertionError(f"expected ProgramError from {ops!r}")


def test_deleted_ops_teach_their_replacement():
    msg = _prog_error([{"ref": "r0", "op": "rank_by",
                        "args": {"items": "cands", "key": "distance to b_a"}}])
    assert msg.startswith("'rank_by' is not an op."), msg
    assert "score_by" in msg and "emphasis" in msg, msg

    msg = _prog_error([{"ref": "r0", "op": "pick",
                        "args": {"items": "cands", "index": 0}}])
    assert msg.startswith("'pick' is not an op."), msg
    assert "index the register" in msg, msg

    msg = _prog_error([{"ref": "r0", "op": "partition_rows",
                        "args": {"items": "cands"}}])
    assert msg.startswith("'partition_rows' is not an op."), msg
    # the redirect lands on LIVE columns, never the deleted family
    assert "left of/right of" in msg and "row_lo" not in msg, msg


def test_deleted_compare_aspects_redirect_to_frame():
    for dead, args in (
        ("view", {"reference": "b_a", "side": "left"}),
        ("between", {"a": "b_a", "b": "d0"}),
        ("neighbors", {"pool": "cands", "ref": "b_a"}),
    ):
        msg = _prog_error([
            {"ref": "r0", "op": "compare",
             "args": {"items": "cands", dead: args}}])
        assert f"compare '{dead}' is not an aspect." in msg, (dead, msg)
        assert "frame" in msg, (dead, msg)


# --------------------------------------------------------------------------- #
# find{query}: returns THE MATCH SET best-first; empty only on a true no-match #
# --------------------------------------------------------------------------- #


def test_find_query_returns_the_match_set_best_first():
    # A block of streets, so the head noun is COMMON and the proper name is
    # RARE -- the shape of every real map, and the one the fuzz band's IDF
    # weighting reads. `Earl Court` matches on the rare word and is retrieved;
    # the fillers match only on `street`, which 5 of 7 rows spell, and stay
    # out. (Three rows with `earl` on two of them inverted that, and made a
    # proper name the LESS discriminating half of the query.)
    anchors = [
        {"fid": "b_a", "name": "Earl Street", "xy": [0.0, 0.0], "class": "building"},
        {"fid": "b_b", "name": "Earl Court", "xy": [10.0, 0.0], "class": "building"},
        {"fid": "b_c", "name": "King Road", "xy": [20.0, 0.0], "class": "building"},
        {"fid": "b_d", "name": "Main Street", "xy": [30.0, 0.0], "class": "building"},
        {"fid": "b_e", "name": "Oak Street", "xy": [40.0, 0.0], "class": "building"},
        {"fid": "b_f", "name": "Pine Street", "xy": [50.0, 0.0], "class": "building"},
        {"fid": "b_g", "name": "Elm Street", "xy": [60.0, 0.0], "class": "building"},
    ]
    out = run_program([], [dict(a) for a in anchors], {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "prior", "query": "Earl Street"}},
    ]})
    rows = out["results"]["r0"]
    # The result IS the match set (len = match count), best-first, ordered;
    # the sub-threshold row stays out (a query-less find browses it).
    assert isinstance(rows, _BroadRows), type(rows)
    assert getattr(rows, "ordered", False), "a query-matched find is ordered"
    # b_b is "Earl Court" -- it shares the word `earl` with an exactly-named
    # "Earl Street" and is a DIFFERENT street, so once the query has named
    # something the band leaves the match set.
    assert {r["fid"] for r in rows} == {"b_a"}, rows
    assert all(isinstance(r.get("name_match"), (int, float)) for r in rows), rows
    scores = [r["name_match"] for r in rows]
    assert scores == sorted(scores, reverse=True), scores  # best-first
    assert rows[0]["fid"] == "b_a", rows  # the exact name match heads the set


def test_find_query_no_match_returns_empty_miss_affordance():
    out = run_program([], [dict(a) for a in _ANCH_MIN], {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "prior", "query": "zxqv wqzzk"}},
    ]})
    rows = out["results"]["r0"]
    assert isinstance(rows, _EmptyFindMiss), type(rows)
    assert len(rows) == 0, rows


# --------------------------------------------------------------------------- #
# sort notes ride the frame-fed chains too                                     #
# --------------------------------------------------------------------------- #


def test_sign_inversion_note():
    # w:-1 on a match column draws the inversion note.
    grid = [dict(c, color_conf=0.9) for c in _GRID]
    out = run_program(grid, [dict(a) for a in _ANCHORS], {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime", "color": "red"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0",
            "emphasis": [{"key": "color_red", "w": -1}]}},
    ]})
    notes = getattr(out["results"]["r1"], "notes", []) or []
    assert any("INVERTS" in n for n in notes), notes


# --------------------------------------------------------------------------- #
# frame form errors stay loud                                                  #
# --------------------------------------------------------------------------- #


def test_viewer_or_auto_frame_with_side_is_loud():
    for frame in ({"reference": "auto", "side": "left"},
                  {"from": "c00", "toward": "c12", "side": "left"}):
        try:
            _frame_grid(frame)
            raise AssertionError(f"side with {frame!r} did not error")
        except ProgramError as e:
            assert "side" in str(e), str(e)


def test_a_bare_areal_reference_declares_no_served_view():
    # The served frame schema has no 'side' key, so a bare non-'auto'
    # reference is unfillable from the grammar: the rejection names the
    # three live forms and says why side is absent.
    try:
        _frame_grid({"reference": "shed_1"})
        raise AssertionError("point reference without side did not error")
    except ProgramError as e:
        msg = str(e)
        assert "EXACTLY ONE form" in msg and "side" in msg, msg
        assert "reference:'auto'" in msg and "along" in msg, msg


def test_frame_form_exclusivity_is_loud():
    # {} names no form; {reference:<line>, toward} is now the DIRECTED
    # chart (valid), so the exclusivity error pins the empty shape and
    # an areal reference+toward stays loud in its own test below.
    try:
        _frame_grid({})
        raise AssertionError("{} did not error")
    except ProgramError as e:
        assert "EXACTLY ONE form" in str(e), str(e)


# --------------------------------------------------------------------------- #
# Directed chart + line-vantage resolution (the frame-unification cutover)     #
# --------------------------------------------------------------------------- #


def test_undeclared_travel_on_a_baked_road_serves_no_side():
    # Undirected: travel falls back to the stored end0->end1 order, which is
    # an arbitrary property of the bake -- road_travel_intrinsic, measured
    # 0.461 against a 0.470-0.535 chance band (lateral_convention, 926
    # mentions, cluster CIs + block-flip + anchor shuffle). The chart still
    # serves the arc coordinate and the lateral DISTANCE; it just does not
    # name a side the engine had to guess.
    rows_u, _ = _frame_grid({"along": "road_1"})
    r = rows_u["c00"]
    assert "left of road_1" not in r and "right of road_1" not in r, r
    assert r["distance to road_1"] > 0, r
    assert abs(rows_u["c02"]["along road_1"] - 20.0) < 1.0


def test_directed_chart_flips_sign_with_declared_travel():
    # Declaring the travel is the model's own claim, so the pair comes back.
    # Directed toward the shed (projects mid-line nearer end0): travel is
    # west; right = cw(west) = north, so a car at y=+10 reads RIGHT, and
    # along re-zeroes at the far end (c02 at x=20 now reads along 80, not 20).
    rows_d, _ = _frame_grid({"along": "road_1", "toward": "shed_1"})
    assert rows_d["c00"]["right of road_1"] > 0
    assert "left of road_1" not in rows_d["c00"]
    assert abs(rows_d["c02"]["along road_1"] - 80.0) < 1.0


def test_directed_chart_compass_word_orients_net_displacement():
    # 'west' opposes the stored +x run -> reversed; 'east' keeps it.
    # Traveling west, right = cw(west) = north: the +y car reads RIGHT;
    # traveling east, right = south: it reads LEFT.
    rows_w, _ = _frame_grid({"along": "road_1", "toward": "west"})
    rows_e, _ = _frame_grid({"along": "road_1", "toward": "east"})
    assert rows_w["c00"]["right of road_1"] > 0
    assert rows_e["c00"]["left of road_1"] > 0


def test_directed_chart_receipt_names_the_travel():
    _, res = _frame_grid({"along": "road_1", "toward": "shed_1"})
    # the receipt view fragment rides the result header; travel is stated
    import json as _json
    blob = _json.dumps(res)
    assert "along road_1" in blob
    _, res_u = _frame_grid({"along": "road_1"})
    assert "along road_1" in _json.dumps(res_u)


def test_viewer_from_a_road_stands_at_nearest_centerline_point():
    # from:road_1 (centerline y=0, x 0..100), toward:shed_1 at [15,-30]:
    # the vantage is [15,0] (nearest centerline point to the shed), NOT
    # the road row's xy [50,0]. Looking south (toward the shed), the
    # viewer's right hand points WEST (right = cw(south) = -x) from x=15:
    # c00 (x=10) reads +vx, c02 (x=20) reads -vx — with the old centroid
    # vantage both sat one side of x=50 and one sign was WRONG.
    rows, _ = _frame_grid({"toward": "shed_1", "from": "road_1"})
    assert rows["c00"]["right of road_1_shed_1"] > 0, rows["c00"]
    assert rows["c02"]["left of road_1_shed_1"] > 0, rows["c02"]
    # cars sit behind the viewer (the shed is across the road from them)
    assert rows["c00"]["behind road_1_shed_1"] > 0


def test_along_misassemblies_are_loud_and_name_the_form():
    import pytest
    # reference never combines with toward -- the chart's name is along
    with pytest.raises(ProgramError, match="does not combine"):
        _frame_grid({"reference": "shed_1", "toward": "road_1"})
    with pytest.raises(ProgramError, match="does not combine"):
        _frame_grid({"reference": "auto", "toward": "shed_1"})
    with pytest.raises(ProgramError, match="does not combine"):
        _frame_grid({"reference": "road_1", "toward": "shed_1",
                     "side": "right"})
    # along stands alone (no reference/side riders)
    with pytest.raises(ProgramError, match="CHART form on its own"):
        _frame_grid({"along": "road_1", "side": "right"})
    # along needs a line to travel
    with pytest.raises(ProgramError, match="centerline-carrying"):
        _frame_grid({"along": "shed_1"})
    # the retired line-reference chart redirects to its new name
    with pytest.raises(ProgramError, match="along"):
        _frame_grid({"reference": "road_1"})
