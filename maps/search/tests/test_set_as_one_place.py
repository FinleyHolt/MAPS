"""A set in a slot that MEASURES is one place: the gap to its nearest member."""
from __future__ import annotations

import pytest

from maps.query import program as gp
from maps.query.program import compact_results
from maps.search.tests.fakes import drive, ops_of_block, run, tool_results


def _house(fid, x, name=None):
    return {"fid": fid, "xy": [float(x), 0.0], "class": "house",
            "mask": [[[float(x), 0.0]]], "centerline": None, "cells": [],
            "name": name, "search_text": "house"}


def _pool(fid, x, y):
    return {"fid": fid, "xy": [float(x), float(y)], "class": "pool",
            "mask": [[[float(x), float(y)]]], "centerline": None, "cells": [],
            "search_text": "pool"}


def test_the_house_with_a_pool_in_the_backyard_is_one_line():
    """The whole point, end to end: compile it and run it."""
    ops = ops_of_block('pools = find("pool")\n'
                       'houses = find("house", nearest=pools)\n')
    cmp_, = [o for o in ops if o["op"] == "compare"]
    ref, = cmp_["args"]["refs"]
    assert ref["as"] == "pools", cmp_
    assert isinstance(ref["ref"], list) and len(ref["ref"]) == 1, cmp_
    sort, = [o for o in ops if o["op"] == "score_by"]
    assert [(e["key"], e["w"]) for e in sort["args"]["emphasis"]] == [
        ("distance to pools", -1)], sort

    out = gp.run_program(
        [_pool("pool_1", 100.0, 8.0)],
        [_house("h_a", 0.0, "A"), _house("h_b", 100.0, "B"),
         _house("h_c", 400.0, "C")],
        {"ops": ops})
    ranked = [r["fid"] for r in out["results"][ops[-1]["ref"]]]
    assert ranked[0] == "h_b", ranked
    by = {r["fid"]: r["distance to pools"]
          for r in out["results"][ops[-1]["ref"]]}
    assert by["h_b"] == pytest.approx(8.0, abs=0.1), by
    # ONE column, not one per pool: that is what makes it survive a real set.
    assert not [k for k in out["results"][ops[-1]["ref"]][0]
                if k.startswith("distance to pool_")], by


def test_the_union_is_the_NEAREST_member_not_a_centroid():
    """Two pools far apart: each house measures against its own closest."""
    ops = ops_of_block('pools = find("pool")\n'
                       'houses = find("house", nearest=pools)\n')
    out = gp.run_program(
        [_pool("pool_1", 0.0, 5.0), _pool("pool_2", 1000.0, 5.0)],
        [_house("h_a", 0.0), _house("h_b", 1000.0), _house("h_c", 500.0)],
        {"ops": ops})
    by = {r["fid"]: r["distance to pools"]
          for r in out["results"][ops[-1]["ref"]]}
    # a centroid at x=500 would put h_c first and both real neighbours last
    assert by["h_a"] == pytest.approx(5.0, abs=0.1), by
    assert by["h_b"] == pytest.approx(5.0, abs=0.1), by
    assert by["h_c"] > 400.0, by


def test_a_set_cuts_as_well_as_orders():
    """`within=` keeps what TOUCHES the set -- houses that hold a pool."""
    ops = ops_of_block('pools = find("pool")\n'
                       'houses = find("house").filter(within=pools)\n')
    cut, = [o for o in ops if o["op"] == "filter_items"]
    assert cut["args"]["key"] == "distance to pools", cut
    assert (cut["args"]["op"], cut["args"]["value"]) == ("le", 0), cut


def test_an_empty_set_fails_loud_and_names_the_handle_and_the_fix():
    ops = ops_of_block('pools = find("pool")\n'
                       'houses = find("house", nearest=pools)\n')
    with pytest.raises(gp.ProgramError) as exc:
        gp.run_program([], [_house("prior_1", 0.0)], {"ops": ops})
    msg = str(exc.value)
    assert "pools is empty" in msg, msg          # the model's own name
    assert "r0" not in msg, msg
    assert "look_for" in msg and "flight" in msg, msg


def test_an_unseen_row_is_not_buried_by_a_colour_ordering():
    """`color=` must not become a coverage ordering wearing a colour's name."""
    seen_right = _house("seen_white", 0.0)
    seen_right.update(color="white", color_conf=0.9)
    seen_wrong = _house("seen_red", 10.0)
    seen_wrong.update(color="red", color_conf=0.9)
    ops = ops_of_block('r1 = find("house", color="white")\n')
    out = gp.run_program([seen_right, seen_wrong],
                         [_house("unseen", 20.0)], {"ops": ops})
    assert [r["fid"] for r in out["results"][ops[-1]["ref"]]] == [
        "seen_white", "unseen", "seen_red"]


# --------------------------------------------------------------------------- #
# Through the REAL loop, not just the executor                                #
# --------------------------------------------------------------------------- #


def test_the_whole_chain_survives_the_reasoner_loop():
    """`near=<set>` end to end: compiled, dispatched, served."""

    cands = [
        {"fid": "house_1", "xy": [0.0, 0.0], "class": "house",
         "search_text": "house"},
        {"fid": "house_2", "xy": [100.0, 0.0], "class": "house",
         "search_text": "house"},
        {"fid": "house_3", "xy": [400.0, 0.0], "class": "house",
         "search_text": "house"},
        {"fid": "pool_1", "xy": [100.0, 8.0], "class": "pool",
         "search_text": "pool"},
    ]
    _reply, fr, _flights, _d = drive([
        [("run", {"code": 'look_for("pool")\n'
                          'pools = find("pool")\n'
                          'houses = find("house", nearest=pools)\n'
                          'houses\n'})],
        "done",
    ], cands=cands)
    out = tool_results(fr)[0]
    assert "error" not in out, out
    rows = next(iter(out["result"]["results"].values()))["rows"]
    fids = [ln.strip().strip("|").split("|")[0].strip()
            for ln in rows.splitlines()[2:] if ln.strip()]
    assert fids[0] == "house_2", rows       # the one with the pool behind it
    assert "distance to pools" in rows, rows


def test_an_unseen_colour_drops_its_term_instead_of_killing_the_block():
    """Nothing carries a colour until a view has passed over it."""
    road = {"fid": "road_1", "name": "Mill Lane", "xy": [0.0, 0.0],
            "class": "road", "centerline": [[0.0, -50.0], [0.0, 50.0]],
            "mask": [[[0.0, float(y)] for y in range(-50, 51, 10)]],
            "cells": [], "search_text": "mill lane"}
    ops = [
        {"ref": "r0", "op": "find",
         "args": {"query": "house"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["road_1"]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1", "emphasis": [
             {"key": "distance to road_1", "w": -1},
             {"key": "color_white", "w": 1}]}},
    ]
    out = gp.run_program([], [_house("h_far", 60.0), _house("h_near", 1.0),
                              road], {"ops": ops})
    ranked = [r["fid"] for r in out["results"]["r2"]
              if str(r["fid"]).startswith("h_")]
    assert ranked == ["h_near", "h_far"], ranked   # distance still ranks
    hint = compact_results(out["results"])["r2"].get("hint") or ""
    assert "carries a perceived colour yet" in hint, hint  # NEVER silent


def test_a_dropped_colour_says_the_set_has_none_not_the_surface():
    # A car on the surface was seen in red; the buildings ranked were not.
    seen = {"fid": "c1", "class": "car", "color": "red", "color_conf": 0.9,
            "xy": [0.0, 0.0], "search_text": "car"}
    ops = [{"ref": "r0", "op": "find", "args": {"query": "building"}},
           {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
               {"key": "color_red", "w": 1.0},
               {"key": "footprint_area_m2", "w": 1.0}]}}]
    buildings = [{"fid": f"b{i}", "class": "building", "xy": [10.0 * i, 0.0],
                  "search_text": "building", "footprint_area_m2": 50.0 * i}
                 for i in (1, 2)]
    notes = gp.run_program([seen], buildings, {"ops": ops})["results"]["r1"].notes
    assert notes == [
        "'color_red' ranked nothing here: no row in this set carries a "
        "perceived colour yet (a row gets one once a flight has passed over "
        "it); the other keys still rank."], notes


def test_a_colour_alone_with_nothing_seen_is_still_a_loud_refusal():
    ops = ops_of_block('r1 = find("house", color="white")\n')
    with pytest.raises(gp.ProgramError, match="nothing orders by"):
        gp.run_program([], [_house("h_a", 0.0), _house("h_b", 10.0)],
                       {"ops": ops})


def test_a_union_over_a_whole_class_is_capped_rather_than_slow():
    ops = ops_of_block('everything = find("house")\n'
                       'r1 = find("house", nearest=everything)\n')
    many = [_house(f"h{i:04d}", float(i)) for i in range(gp._UNION_MAX_MEMBERS + 5)]
    with pytest.raises(gp.ProgramError,
                       match=r"folds at most.*first \(within=, a cardinal"):
        gp.run_program([], many, {"ops": ops})


def test_a_fid_less_member_is_measured_by_its_own_point():
    """compare over [x, y] items yields rows without a fid; as a union member
    such a row is a place like any other: its point."""
    out = gp.run_program(
        [], [_house("h_a", 0.0), _house("h_b", 100.0)],
        {"ops": [
            {"ref": "r1", "op": "compare",
             "args": {"items": [[10.0, 0.0]], "refs": ["h_a"]}},
            {"ref": "r2", "op": "compare",
             "args": {"items": ["h_a", "h_b"],
                      "refs": [{"ref": ["r1"], "as": "spot"}]}}]})
    by = {r["fid"]: r["distance to spot"] for r in out["results"]["r2"]}
    assert by == {"h_a": pytest.approx(10.0), "h_b": pytest.approx(90.0)}, by


def _fids_printed(code):
    _reply, fr, _f, _d = drive([[run(code + "print(sorted(r.fid for r in hit))\n")],
                                "done"])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    return out["printed"]


def test_a_python_set_of_rows_is_its_union_like_the_handle():
    """set(rows) in within= or near= is the union of those rows, as the handle
    is: the same rows survive the cut."""
    for kw in ('within=set(cars)', 'near=set(cars), dwithin=15',
               'near=[r for r in cars], dwithin=15'):
        handle = kw.replace("set(cars)", "cars").replace(
            "[r for r in cars]", "cars")
        got = _fids_printed(f'cars = find("car")\nhit = find("", {kw})\n')
        want = _fids_printed(f'cars = find("car")\nhit = find("", {handle})\n')
        assert got == want, (kw, got, want)


def test_a_point_union_member_is_measured_by_its_own_point():
    """A fid-less row (a terminus) in a union rides as its [x, y]."""
    out = gp.run_program(
        [], [_house("h_a", 0.0), _house("h_b", 100.0), _house("h_c", 300.0)],
        {"ops": [{"ref": "r1", "op": "compare",
                  "args": {"items": ["h_a", "h_b"],
                           "refs": [{"ref": ["h_c", [10.0, 0.0]],
                                     "as": "spots"}]}}]})
    by = {r["fid"]: r["distance to spots"] for r in out["results"]["r1"]}
    assert by == {"h_a": pytest.approx(10.0), "h_b": pytest.approx(90.0)}, by


def test_a_huge_user_value_in_a_refusal_stays_short():
    with pytest.raises(gp.ProgramError) as e:
        gp.run_program([], [_house("h_a", 0.0)], {"ops": [
            {"ref": "r1", "op": "compare",
             "args": {"items": ["h_a"], "refs": [["x" * 100_000, 1]]}}]})
    assert "x" * 200 not in str(e.value)
