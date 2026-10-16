"""The signed-relation kwargs, end to end: kwarg -> ops -> served column."""

import numpy as np
import pytest

from maps.search.analyze_scene import geom_program as gp
from maps.search.parse_compile.code_surface import CodeError
from maps.search.tests.fakes import acts_of_block, ops_of_block, raising_ops_of_block


def _road(x, y0=-400.0, y1=400.0, n=9):
    return [[[float(x), float(y)] for y in np.linspace(y0, y1, n)]]


def _surface():
    prior = [
        {"fid": "prior_A", "xy": [0.0, 0.0], "name": "Orchid Drive",
         "mask": _road(0.0), "centerline": None, "cells": [],
         "search_text": "orchid drive"},
        {"fid": "prior_B", "xy": [600.0, 0.0], "name": "North Loop Boulevard",
         "mask": _road(600.0), "centerline": None, "cells": [],
         "search_text": "north loop boulevard"},
    ]
    cands = [{"fid": f"car_{i}", "xy": [float(x), 10.0], "class": "car",
              "mask": [[[float(x), 10.0]]], "centerline": None, "cells": [],
              "search_text": "car"}
             for i, x in enumerate((-120.0, 200.0, 900.0))]
    return cands, prior


def test_the_compiled_key_is_the_key_the_executor_writes():
    """The two halves, asserted against each other and not separately."""
    ops = ops_of_block(
        'r1 = find("car", east_of="Orchid Drive", '
        'west_of="North Loop Boulevard")\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    score = next(o for o in ops if o["op"] == "score_by")
    # The CUT names the side columns; the SORT names proximity.
    cuts = [o for o in ops if o["op"] == "filter_items"]
    assert [o["args"]["key"] for o in cuts] == [
        "east of orchid_drive", "west of north_loop_boulevard"]
    assert [e["key"] for e in score["args"]["emphasis"]] == [
        "distance to orchid_drive", "distance to north_loop_boulevard"]
    assert all(e["w"] == -1 for e in score["args"]["emphasis"])
    asked = [o["args"]["key"] for o in cuts]
    assert [c["dir"] for c in cmp_op["args"]["cardinals"]] == ["east", "west"]
    # ...and the executor really writes those exact strings.
    cands, prior = _surface()
    res = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_A", "as": "Orchid Drive"},
                     {"ref": "prior_B", "as": "North Loop Boulevard"}],
            "cardinals": cmp_op["args"]["cardinals"]}}]})
    written = set()
    for row in res["results"]["r1"]:
        written |= {k for k in row if k.startswith(("east of ", "west of "))}
    assert written == set(asked)


def test_the_column_is_signed_and_orders_by_side():
    cands, prior = _surface()
    res = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_A", "as": "Orchid Drive"},
                     {"ref": "prior_B", "as": "North Loop Boulevard"}],
            "cardinals": [{"dir": "east", "ref": "orchid_drive"},
                          {"dir": "west", "ref": "north_loop_boulevard"}]}}]})
    by = {r["fid"]: r for r in res["results"]["r1"]}
    # west of Orchid: NOT east of it, and the magnitude is the clearance.
    assert by["car_0"]["east of orchid_drive"] == -120.0
    # between the two roads: satisfies BOTH clauses.
    assert by["car_1"]["east of orchid_drive"] > 0
    assert by["car_1"]["west of north_loop_boulevard"] > 0
    # past the far road: east of the near one, NOT west of the far one.
    assert by["car_2"]["east of orchid_drive"] > 0
    assert by["car_2"]["west of north_loop_boulevard"] < 0


def test_a_cardinal_CUTS_the_wrong_side_and_KEEPS_the_unplaced():
    """Violating and unplaceable are different facts."""
    cuts = [o for o in ops_of_block(
        'r1 = find("car", east_of="Orchid Drive")\n')
        if o["op"] == "filter_items"]
    assert [o["args"]["op"] for o in cuts] == ["on_side"]

    _, prior = _surface()
    cands = [{"fid": f, "xy": list(xy), "class": "car",
              "mask": [[list(xy)]], "centerline": None, "cells": [],
              "search_text": "car"}
             for f, xy in (("east_ok", [200.0, 10.0]),
                           ("west_violates", [-200.0, 10.0]),
                           ("no_side", [50.0, 900.0]))]
    res = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_A", "as": "Orchid Drive"}],
            "cardinals": [{"dir": "east", "ref": "orchid_drive"}]}},
        {"ref": "r2", "op": "filter_items", "args": {
            "items": "r1", "key": "east of orchid_drive", "op": "on_side"}}]})
    assert {r["fid"] for r in res["results"]["r1"]} == {
        "east_ok", "west_violates", "no_side"}
    assert {r["fid"] for r in res["results"]["r2"]} == {"east_ok", "no_side"}


def test_conjoined_cardinals_chain_into_an_AND():
    """Two cuts in series is the conjunction a weighted sum cannot express."""
    cuts = [o for o in ops_of_block(
        'r1 = find("car", east_of="Orchid Drive", '
        'west_of="North Loop Boulevard")\n') if o["op"] == "filter_items"]
    assert [o["args"]["key"] for o in cuts] == [
        "east of orchid_drive", "west of north_loop_boulevard"]
    cands, prior = _surface()
    ops = [{"ref": "r1", "op": "compare", "args": {
        "items": [c["fid"] for c in cands],
        "refs": [{"ref": "prior_A", "as": "Orchid Drive"},
                 {"ref": "prior_B", "as": "North Loop Boulevard"}],
        "cardinals": [{"dir": "east", "ref": "orchid_drive"},
                      {"dir": "west", "ref": "north_loop_boulevard"}]}}]
    src = "r1"
    for i, k in enumerate(("east of orchid_drive",
                           "west of north_loop_boulevard")):
        ops.append({"ref": f"c{i}", "op": "filter_items",
                    "args": {"items": src, "key": k, "op": "on_side"}})
        src = f"c{i}"
    res = gp.run_program(cands, prior, {"ops": ops})
    # car_0 is west of Orchid; car_2 is past the far road. Only the one
    # BETWEEN them satisfies both.
    assert {r["fid"] for r in res["results"][src]} == {"car_1"}


def test_an_abstained_cardinal_is_ABSENT_never_zero():
    """A zero would read as 'exactly on the line' and sort like a real measurement; the
    missing cell is score_by's neutral."""
    ew = [[[float(x), 0.0] for x in np.linspace(-400, 400, 9)]]
    prior = [{"fid": "prior_E", "xy": [0.0, 0.0], "name": "Quezon Avenue",
              "mask": ew, "centerline": None, "cells": [],
              "search_text": "quezon avenue"}]
    cands = [{"fid": "car_0", "xy": [0.0, 200.0], "class": "car",
              "mask": [[[0.0, 200.0]]], "centerline": None, "cells": [],
              "search_text": "car"}]
    res = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": ["car_0"],
            "refs": [{"ref": "prior_E", "as": "Quezon Avenue"}],
            "cardinals": [{"dir": "east", "ref": "quezon_avenue"},
                          {"dir": "north", "ref": "quezon_avenue"}]}}]})
    row = res["results"]["r1"][0]
    assert "east of quezon_avenue" not in row      # no side to name
    assert row["north of quezon_avenue"] == 200.0  # the orthogonal one reads


def test_a_corner_is_two_cuts_against_one_crossing():

    ops = raising_ops_of_block(
        'corner = intersection("86th Street", "Jacaranda Avenue")\n'
        'r1 = find("house", north_of=corner, east_of=corner)\n')
    kinds = [o["op"] for o in ops]
    assert kinds.count("crossings") == 1, kinds
    cmp_op = next(o for o in ops if o["op"] == "compare")
    # the anchor is named twice and rides as ONE ref -- a second would make
    # the executor suffix its label and the cut key would name no column
    assert len(cmp_op["args"]["refs"]) == 1, cmp_op["args"]["refs"]
    assert {c["dir"] for c in cmp_op["args"]["cardinals"]} == {"north", "east"}
    cuts = [o for o in ops if o["op"] == "filter_items"]
    assert len(cuts) == 2, cuts
    assert {o["args"]["key"] for o in cuts} == {"north of corner",
                                                "east of corner"}
    assert all(o["args"]["op"] == "on_side" for o in cuts), cuts


def test_a_filter_naming_one_anchor_twice_sends_ONE_ref():
    # two cardinal cuts against one place are one compare against it: a
    # second copy is suffixed by the executor and serves a duplicate set of
    # columns under the suffixed label
    ops = ops_of_block('r1 = find("house")\n'
                       'r2 = r1.filter(east_of="Jacaranda Avenue", '
                       'north_of="Jacaranda Avenue")\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    assert len(cmp_op["args"]["refs"]) == 1, cmp_op["args"]["refs"]
    assert [c["dir"] for c in cmp_op["args"]["cardinals"]] == ["east", "north"]


@pytest.mark.parametrize("kw, pair", [("southwest_of", ("south_of", "west_of")),
                                      ("northeast_of", ("north_of", "east_of"))])
def test_a_diagonal_filter_kwarg_refuses_and_names_the_conjunction(kw, pair):
    with pytest.raises(CodeError) as e:
        raising_ops_of_block(f'r1 = find("car")\nr2 = r1.filter({kw}="Orchid Drive")\n')
    assert f"{pair[0]}=<place>, {pair[1]}=<place>" in str(e.value), str(e.value)
    with pytest.raises(CodeError) as e:
        raising_ops_of_block('r1 = find("car")\nr2 = r1.filter(bogus="Orchid Drive")\n')
    assert "diagonal" not in str(e.value), str(e.value)


def test_an_unknown_filter_kwarg_is_refused_never_dropped():
    from maps.search.parse_compile.python_surface import (
        CodeError,
        MissionPython,
        Row,
        RowList,
    )
    from maps.search.tests.fakes import quiet_env

    mp = MissionPython(quiet_env())
    rows = RowList([Row({"fid": "detected_1", "xy": [0.0, 0.0]}, mp=mp)])

    # the diagonal: refused, and the refusal names the two cuts that work
    with pytest.raises(CodeError) as e:
        rows.filter(southeast_of="Poppy Boulevard")
    msg = str(e.value)
    assert "south_of=" in msg and "east_of=" in msg, msg

    # any other unknown name: refused by name rather than swallowed
    with pytest.raises(CodeError) as e:
        rows.filter(bogus=1)
    assert "bogus" in str(e.value), str(e.value)

    # ...and a real cardinal still passes through untouched
    rows.filter(north_of="Poppy Boulevard")


def test_a_set_anchor_places_a_cardinal_and_abstains_when_it_straddles():
    """A SET is one place for a cardinal too, and the band rule decides."""
    _cands, prior = _surface()

    def _run(x):
        cands = [{"fid": "car_9", "xy": [float(x), 10.0], "class": "car",
                  "mask": [[[float(x), 10.0]]], "centerline": None,
                  "cells": [], "search_text": "car"}]
        return gp.run_program(cands, prior, {"ops": [
            {"ref": "r1", "op": "compare", "args": {
                "items": ["car_9"],
                "refs": [{"ref": ["prior_A", "prior_B"], "as": "roads"}],
                "cardinals": [{"dir": "east", "ref": "roads"}]}}]})

    # prior_A sits at x=0, prior_B at x=600
    straddling = _run(200.0)["results"]["r1"][0]
    assert "distance to roads" in straddling, straddling
    assert "east of roads" not in straddling, straddling

    past_both = _run(900.0)["results"]["r1"][0]
    assert past_both.get("east of roads") == 300.0, past_both


def test_an_anchor_may_be_a_SET_of_objects():
    ops = ops_of_block(
        'r1 = find("car", east_of=["prior_A", "prior_B"])\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    ref = cmp_op["args"]["refs"][0]
    assert ref["ref"] == ["prior_A", "prior_B"], ref
    assert cmp_op["args"]["cardinals"] == [
        {"dir": "east", "ref": ref["as"].lower()}], cmp_op["args"]
    cuts = [o for o in ops if o["op"] == "filter_items"]
    assert len(cuts) == 1, cuts
    # the compiled key IS the key the compare writes, by construction --
    # both sides call ref_of
    assert cuts[0]["args"]["key"] == f"east of {ref['as'].lower()}", cuts

    # [one thing] is the thing: 18 of the observed refusals
    one = ops_of_block('r1 = find("car", east_of=["prior_A"])\n')
    bare = ops_of_block('r1 = find("car", east_of="prior_A")\n')
    assert ([o["args"] for o in one if o["op"] == "filter_items"]
            == [o["args"] for o in bare if o["op"] == "filter_items"])
    # a wrapped point unwraps to the point: 13 more
    assert ops_of_block('r1 = find("car", east_of=[[120, 40]])\n')

    # a set holds PLACES; a flat run of numbers is NAMED, never guessed at
    with pytest.raises(CodeError, match="bare point"):
        ops_of_block('r1 = find("car", east_of=[10, 20, 30, 40])\n')
    with pytest.raises(CodeError, match="empty list"):
        ops_of_block('r1 = find("car", east_of=[])\n')


def test_near_and_a_cardinal_fuse_into_ONE_sort():
    """Two sorts in a row is not two constraints."""
    ops = ops_of_block(
        'r1 = find("house", nearest="Taylor", south_of="Viburnum Avenue")\n')
    sorts = [o for o in ops if o["op"] == "score_by"]
    assert len(sorts) == 1, "two sorts means the first was discarded"
    assert [(e["key"], e["w"]) for e in sorts[0]["args"]["emphasis"]] == [
        ("distance to taylor", -1), ("distance to viburnum_avenue", -1)]


def test_one_anchor_named_twice_is_ONE_ref():
    ops = ops_of_block('r1 = find("house", east_of="Jacaranda Avenue", '
                       'north_of="Jacaranda Avenue")\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    assert len(cmp_op["args"]["refs"]) == 1
    assert [c["dir"] for c in cmp_op["args"]["cardinals"]] == ["east", "north"]
    # ...and ONE proximity term, not one per direction: both cardinals name
    # the same anchor, so its distance column is emphasised once.
    keys = [e["key"] for o in ops if o["op"] == "score_by"
            for e in o["args"]["emphasis"]]
    assert keys == ["distance to jacaranda_avenue"]
    assert [o["args"]["key"] for o in ops if o["op"] == "filter_items"] == [
        "east of jacaranda_avenue", "north of jacaranda_avenue"]
    # ...and the executor answers each direction INDEPENDENTLY on that one
    # ref, abstaining per row where that direction has no side to give. On a
    # north-south road: a row off to the east reads east and abstains north
    # (no road abreast of it in y); a row past the road's north end reads
    # north and abstains east. Both are the honest answer, and a corner cue
    # naming two directions of one road gets whichever actually applies.
    _, prior = _surface()
    cands = [{"fid": "east_of_it", "xy": [200.0, 10.0], "class": "car",
              "mask": [[[200.0, 10.0]]], "centerline": None, "cells": [],
              "search_text": "car"},
             {"fid": "past_north_end", "xy": [50.0, 600.0], "class": "car",
              "mask": [[[50.0, 600.0]]], "centerline": None, "cells": [],
              "search_text": "car"}]
    res = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_A", "as": "Jacaranda Avenue"}],
            "cardinals": [{"dir": "east", "ref": "jacaranda_avenue"},
                          {"dir": "north", "ref": "jacaranda_avenue"}]}}]})
    by = {r["fid"]: r for r in res["results"]["r1"]}
    assert by["east_of_it"]["east of jacaranda_avenue"] == 200.0
    assert "north of jacaranda_avenue" not in by["east_of_it"]
    assert by["past_north_end"]["north of jacaranda_avenue"] == 200.0
    assert "east of jacaranda_avenue" not in by["past_north_end"]


def test_a_POINT_anchor_computes_a_cardinal():
    """A point mask has vertices and no segments."""
    prior = [{"fid": "prior_P", "xy": [0.0, 0.0], "name": "corner",
              "mask": [[[0.0, 0.0]]], "centerline": None, "cells": [],
              "search_text": "corner"}]
    cands = [{"fid": f"c{i}", "xy": [float(x), float(y)], "class": "car",
              "mask": [[[float(x), float(y)]]], "centerline": None,
              "cells": [], "search_text": "car"}
             for i, (x, y) in enumerate([(10.0, 200.0), (10.0, -200.0),
                                         (400.0, 50.0)])]
    res = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_P", "as": "5284 700"}],
            "cardinals": [{"dir": "north", "ref": "5284_700"}]}}]})
    by = {r["fid"]: r for r in res["results"]["r1"]}
    assert by["c0"]["north of 5284_700"] == 200.0
    assert by["c1"]["north of 5284_700"] == -200.0
    # ...and a row with nothing abreast of it still abstains honestly.
    assert "north of 5284_700" not in by["c2"]


def test_within_still_cuts_ahead_of_every_ordering():
    ops = [o["op"] for o in ops_of_block(
        'r1 = find("house", within="Mill Lane", east_of="Orchid Drive")\n')]
    assert ops.index("filter_items") < ops.index("score_by")


def test_the_side_cut_STATES_the_three_way_outcome():
    """The fact only exists at the moment of the cut."""
    _, prior = _surface()
    cands = [{"fid": f, "xy": list(xy), "class": "car",
              "mask": [[list(xy)]], "centerline": None, "cells": [],
              "search_text": "car"}
             for f, xy in (("east_ok", [200.0, 10.0]),
                           ("west_violates", [-200.0, 10.0]),
                           ("no_side", [50.0, 900.0]))]
    ops = [{"ref": "r1", "op": "compare", "args": {
        "items": [c["fid"] for c in cands],
        "refs": [{"ref": "prior_A", "as": "Orchid Drive"}],
        "cardinals": [{"dir": "east", "ref": "orchid_drive"}]}},
        {"ref": "r2", "op": "filter_items", "args": {
            "items": "r1", "key": "east of orchid_drive", "op": "on_side"}}]
    st = gp.run_program(cands, prior, {"ops": ops})["results"]["r2"].cut_stats
    assert st["on_side"] == 1 and st["other_side_dropped"] == 1
    assert st["no_side_kept"] == 1
    assert "nothing_on_that_side" not in st

    # ...and when NOTHING surveyed is on that side, it says where the answer
    # has to come from instead. This is the case that fired zero times.
    west = [{"fid": "w", "xy": [-300.0, 10.0], "class": "car",
             "mask": [[[-300.0, 10.0]]], "centerline": None, "cells": [],
             "search_text": "car"}]
    ops[0]["args"]["items"] = ["w"]
    st2 = gp.run_program(west, prior, {"ops": ops})["results"]["r2"].cut_stats
    assert st2["on_side"] == 0 and st2["other_side_dropped"] == 1
    assert "SEEN" in st2["nothing_on_that_side"]


def test_a_quoted_xy_token_is_a_POINT_in_relating_slots_and_refused_in_moving_ones():
    """A crossing's fid IS its [x, y] token, and the model quotes a fid the way it
    quotes every other one."""
    assert acts_of_block('show_map(around="[3895, 1890]")\n') == [
        ("map", "[3895, 1890]")]
    cmp_op = next(o for o in ops_of_block(
        'r1 = find("house", nearest="[1058, 3418]")\n') if o["op"] == "compare")
    assert cmp_op["args"]["refs"][0]["ref"] == [1058.0, 3418.0]
    # ...and the no-coordinate law still stands for the calls that MOVE,
    # with the remedy naming the row rather than a spelling.
    with pytest.raises(CodeError, match="not a coordinate"):
        acts_of_block('go_to("[2707, 5217]")\n')
    with pytest.raises(CodeError, match="intersection"):
        acts_of_block('go_to("[2707, 5217]")\n')


def test_near_takes_TWO_places_and_fuses_both_distances():
    ops = ops_of_block(
        'r1 = find("house", nearest=["Lime Avenue", "85th Street"])\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    score = next(o for o in ops if o["op"] == "score_by")
    assert len(cmp_op["args"]["refs"]) == 2, cmp_op
    assert [(e["key"], e["w"]) for e in score["args"]["emphasis"]] == [
        ("distance to lime_avenue", -1), ("distance to 85th_street", -1)]
    # one place still compiles to one key
    score = next(o for o in ops_of_block(
        'r1 = find("house", nearest="Lime Avenue")\n') if o["op"] == "score_by")
    assert len(score["args"]["emphasis"]) == 1
    # ...and a bare point is still a POINT, not a two-place list
    from maps.search.parse_compile.code_surface import _is_place_list
    assert not _is_place_list([4630, 2965])
    assert _is_place_list(["A Road", "B St"])


def test_facing_is_a_CUT_and_the_word_rides_as_data():
    ops = raising_ops_of_block(
        'r1 = find("house")\n'
        'r2 = r1.filter(facing="Orchid Drive", direction="south")\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    cut = next(o for o in ops if o["op"] == "filter_items")
    # no cardinals: a facing cut asks for no new column
    assert "cardinals" not in cmp_op["args"], cmp_op["args"]
    # the compiled key IS the key the compare writes -- both go through
    # _label_of, the executor's own sanitizer
    assert cut["args"]["key"] == "facing orchid_drive", cut
    # in_or_blank over the winds that HOLD south, never an equality
    assert cut["args"]["op"] == "in_or_blank", cut
    assert set(cut["args"]["value"]) == {"south", "southeast", "southwest"}

    # the four cardinals, and each expands to the three winds holding it
    for w in ("north", "south", "east", "west"):
        o = next(x for x in raising_ops_of_block(
            f'r1 = find("house")\nr2 = r1.filter(facing="Orchid Drive", '
            f'direction="{w}")\n') if x["op"] == "filter_items")
        assert len(o["args"]["value"]) == 3, o
        assert w in o["args"]["value"], o

    # each half names the other, and a non-compass word names the four
    with pytest.raises(CodeError, match="needs the way it faces"):
        raising_ops_of_block(
            'r1 = find("house")\nr2 = r1.filter(facing="Orchid Drive")\n')
    with pytest.raises(CodeError, match="a cut needs what it faces"):
        raising_ops_of_block(
            'r1 = find("house")\nr2 = r1.filter(direction="south")\n')
    with pytest.raises(CodeError, match="not a compass direction"):
        raising_ops_of_block(
            'r1 = find("house")\n'
            'r2 = r1.filter(facing="Orchid Drive", direction="up")\n')
    # ONE relation per filter, as every other cut here
    with pytest.raises(CodeError, match="ONE relation at a time"):
        raising_ops_of_block(
            'r1 = find("house")\nr2 = r1.filter(facing="Orchid Drive", '
            'direction="south", near="Orchid Drive", dwithin=25)\n')


def test_a_facing_cut_keeps_the_row_it_could_not_measure():
    """`facing <ref>` is ABSENT at contact -- no min-gap vector to take a bearing from
    -- and a house TOUCHING the road it fronts is the common case for "the
    south-facing house on Viburnum Avenue"."""
    prior = [{"fid": "prior_R", "xy": [0.0, 0.0], "name": "Orchid Drive",
              "mask": _road(0.0), "centerline": None, "cells": [],
              "search_text": "orchid drive"}]

    def house(fid, x):
        return {"fid": fid, "xy": [float(x), 100.0], "class": "house",
                "mask": [[[x - 5.0, 95.0], [x + 5.0, 95.0],
                          [x + 5.0, 105.0], [x - 5.0, 105.0]]],
                "centerline": None, "cells": [], "search_text": "house"}

    # east of the road faces west; west of it faces east; ON it abstains
    cands = [house("h_east", 200.0), house("h_west", -200.0),
             house("h_touch", 0.0)]
    prog = {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_R", "as": "orchid_drive"}]}},
        {"ref": "r2", "op": "filter_items", "args": {
            "items": "r1", "key": "facing orchid_drive",
            "op": "in_or_blank",
            "value": ["west", "northwest", "southwest"]}}]}
    out = gp.run_program(cands, prior, prog)
    seen = {r["fid"]: r.get("facing orchid_drive")
            for r in out["results"]["r1"]}
    assert seen["h_east"] == "west", seen
    assert seen["h_west"] == "east", seen
    assert seen["h_touch"] is None, seen          # contact: unmeasured

    kept = {r["fid"] for r in out["results"]["r2"]}
    assert "h_east" in kept                        # reads one of them
    assert "h_west" not in kept                    # reads another, dropped
    assert "h_touch" in kept                       # unmeasured, KEPT

    # and the cut STATES the three-way outcome, or a cut that kept only
    # blanks would read as a cut that found matches
    st = out["results"]["r2"].cut_stats
    assert st["cut"] == "3->2", st
    assert st["reads_one_of_them"] == 1, st
    assert st["reads_another_word_dropped"] == 1, st
    assert st["unmeasured_kept"] == 1, st
    assert st["counts_as"] == ["west", "northwest", "southwest"], st
    assert "nothing_reads_those_words" not in st, st

    # the all-blank case is the one that would otherwise read as a hit
    only_touch = [house("h_touch", 0.0)]
    prog["ops"][0]["args"]["items"] = ["h_touch"]
    st2 = gp.run_program(only_touch, prior, prog)["results"]["r2"].cut_stats
    assert st2["reads_one_of_them"] == 0 and st2["unmeasured_kept"] == 1, st2
    assert "nothing_reads_those_words" in st2, st2


def test_a_direction_COMPOSES_so_southwest_faces_south_and_west():
    from maps.search.parse_compile.code_surface import (
        _winds_holding, split_diagonal)

    # derived from the wind's spelling, never tabulated
    assert split_diagonal("southwest") == ("south", "west")
    assert split_diagonal("south") is None
    for c in ("north", "south", "east", "west"):
        holds = _winds_holding(c)
        assert len(holds) == 3, (c, holds)
        assert c in holds
        assert all(c in (split_diagonal(w) or (w,)) for w in holds), (c, holds)
    assert set(_winds_holding("south")) == {"south", "southeast", "southwest"}
    # ...and no wind is claimed by an axis it does not hold
    assert "north" not in _winds_holding("south")
    assert "southwest" not in _winds_holding("north")

    # a diagonal is REFUSED and the refusal writes the chain
    with pytest.raises(CodeError, match="faces south AND west"):
        raising_ops_of_block(
            'r1 = find("house")\n'
            'r2 = r1.filter(facing="Orchid Drive", direction="southwest")\n')

    # end to end: a SOUTHWEST-facing row survives direction="south",
    # and chaining west narrows to the diagonal
    prior = [{"fid": "prior_R", "xy": [0.0, 0.0], "name": "Orchid Drive",
              "mask": _road(0.0), "centerline": None, "cells": [],
              "search_text": "orchid drive"}]
    cands = [{"fid": f, "xy": list(xy), "class": "house",
              "mask": [[list(xy)]], "centerline": None, "cells": [],
              "search_text": "house"}
             # the road runs N-S at x=0 and ENDS at y=400, so a house
             # past that end sees its nearest cell to the SOUTHWEST while
             # one abreast of the road sees it due WEST
             for f, xy in (("h_sw", [300.0, 600.0]),
                           ("h_w", [300.0, 0.0]))]
    out = gp.run_program(cands, prior, {"ops": [
        {"ref": "r1", "op": "compare", "args": {
            "items": [c["fid"] for c in cands],
            "refs": [{"ref": "prior_R", "as": "orchid_drive"}]}}]})
    words = {r["fid"]: r.get("facing orchid_drive")
             for r in out["results"]["r1"]}
    assert words["h_sw"] == "southwest", words
    assert words["h_w"] == "west", words


def test_a_direction_ORDERS_globally_and_takes_no_anchor():
    from maps.search.parse_compile.python_surface import Row, RowList, _direction_key

    rows = RowList([Row({"fid": f, "xy": list(xy), "class": "pool"}, None)
                    for f, xy in (("p_n", [0.0, 500.0]),
                                  ("p_s", [0.0, -500.0]),
                                  ("p_e", [500.0, 0.0]),
                                  ("p_w", [-500.0, 0.0]))])
    # rank 0 is the FURTHEST that way, on all eight winds
    assert [r.fid for r in rows.order(direction="north")][0] == "p_n"
    assert [r.fid for r in rows.order(direction="south")][0] == "p_s"
    assert [r.fid for r in rows.order(direction="east")][0] == "p_e"
    assert [r.fid for r in rows.order(direction="west")][0] == "p_w"
    # a diagonal is BOTH terms at once, off the one compass table --
    # nothing here re-derives a per-word axis
    ne = [r.fid for r in rows.order(direction="northeast")]
    assert set(ne[:2]) == {"p_n", "p_e"}, ne
    assert set(ne[2:]) == {"p_s", "p_w"}, ne

    # no anchor: naming one is the other question, and says so
    with pytest.raises(CodeError, match="not both"):
        rows.order(direction="north", nearest="Orchid Drive")
    # eight winds, and a typo names them rather than sorting by nothing
    for bad in ("up", "norht", "left"):
        with pytest.raises(CodeError, match="not a compass word"):
            rows.order(direction=bad)
    # ...and the refusal points at the RELATIVE question, since that is
    # what a model reaching for an anchor here actually wants
    with pytest.raises(CodeError, match=r"order\(nearest=<place>\)"):
        rows.order(direction="up")

    # the key is shared, so a handle and a list cannot drift apart
    assert _direction_key("north")(rows[0]) == -500.0


def test_order_takes_reverse_on_a_key_and_names_the_word_on_a_direction():
    """`sorted()`'s own kwarg, because that is what the model writes."""
    from maps.search.parse_compile.python_surface import Row, RowList

    rows = RowList([Row({"fid": f, "xy": list(xy), "class": "pool"}, None)
                    for f, xy in (("a", [0.0, 0.0]), ("b", [10.0, 0.0]),
                                  ("c", [20.0, 0.0]))])
    fwd = [r.fid for r in rows.order(lambda r: r.xy[0])]
    rev = [r.fid for r in rows.order(lambda r: r.xy[0], reverse=True)]
    assert fwd == ["a", "b", "c"], fwd
    assert rev == ["c", "b", "a"], rev

    # direction= names the opposite word instead of taking a flag
    for w, opp in (("north", "south"), ("east", "west")):
        with pytest.raises(CodeError, match=f'direction="{opp}"'):
            rows.order(direction=w, reverse=True)
    # ...and the plain direction sort still works
    assert [r.fid for r in rows.order(direction="east")] == ["c", "b", "a"]
