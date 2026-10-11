"""The signed-relation kwargs, end to end: kwarg -> ops -> served column.

The failure this guards is the one the arm already paid for twice. Bug #2 of
the cutover was `near=[literal point]` refused because "the compare labelled
the column one way and key_for another, so score_by asked for a column that
was never written". A cardinal has the same two halves and the same way to
drift, so the compiled key and the executor's written key are asserted to be
the SAME STRING here rather than each checked alone.
"""

import numpy as np
import pytest

from maps.search.analyze_scene import geom_program as gp
from maps.search.parse_compile.code_surface import CodeError
from maps.search.tests.fakes import ops_of_block, plan_of_block, raising_ops_of_block


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


def test_every_cardinal_is_a_find_kwarg():
    import inspect

    from maps.search.parse_compile.python_surface import MissionPython

    params = inspect.signature(MissionPython._call_find).parameters
    for d in ("east", "west", "north", "south"):
        assert f"{d}_of" in params


def test_the_compiled_key_is_the_key_the_executor_writes():
    """The two halves, asserted against each other and not separately."""
    ops = ops_of_block(
        'r1 = find("car", east_of="Orchid Drive", '
        'west_of="North Loop Boulevard")\n')
    cmp_op = next(o for o in ops if o["op"] == "compare")
    score = next(o for o in ops if o["op"] == "score_by")
    # The CUT names the side columns; the SORT names proximity. A cardinal
    # is not a ranking term -- ranking by signed magnitude put rank 0 at the
    # far edge of the map (median 1,220 m from the anchor on 25713026) while
    # 194 of this corpus's 208 cardinal cues mean ADJACENT.
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
    """Violating and unplaceable are different facts.

    Ordering alone handed the model the least-wrong row at rank 0 and it
    read like a hit: find("parking", south_of="California City Middle
    School") ranked a lot 1,552 m NORTH of the school first, across 163 rows
    none of which was south of it (25713025). A row the geometry PLACES on
    the other side is not a candidate.

    But `gt 0` would drop the unplaced rows with the violating ones, and on
    that same case all 19 real parking rows in the bake ABSTAINED -- the cut
    would have thrown away exactly what was still in play. `on_side` keeps
    them.
    """
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
    """A zero would read as 'exactly on the line' and sort like a real
    measurement; the missing cell is score_by's neutral."""
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


def test_a_diagonal_refuses_rather_than_answering():
    cands, prior = _surface()
    with pytest.raises(gp.ProgramError, match="diagonal"):
        gp.run_program(cands, prior, {"ops": [
            {"ref": "r1", "op": "compare", "args": {
                "items": ["car_0"],
                "refs": [{"ref": "prior_A", "as": "Orchid Drive"}],
                "cardinals": [{"dir": "northeast", "ref": "orchid_drive"}]}}]})


def test_a_corner_is_two_cuts_against_one_crossing():
    """The shape 95 of the 1,023 cues ask for, and the legend now shows.

    "The house on the northeast corner of 86th Street and Jacaranda Avenue"
    is a QUADRANT OF A POINT, where north-of AND east-of is exact rather
    than an approximation -- which is why the geometry offers four winds and
    the surface spells a diagonal as its two halves. 95 cues read "<diag>
    corner of A and B" and 17 more are a diagonal of a crossing or a street's
    end; only ~14 in the whole corpus ask a diagonal of an extended feature.

    The compile is pinned because the legend now teaches it: one crossings
    op, ONE deduped ref for the anchor named twice, and the two cuts chained
    so each says what it dropped.
    """

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


def test_a_diagonal_kwarg_refuses_and_names_the_conjunction():
    """A `<diag>_of=` mistake earns the COMPOSITION, not just the four names.

    Listing north_of=/south_of=/east_of=/west_of= leaves the model to invent
    the conjunction on its own, and the conjunction is what 95 of the 1,023
    cues ask for ("the house on the southwest corner of A and B"). On a
    quadrant of a POINT it is exact rather than an approximation, which is
    why cardinal_side offers four winds and the refusal names the pairing.
    """
    from maps.search.analyze_scene.geom_compute import CARDINALS, COMPASS_UNIT
    from maps.search.parse_compile.python_surface import (
        _CARDINAL_KWARGS,
        _diagonal_remedy,
        _split_diagonal,
    )

    # the kwarg set IS the declared four-subset -- no fifth spelling
    assert _CARDINAL_KWARGS == {f"{w}_of" for w in CARDINALS}
    diagonals = set(COMPASS_UNIT) - set(CARDINALS)
    assert len(diagonals) == 4, sorted(diagonals)     # vacuity floor
    for w in diagonals:
        a, b = _split_diagonal(w)
        assert a in CARDINALS and b in CARDINALS, w
        msg = _diagonal_remedy([f"{w}_of"])
        assert f"{a}_of=" in msg and f"{b}_of=" in msg, (w, msg)
    assert _diagonal_remedy(["bogus"]) == ""          # only for diagonals


def test_an_unknown_filter_kwarg_is_refused_never_dropped():
    """`filter(southeast_of=X)` used to cut NOTHING and say nothing.

    Row.filter and RowList.filter forward **kw straight to _rows_filter,
    whose **cards swallowed any name the compiler's _CARDINALS comprehension
    then ignored -- so the model asserted a constraint the call never
    applied, in silence. That is the exact shape of `toward=`: built,
    accepted, compiled to nothing, deleted. code_surface's own law is that a
    kwarg outside the set is REFUSED, never ignored, and _rows_filter is the
    one funnel all three filter paths reach.
    """
    from maps.search.parse_compile.python_surface import (
        CodeError,
        MissionEnv,
        MissionPython,
        Row,
        RowList,
    )

    mp = MissionPython(MissionEnv(
        run_plan=lambda plan, said: {}, rows_of_ref=lambda ref: [],
        plan_rn=[0], discourse={}, minted=[],
        say_err=lambda out, msg: {**out, "error": msg},
        row_of_fid={}.get))
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
    """A SET is one place for a cardinal too, and the band rule decides.

    Both halves are run, not assumed, because they are indistinguishable
    from the compile side. A car BETWEEN the two roads is not east of the
    pair, and the column is ABSENT rather than 0.0 -- unmeasured, never
    refuted. A car past both is east of the union by its nearest member's
    offset.

    This replaces an assertion that read the straddle case as a structural
    "a union writes no cardinal column". It is not structural: the same
    program with the item moved DOES write the column, so a surface law
    built on the absent case would have rested on one geometry.
    """
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
    """156 blocks on 112 episodes died handing a LIST to a relating slot.

    `near=[A, B]` had worked the whole time, so the model was reading a
    slot that takes a set and generalising correctly; the cardinal and
    containment slots refused the same shape with the executor's
    `compare refs element ... must be a bare fid string`, the top
    structural refusal on the surface over six 500-episode roots. A set
    folds to its UNION: one ref, one column, one cut. Two cuts against two
    anchors is `.filter(east_of=A).filter(east_of=B)`, which the surface
    already spells and this must not silently become.
    """
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
    """Two sorts in a row is not two constraints.

    The second score_by re-ranks the rows the first ordered, so the first
    contributes NOTHING: `find("house", nearest="Taylor", south_of="Viburnum
    Avenue")` ordered by the side alone and dropped the proximity, with no
    error. Live on 25713014 before this fused them.
    """
    ops = ops_of_block(
        'r1 = find("house", nearest="Taylor", south_of="Viburnum Avenue")\n')
    sorts = [o for o in ops if o["op"] == "score_by"]
    assert len(sorts) == 1, "two sorts means the first was discarded"
    assert [(e["key"], e["w"]) for e in sorts[0]["args"]["emphasis"]] == [
        ("distance to taylor", -1), ("distance to viburnum_avenue", -1)]


def test_one_anchor_named_twice_is_ONE_ref():
    """The executor suffixes a repeated ref's label (`x_2`), and the key
    built here would still name the first -- score_by would then ask for a
    column that was never written. Naming one CORNER does exactly this."""
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
    """A point mask has vertices and no segments.

    "North of that corner" is the simplest cardinal there is, and without a
    vertex fallback the column was never written: the model wrote
    `find("house", north_of=[5284, 700])` and got a loud "nothing orders by
    'north of 5284_700'" nine times on 25713024.
    """
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


def test_the_legend_states_the_capability_and_its_limit():
    from maps.search.parse_compile.prompts import CODE_DOC
    assert "east_of=" in CODE_DOC and "north_of=" in CODE_DOC
    # The served text must match what the executor DOES. It said "Nothing is
    # dropped" for a whole run after the cut shipped.
    assert "Nothing is dropped" not in CODE_DOC
    assert "NEAREST first" in CODE_DOC
    # ...and a blank cell means unknown, never "no".
    assert "blank" in CODE_DOC and "never" in CODE_DOC


def test_the_side_cut_STATES_the_three_way_outcome():
    """The fact only exists at the moment of the cut.

    A note written in the reasoner loop could never fire: the loop sees only
    SURVIVORS, which are positive by construction, and when nothing survives
    the list is empty. Measured 0 firings in 194 episodes -- the
    present-but-always-empty failure this repo has a rule about. The count
    lives with the op that does the dropping.
    """
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
    """A crossing's fid IS its [x, y] token, and the model quotes a fid the
    way it quotes every other one.

    Grounded as a NAME it drew `rule N: nothing named "[2707, 5217]" is on
    this surface` -- the wrong wrongness twice over. For a RELATING slot the
    point is welcome (the surface's own text says so); for a MOVING one the
    objection is not that nothing bears that name, it is that a coordinate is
    not a thing. 19 quoted-xy refs on 25713032, 10 of them still refused
    before this.
    """
    plan = plan_of_block('show_map(around="[3895, 1890]")\n')
    assert [s.get("do") for s in plan["steps"]] == ["map"]
    assert plan["steps"][0]["around"] == ("xy", [3895.0, 1890.0]), plan
    cmp_op = next(o for o in ops_of_block(
        'r1 = find("house", nearest="[1058, 3418]")\n') if o["op"] == "compare")
    assert cmp_op["args"]["refs"][0]["ref"] == [1058.0, 3418.0]
    # ...and the no-coordinate law still stands for the calls that MOVE,
    # with the remedy naming the row rather than a spelling.
    with pytest.raises(CodeError, match="not a coordinate"):
        plan_of_block('go_to("[2707, 5217]")\n')
    with pytest.raises(CodeError, match="intersection"):
        plan_of_block('go_to("[2707, 5217]")\n')


def test_near_takes_TWO_places_and_fuses_both_distances():
    """The multi-anchor gap this arm closes.

    Ranking rows by closeness to BOTH features a cue names takes top-24 from
    68.7% to 92.9% on the 281 episodes whose cue names two -- +6.65 points
    corpus-wide, held out across three anchor splits. The SENTENCE surface
    has said this since before any of these runs ("find <thing> near <place>
    near <place>.") and the executor already fuses the keys in score_by;
    only the code surface could not spell it.
    """
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
    """The largest directional family in the corpus, 325 of 1,023 cues.

    `facing <place>` has been a served compass WORD since 8e00c408, read
    30 times over 6 of 29 episodes as
    `houses.filter(lambda r: r.get("facing viburnum_avenue") == "south")`.
    That is a cut spelled as a predicate over a column, so this is the cut
    itself -- no new column, only the filter_items that reads the one
    every compare already stamps.

    The word rides in direction= as DATA. Putting it in the kwarg name
    (`faces_south=`) reached 3 of 45 episodes before 8e00c408 deleted it.
    FOUR cardinals, because a direction COMPOSES -- see the composition
    test below.
    """
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
    """`facing <ref>` is ABSENT at contact -- no min-gap vector to take a
    bearing from -- and a house TOUCHING the road it fronts is the common
    case for "the south-facing house on Viburnum Avenue". Plain `eq`
    compares None to the word and drops it, refuting a row that was never
    measured. Same law `on_side` keeps for the signed columns.
    """
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
    """The rule, and the ground truth that priced it.

    A thing facing southwest faces south AND west. Measured on the
    corpus's own targets: asking the served column for an EXACT word
    answered the cue on 12.1% of corner-facing cues -- a house sits
    DIAGONALLY from the crossing it is on the corner of, so the column
    reads `southwest` where the cue says `south` -- against 87.9% under
    composition. The named-road cell went 87.8% -> 89.9% at the same
    time, because a cue asking a diagonal now matches a measured one. No
    cell loses.

    The cut is LOOSER by construction, three winds of eight per cardinal
    against one, and chaining the two cardinals of a diagonal takes the
    precision back -- which is the surface's own law that filters compose.
    """
    from maps.search.parse_compile.code_surface import _wind_parts, _winds_holding

    # derived from the wind's spelling, never tabulated
    assert _wind_parts("southwest") == {"south", "west"}
    assert _wind_parts("south") == {"south"}
    for c in ("north", "south", "east", "west"):
        holds = _winds_holding(c)
        assert len(holds) == 3, (c, holds)
        assert c in holds
        assert all(c in _wind_parts(w) for w in holds), (c, holds)
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
    """The axis the whole surface splits on.

    "The northernmost pool" is a fact about the MAP FRAME, not a relation
    to any object, so ordering by a direction takes no anchor. Every
    relation to a place is a CUT and lives on filter(). Treating the
    global fact as a relative one is what `order(further=<dir>)` did, and
    extremal_ceiling.py then priced that ordering BELOW its own chance
    rate (-3.8% margin on misses) before 0aea70c7 deleted it.

    This buys REGISTER, not capability: the model already reaches the same
    ordering as `sorted(rows, key=lambda h: h.xy[1])` on 44-58 episodes
    per run at 82% success. It is here so a global question is spelled
    globally, on the method a cut and a sort already live on.
    """
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


def test_the_order_docstring_does_not_promise_an_unwritten_column():
    """It said "every compare writes `facing <place>` and `<dir> of
    <place>`". Only `facing` is true: geom_program stamps the signed
    column ONLY for a direction a cardinal cut asked for, so the idiom the
    docstring taught -- rows.order(lambda r: -r["east of poppy"]) -- was a
    KeyError on any rows that had not already been cut.
    """
    from maps.search.parse_compile.python_surface import Rows
    doc = Rows.order.__doc__
    assert "<dir> of <place>" in doc          # still described...
    assert "rides\n        a cardinal CUT" in doc or "cardinal CUT" in doc
    # ...and never as something every compare writes
    assert "every compare\n        writes" not in doc
    assert "ORDERS GLOBALLY" in doc


def test_order_takes_reverse_on_a_key_and_names_the_word_on_a_direction():
    """`sorted()`'s own kwarg, because that is what the model writes.

    Live on the directional arm: the model wrote
    `.order(lambda r: r.xy[0], reverse=True)` and got
    `Rows.order() got an unexpected keyword argument 'reverse'`. A method
    surface that reads like the stdlib has to take the stdlib's kwarg.

    On direction= it is REFUSED, because the opposite direction is a word
    the model already has and two spellings of one ordering is what this
    surface keeps deleting.
    """
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
