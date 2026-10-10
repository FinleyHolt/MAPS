"""A set in a slot that MEASURES is one place: the gap to its nearest member.

This is what replaced the looking question. `look(rows, "which of these has
a pool behind it?")` answered into the picture and bound nothing, so the
distinction it resolved had nowhere to land: the model asked it in 89 of 500
episodes and then indexed rank 0 by DISTANCE anyway. The same question is a
QUERY now -- arm the class, cover the candidates, and the thing the question
was about comes back as ROWS that sort and cut like any others.

The executor could always fold a set to one column (the union ref, an exact
min gap over the members). Only the surface could not spell it: `near=<set>`
was refused outright with "index it -- r0[0] -- or use a name", which is the
wrong answer to the question being asked -- "the house with a pool in the
backyard" is not about pool number one. Fanning the set out instead measures
every instance separately and dies on the executor's token guard the moment
either side is large: 3,414 houses against 40 pools is 136,560 cells.

Pure: no GPU, no tile, no vLLM.
"""
from __future__ import annotations

import pytest

from maps.search.analyze_scene import geom_program as gp
from maps.search.analyze_scene.geom_program import compact_results
from maps.search.parse_compile.code_surface import CodeError
from maps.search.tests.test_code_surface_arm import ops_of_block, plan_of_block


def _house(fid, x, name=None):
    return {"fid": fid, "xy": [float(x), 0.0], "class": "house",
            "mask": [[[float(x), 0.0]]], "centerline": None, "cells": [],
            "name": name, "search_text": "house"}


def _pool(fid, x, y):
    return {"fid": fid, "xy": [float(x), float(y)], "class": "pool",
            "mask": [[[float(x), float(y)]]], "centerline": None, "cells": [],
            "search_text": "pool"}


def test_the_house_with_a_pool_in_the_backyard_is_one_line():
    """The whole point, end to end: compile it and run it.

    Three houses, one pool, and the pool is behind the middle one. Nothing
    in any row says "has a pool" -- the answer is the GAP, and the union is
    what turns forty pools into one column instead of forty.
    """
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


def test_a_slot_that_POSITIONS_still_needs_one_place():
    """A union has a region; it has no one point to stand at or centre on."""
    for src, slot in (("show_map(around=r0)\n", "around"),
                      ('stand(at=r0, facing="north")\n', "at")):
        with pytest.raises(CodeError, match=f"{slot}= takes ONE place"):
            plan_of_block(src, bound={"r0": "r7"})


def test_the_looking_question_is_off_this_surface():
    """look() shows a thing. It does not take a sentence about it."""
    with pytest.raises(CodeError, match="positional argument"):
        plan_of_block('look(r0, "which of these has a pool?")\n',
                      bound={"r0": "r7"})


def test_an_empty_set_fails_loud_and_names_the_handle_and_the_fix():
    """The overwhelmingly common shape: asking before anything has looked.

    Nothing is segmented until look_for names the class AND a flight goes
    over the place, so `find("pool")` on turn 1 is empty by construction.
    Ordering against an empty union would rank by nothing at all, which is
    the silent failure this tree keeps paying for -- so it raises, and the
    message names the handle the MODEL wrote (not the executor ref it
    compiled to) plus the thing to do about it.
    """
    ops = ops_of_block('pools = find("pool")\n'
                       'houses = find("house", nearest=pools)\n')
    with pytest.raises(gp.ProgramError) as exc:
        gp.run_program([], [_house("prior_1", 0.0)], {"ops": ops})
    msg = str(exc.value)
    assert "pools is empty" in msg, msg          # the model's own name
    assert "r0" not in msg.split("; signature")[0], msg
    assert "look_for" in msg and "flight" in msg, msg


def test_an_unseen_row_is_not_buried_by_a_colour_ordering():
    """`color=` must not become a coverage ordering wearing a colour's name.

    A prior carries a perceived colour only after a view passes over it, so
    on a cold surface EVERY candidate is unknown. Scoring unknown as
    worse-than-any-measured put a seen GREY house above every unseen house
    under color="white" -- and on this bench the target is usually the
    unseen one.
    """
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
    """`near=<set>` end to end: compiled, dispatched, served.

    A 500-episode fleet run is the thing this is about to be validated by,
    and the executor being right is not the same as the round being right --
    the plan is split at selections, dispatched step by step and rendered
    back. So drive it.
    """
    from maps.search.tests.test_mission_loop import _drive, _tool_results

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
    _reply, fr, _flights, _d = _drive([
        [("run", {"code": 'look_for("pool")\n'
                          'pools = find("pool")\n'
                          'houses = find("house", nearest=pools)\n'
                          'houses\n'})],
        "done",
    ], cands=cands)
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    rows = next(iter(out["result"]["results"].values()))["rows"]
    fids = [ln.strip().strip("|").split("|")[0].strip()
            for ln in rows.splitlines()[2:] if ln.strip()]
    assert fids[0] == "house_2", rows       # the one with the pool behind it
    assert "distance to pools" in rows, rows


def test_an_unseen_colour_drops_its_term_instead_of_killing_the_block():
    """Nothing carries a colour until a view has passed over it.

    So on every pre-takeoff surface `color=` names an absent column -- and
    a raise there kills the WHOLE block (`reasoner_loop` breaks the segment
    on the first error), taking down the `look_for` that would have armed
    the segmenter. The model is then unable to ever see the thing whose
    colour it asked about, from one legal call the legend teaches.

    The ops are built here rather than compiled: `near=<name>` compiles to
    a grounding placeholder the round resolves, and what is under test is
    the executor's own drop.
    """
    road = {"fid": "road_1", "name": "Mill Lane", "xy": [0.0, 0.0],
            "class": "road", "centerline": [[0.0, -50.0], [0.0, 50.0]],
            "mask": [[[0.0, float(y)] for y in range(-50, 51, 10)]],
            "cells": [], "search_text": "mill lane"}
    ops = [
        {"ref": "r0", "op": "find",
         "args": {"query": "house", "provenance": "prior"}},
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
    assert "has been SEEN yet" in hint, hint       # and it is NEVER silent


def test_a_colour_alone_with_nothing_seen_is_still_a_loud_refusal():
    """A sort with no term left is not an ordering, and serving one as
    though it were is the silent failure this tree keeps paying for."""
    ops = ops_of_block('r1 = find("house", color="white")\n')
    with pytest.raises(gp.ProgramError, match="nothing orders by"):
        gp.run_program([], [_house("h_a", 0.0), _house("h_b", 10.0)],
                       {"ops": ops})


def test_a_union_over_a_whole_class_is_capped_rather_than_slow():
    """The old guard was that a union needed hand-typed fids.

    An rN member removes it, and `near=find("house")` folds every house on
    the tile -- the members x items shape the union comment records as
    having "ran 885 s without returning". A wall-clock cliff found inside a
    500-episode fleet run is the expensive place to find it.
    """
    ops = ops_of_block('everything = find("house")\n'
                       'r1 = find("house", nearest=everything)\n')
    many = [_house(f"h{i:04d}", float(i)) for i in range(gp._UNION_MAX_MEMBERS + 5)]
    with pytest.raises(gp.ProgramError, match="folds at most"):
        gp.run_program([], many, {"ops": ops})


def test_a_served_block_flies_and_then_queries_what_it_found():
    """The execution model, asserted against the served bytes.

    Lines run eagerly in source order, so a block that covers a place and
    then asks for what the pass found reads the rows the pass just
    detected -- the chain the fold-order surface (analyze before every
    flight) had to split across two blocks. The worked example teaches
    exactly that shape, so the served text and the execution model state
    ONE fact.

    A static guard because it is a property of the SERVED TEXT, and the
    same shape the `_WORKED` import-time guards already take.
    """
    import ast

    from maps.search.parse_compile.prompts import CODE_DOC
    from maps.search.parse_compile.python_surface import scan_block

    body = CODE_DOC.partition("This is one block:\n")[2].partition(
        "\nEach line runs")[0]
    block = "\n".join(ln.strip() for ln in body.splitlines()
                      if ln.startswith("  ") and ln.strip())
    calls = [n.func.id for n in ast.walk(scan_block(block))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    flies_at = min(i for i, c in enumerate(calls)
                   if c in ("explore", "go_to"))
    # the find AFTER the pass is the point: it reads the pass's detections
    assert [c for c in calls[flies_at:] if c == "find"], (calls, block)
    # ...and the ranking find carries the colour term beside the place
    assert "color=" in block and "nearest=" in block, block
