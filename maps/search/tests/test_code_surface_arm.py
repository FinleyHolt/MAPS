from __future__ import annotations

import json
import re

import pytest

from maps.search.tests.fakes import (
    CANDS,
    START,
    STOPS,
    drive,
    explore_drive,
    grow_drive,
    acts_of_block,
    calls_of_block,
    grounded_in_block,
    ops_of_block,
    raising_ops_of_block,
    run,
    tool_results,
)


def test_the_whole_vocabulary_in_one_episode():
    """find -> narrow -> order -> select -> go_to, plus adjective ordering."""
    reply, fr, flights, _d = drive([
        [run('look_at = 0\n')],           # ordinary Python, a silent bind
        [run('r1 = find("car")\n'
              'r2 = r1.filter(within="the depot")\n'
              'r3 = r1.order(nearest="the depot")\n'
              'r4 = r3[0]\n')],
        [run('go_to(r4)\n')],
        "the nearest car",
    ])
    out = tool_results(fr)
    assert "error" not in out[0], out[0]
    assert "error" not in out[1], out[1]
    assert out[1]["handles"] == ["r1", "r2", "r3", "r4"], out[1]
    # each block is recorded append-only and counted, and r4 outlives its block
    assert [o["script_lines"] for o in out[:3]] == [1, 5, 6], out
    assert "r4" in out[2]["handles"], out[2]
    # size= compiles to a footprint_area_m2 sort (asserted in
    # code_surface.demo); this fixture's synthetic cars carry no footprint,
    # so the executor says so rather than ordering by nothing.
    assert len(flights) == 1
    assert flights[0]["target_fid"] == "car_1", flights


def test_the_watchlist_and_the_loud_errors_are_the_same_ones():
    reply, fr, _flights, _d = drive([
        [run('r1 = find("car", nearest="nowhere at all")\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "error" in out[0] and "nothing named" in out[0]["error"], out[0]
    assert out[0]["error"].endswith("or look_for + fly more first"), out[0]
    assert out[0]["watching"], out[0]


def test_a_loop_that_flies_is_refused_before_anything_runs():
    """The one structural cut, at the surface the model actually writes to."""
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\nfor c in r1:\n    go_to(c)\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "error" in out[0]
    assert "cannot go inside a loop" in out[0]["error"], out[0]["error"]
    assert flights == [], flights


# --- explore: the covering pass, and the return leg that makes it free ----

def test_an_explore_flies_the_stops_and_ends_on_the_last_one():
    """A pass covers the thing and STAYS where it finished."""
    reply, out, flights, asked, _rend = explore_drive([
        [run('explore("prior_7")\n')],
        "done",
    ])
    assert "error" not in out[0], out[0]
    # the stops of the thing, and NOTHING after them
    assert len(flights) == len(STOPS), flights
    from maps.util.xy_token import parse_xy_token
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == STOPS, flown
    ack = out[0]["explored"]
    assert ack["stops"] == len(STOPS), ack
    # ONE ack for the pass, not a list of flight reports
    assert "flown" not in out[0], out[0]
    assert ack["new_detections"] == len(STOPS), ack
    # No coordinate is served for where the walk ended: the answer standing
    # is named as the pass's last stop over the thing explored.
    assert "ended_at" not in ack, ack
    assert ack["answer_if_you_reply_now"].startswith(
        f"the last stop of the pass over {ack['explored']}"), ack
    # it did NOT go home
    assert list(START) not in flown, flown


def test_an_explore_spends_half_an_observation_of_the_run_per_stop():
    """The spacing is the RUN's, never a constant."""
    _r, _o, _f, asked, _rend = explore_drive([
        [run('explore("prior_7")\n')],
        "done",
    ], obs_span_m=320.82)
    assert asked == [("prior_7", 320.82 / 2.0)], asked


def test_an_explore_of_a_thing_with_no_footprint_is_the_one_stop_it_is():
    """A fid carrying no cells: there is nothing to cover, and the pass says so rather
    than reporting a cover it did not fly."""
    reply, out, flights, _a, _rend = explore_drive([
        [run('explore("prior_7")\n')],
        "done",
    ], stops=None)
    assert "error" not in out[0], out[0]
    assert len(flights) == 1, flights
    assert flights[0]["target_fid"] == "prior_7"
    assert "no footprint to cover" in (out[0]["explored"].get("note") or "")


def test_a_call_indexed_inside_an_argument_is_its_own_dispatch():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'r2 = r1.filter(within=find("gate")[0])\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "error" not in out[0], out[0]
    assert "distance to prior_1" in json.dumps(out[0]), out[0]
    assert out[0]["handles"] == ["r1", "r2"], out[0]


def test_a_bare_handle_line_is_its_one_display():
    assert calls_of_block("r1\n", bound={"r1": "r7"}) == [("analyze", [
        {"ref": "r0", "op": "filter_items",
         "args": {"items": "r7", "key": "xy", "op": "truthy"}}],
        "r0", None)]
    # one way to display: no show() verb beside the bare line
    _, fr, _, _ = drive([[run('r1 = find("car")\nshow(r1)\n')], "done"])
    assert "there is no show here" in tool_results(fr)[0]["error"]


def test_a_fid_in_an_anchor_slot_is_the_row_it_names():
    def _refs(src):
        out = [o["args"]["refs"] for o in ops_of_block(src, {"r1": "r7"})
               if o["args"].get("refs")]
        return out, grounded_in_block(src, {"r1": "r7"})

    for src in ('r2 = r1.order(nearest="prior_6087")\n',
                'r2 = r1.filter(within="prior_5631")\n',
                'r0 = intersection("prior_5133", "prior_5954")\n',
                'r2 = r1.order(nearest="detected_12")\n'):
        refs, grounds = _refs(src)
        flat = [v for group in refs for v in group]
        assert all(isinstance(v, str) for v in flat), (src, refs)
        assert grounds == [], (src, grounds)
    # THE VACUITY FLOOR: a real NAME must still ground, or this "fix"
    # would be a surface that grounds nothing at all.
    refs, grounds = _refs('r2 = r1.order(nearest="Green Street")\n')
    assert grounds == ["Green Street"], grounds
    assert refs[0][0] == {"ref": "<Green Street>", "as": "Green Street"}, refs


def test_an_alias_gives_a_result_a_second_name():
    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    # BOTH NAMES REACH THE SAME ROWS: the alias is the same Rows object
    # under a second name, so it mints nothing and re-runs nothing -- the
    # ordering compiles straight off the source's own executor ref.
    ops = ops_of_block('segs = r1\nr2 = segs.order(nearest="A St")\n',
                       bound={"r1": "r7"})
    assert [o["op"] for o in ops] == ["compare", "score_by"], ops
    assert ops[0]["args"]["items"] == "r7", ops
    # An alias of NOTHING is still refused, and says which name is unbound.
    with pytest.raises(CodeError, match="name 'nope' is not defined"):
        calls_of_block("x = nope\n")


def test_a_snake_case_class_is_the_class_it_spells():
    # THE WIRE GETS WORDS.
    assert acts_of_block('look_for("solar_panel", "parking_lot")\n') == [
        ("watch", ["solar panel", "parking lot"])]


def test_look_for_refuses_a_class_that_is_not_a_string():
    # A list would arm the segmenter on the class "['car']".
    from maps.search.parse_compile.python_surface import CodeError
    with pytest.raises(CodeError) as e:
        acts_of_block('look_for(["car"])\n')
    assert 'look_for takes class names as strings: look_for("house", "pool")' \
        in str(e.value), str(e.value)


def test_an_inline_point_relates_exactly_as_a_bound_one_does():
    def _cols(ops):
        cmp_ = [o for o in ops if o["op"] == "compare"][0]
        sc = [o for o in ops if o["op"] == "score_by"][0]
        return cmp_["args"]["refs"][0], sc["args"]["emphasis"][0]["key"]

    inline = ops_of_block('r1 = find("tree", nearest=[4630, 2965])\n')
    bound = ops_of_block('pt = [4630, 2965]\nr1 = find("tree", nearest=pt)\n')
    for ops in (inline, bound):
        ref, key = _cols(ops)
        # THE TWO HALVES AGREE BY CONSTRUCTION: the compare labels the
        # column and the ordering names that same label.
        assert isinstance(ref, dict) and "as" in ref, ref
        assert key == f"distance to {ref['as'].replace(' ', '_')}", (ref, key)
    # ...and the point itself survives as a point, not as a name to ground.
    ref, _ = _cols(inline)
    assert ref["ref"] == [4630.0, 2965.0], ref
    assert grounded_in_block('r1 = find("tree", nearest=[4630, 2965])\n') \
        == []


def test_the_moving_calls_refuse_a_coordinate_in_every_spelling():
    """A COORDINATE IS THE MODEL'S ESTIMATE; a row's position is a measurement."""
    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    for bad in ("go_to([3411, 1468])\n",
                "explore([3411, 1468])\n",
                "target = [4630, 2965]\ngo_to(target)\n",
                "corner = [3411, 1468]\nexplore(corner)\n",
                'go_to("[3411, 1468]")\n'):
        with pytest.raises(CodeError, match="takes a THING on the map"):
            calls_of_block(bad)
    with pytest.raises(CodeError, match="positional argument"):
        calls_of_block("go_to(3411, 1468)\n")
    # ...and the refusal names the instrument that replaces it.
    try:
        calls_of_block("go_to([3411, 1468])\n")
    except CodeError as exc:
        assert "look_for" in str(exc) and "explore the thing" in str(exc), str(exc)
    # A POINT STILL RELATES AND STILL LOOKS — only moving refuses it.
    assert calls_of_block('corner = [1, 2]\nr1 = find("car", nearest=corner)\n')
    assert calls_of_block("show_map(around=[1, 2])\n")
    assert calls_of_block('stand(at=[1, 2], facing="north")\n')


def test_go_to_refuses_a_python_set_as_a_set():
    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    with pytest.raises(CodeError, match="takes ONE thing and that is a set of 2"):
        calls_of_block('go_to({"prior_0", "prior_1"})\n')
    with pytest.raises(CodeError, match="takes ONE thing and that is a set of 2"):
        calls_of_block("go_to({a, b})\n", bound={"a": "prior_0", "b": "prior_1"})


def test_a_name_constraint_gates_or_fails_loud_never_vanishes():
    """``find(query, name=...)`` gates by name or says nothing bears it."""
    reply, fr, _f, _d = drive([
        [run('r1 = find("church", name="Desert Song Foursquare Church")\n')],
        "done",
    ])
    err = tool_results(fr)[0].get("error") or ""
    # never silently dropped: either it gates, or it says nothing bears
    # that name. What it must NEVER do is succeed while ignoring it.
    assert "NAMED" in err and "Desert Song" in err, err
    assert "never fuzzily" in err, err
    assert "find('Desert Song Foursquare Church')" in err, err


def test_a_name_with_no_query_is_refused_as_the_query_itself():
    """``find(name=X)`` alone browsed every row; the name is the query."""
    from maps.search.parse_compile.python_surface import CodeError
    with pytest.raises(CodeError, match=r"the name IS the query: find\('Mill Lane'\)"):
        raising_ops_of_block('r1 = find(name="Mill Lane")\n')


def test_the_vision_calls_are_the_sentence_surface_s_own_steps():
    """look / the looking question / the map compile to the SAME steps."""
    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    bound = {"r1": "r7"}
    assert acts_of_block("look(r1)\n", bound) == [("crops", "r7", None, None)]
    # AND IT TAKES NOTHING ELSE. The free-text "looking question" is off this surface:
    # it answered into the picture and bound nothing, so a distinction it resolved had
    # nowhere to land.
    with pytest.raises(CodeError, match="positional argument"):
        calls_of_block('look(r1, "which one has a pool?")\n', bound)
    # around= POSITIONS rather than measures, so a SET is refused with the
    # fix in the message and a ROW maps straight through.
    with pytest.raises(CodeError, match="around= takes ONE place"):
        calls_of_block("show_map(around=r1)\n", bound)
    assert acts_of_block("show_map(around=r2)\n", {"r2": "prior_5"}) == [
        ("map", "prior_5")]


def test_no_argument_is_dropped_on_the_floor():
    import pytest

    from maps.search.parse_compile.code_surface import CodeError

    # find("road", query="Sherwood Avenue") -- the positional won and the
    # keyword vanished. 39 calls; one served all 810 roads on the bake.
    with pytest.raises(CodeError, match="multiple values for argument"):
        calls_of_block('r1 = find("road", query="Sherwood Avenue")\n')
    # ...and either one alone still works.
    assert ops_of_block('r1 = find(query="Sherwood Avenue")\n')[0]["args"] \
        == {"query": "Sherwood Avenue"}

    # go_to() refuses a keyword it does not take: an ignored
    # offset=(400, 0) would fly to the identical pose as offset=(-100, 0).
    with pytest.raises(CodeError, match="unexpected keyword argument 'offset'"):
        calls_of_block('go_to(r1, offset=(400, 0))\n', {"r1": "prior_1"})
    assert acts_of_block("go_to(r1)\n", {"r1": "prior_1"}) \
        == [("move", "prior_1")]

    # near= and within= together: the compiler read `near or inside` and filtered on
    # near='s anchor, swapping the semantics in silence.
    ops = ops_of_block(
        'r1 = find("building", nearest="A Street", within="B Street")\n')
    cut, = [o for o in ops if o["op"] == "filter_items"]
    sort, = [o for o in ops if o["op"] == "score_by"]
    assert cut["args"]["key"] == "distance to b_street", ops
    assert [e["key"] for e in sort["args"]["emphasis"]] == [
        "distance to a_street"], ops

    # The kwarg gate's twin: an EXTRA POSITIONAL was dropped just as
    # silently, and it is the same sentence spelled the other way.
    with pytest.raises(CodeError, match="find takes ONE query") as e:
        calls_of_block('r1 = find("road", "Sherwood Avenue")\n')
    assert "intersection(" in str(e.value), str(e.value)
    with pytest.raises(CodeError, match="positional argument"):
        calls_of_block('go_to("prior_1", "prior_2")\n')
    # ...and too FEW: go_to() flew to the string "None".
    with pytest.raises(CodeError, match="missing 1 required positional"):
        calls_of_block("go_to()\n")
    with pytest.raises(CodeError, match="order needs its key") as e:
        calls_of_block("r2 = r1.order()\n", {"r1": "r1"})
    assert 'rows.order(direction="north")' in str(e.value), str(e.value)


def test_an_empty_colour_orders_by_nothing_and_is_refused():
    from maps.search.parse_compile.python_surface import CodeError
    with pytest.raises(CodeError, match="order ranks by nearest= or color="):
        raising_ops_of_block('r1 = find("car")\nr2 = r1.order(color="")\n')


def test_any_earlier_handle_stays_indexable_not_just_the_newest():
    """`go_to(r1[0])` with a block in between."""
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\n')],
        [run('r2 = find("tower")\n')],       # r1 is no longer the newest
        [run('go_to(r1[0])\n')],
        "the car",
    ])
    out = tool_results(fr)
    assert "error" not in out[2], out[2]
    assert len(flights) == 1
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_a_pick_never_inherits_the_previous_block_s_row():
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\n')],
        [run('go_to(r1[0])\n')],                # binds the internal _p0
        [run('go_to(r9[0])\n')],                # r9 is not a handle
        "done",
    ])
    out = tool_results(fr)
    assert flights and flights[0]["target_fid"] == "car_1", flights
    # The unbound pick resolves to NOTHING -- never to the row the last
    # block's `_p0` happened to hold.
    assert "car_1" not in [f["target_fid"] for f in flights[1:]], flights
    # ...and no internal name outlived its own block.
    assert not [h for h in out[1].get("handles") or [] if h.startswith("_")], \
        out[1]


def test_a_pick_is_usable_in_the_block_that_made_it():
    """The whole chain in ONE block, which is what the surface is for."""
    reply, fr, flights, _d = drive([
        [run('r1 = find("car", nearest="the depot")\n'
              'r2 = r1[0]\n'
              'r3 = find("gate", nearest=r2)\n'
              'go_to(r2)\n')],
        "the car by the depot",
    ])
    out = tool_results(fr)
    assert "error" not in out[0], out[0]
    # ONE BLOCK, ONE TABLE: the block's last result keeps its rows and the
    # earlier find folds to its hint, attributed to the model's name.
    res = out[0]["result"]["results"]
    assert list(res) == ["r3"], res
    assert "r1:" in (res["r3"].get("hint") or ""), res["r3"]
    assert out[0]["handles"] == ["r1", "r2", "r3"], out[0]
    assert len(flights) == 1
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_a_selected_crossing_can_anchor_the_next_call():
    """The surface's headline chain: two roads, their crossing, what is near."""
    from maps.search.parse_compile.code_surface import bind_row
    assert bind_row({"fid": "[6076, 4226]"}) == [6076.0, 4226.0]
    assert bind_row({"fid": "prior_9"}) == "prior_9"   # every other row type

    xing = [{"fid": "[10, 20]", "xy": [10.0, 20.0], "class": "crossing",
             "search_text": "the crossing"}] + list(CANDS)
    reply, fr, flights, _d = drive([
        [run('r1 = find("the crossing")\n'
              'r2 = r1[0]\n'
              'r3 = find("car", nearest=r2)\n')],
        [run("go_to(r2)\n")],
        "done",
    ], cands=xing)
    out = tool_results(fr)
    assert "error" not in out[0], out[0]      # not `nothing named "[10, 20]"`
    assert "error" not in out[1], out[1]
    assert flights and flights[0]["target_fid"] == "[10, 20]", flights
    # The anchor went in as a POINT, so compare labelled its columns after
    # the handle. Bound as a string it takes the grounding path instead and
    # the columns come back named 'distance to 10_20' -- the same shape
    # that names nothing on a real bake, where no row is named "[10, 20]".
    assert "distance to r2" in out[0]["result"]["results"]["r3"]["rows"], \
        out[0]["result"]["results"]["r3"]["rows"]

    # ...and the point rides every one-place slot the same way. The row a
    # selection binds carries the point in `bind`, which is what every
    # slot reads -- the token that spells it never reaches one.
    from maps.search.parse_compile.python_surface import Row
    pt = {"r2": Row({"fid": "[10, 20]", "xy": [10.0, 20.0]})}
    assert grounded_in_block('r3 = find("car", nearest=r2)\n', pt) == []
    assert ops_of_block('r3 = find("car", nearest=r2)\n', pt)[1]["args"]["refs"] \
        == [{"ref": [10.0, 20.0], "as": "r2"}]
    assert acts_of_block("go_to(r2)\n", pt) == [("move", "[10, 20]")]
    assert acts_of_block("show_map(around=r2)\n", pt) == [("map", "[10, 20]")]


def test_only_the_model_s_own_names_are_served_back():
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\n'
              'r2 = find("gate", nearest="the depot")\n')],
        [run("go_to(r1[0])\n")],               # mints an internal _p0
        [run('r3 = find("tower")\n')],
        "done",
    ])
    out = tool_results(fr)
    # The receipt is keyed by what the MODEL called it. r2's op chain is
    # find -> compare -> score_by, so the executor's own name for it is r3.
    assert list(out[0]["result"]["results"]) == ["r2"], \
        list(out[0]["result"]["results"])
    # No internal name anywhere in what is served back.
    for res in out:
        assert "_p" not in json.dumps(res), res
    assert out[2]["handles"] == ["r1", "r2", "r3"], out[2]
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_an_unbindable_inline_pick_names_what_the_model_wrote():
    reply, fr, flights, _d = drive([
        [run("go_to(r9[0])\n")],
        "done",
    ])
    err = tool_results(fr)[0].get("error") or ""
    assert "name 'r9' is not defined" in err, err
    assert "_p" not in err, err
    assert flights == [], flights


def test_a_ref_is_told_from_a_name_by_shape_not_by_first_letter():
    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    src = 'r3 = find("gate", nearest=r2)\n'
    held = {"r1": "r0", "r2": "car_1"}
    # A bound fid is a fid: it rides the ref slot, it is not looked up.
    assert grounded_in_block(src, held) == []
    assert ops_of_block(src, held)[1]["args"]["refs"] == [
        {"ref": "car_1", "as": "r2"}]
    # A place the model named is grounded whatever letter it starts with.
    assert grounded_in_block('r1 = find("car", nearest="Riverside Drive")\n') \
        == ["Riverside Drive"]
    # An unbound handle never reaches the compiler at all: it is a
    # NameError on the model's own line, rather than an op ref the
    # executor answers with "reference cycle".
    with pytest.raises(CodeError, match="name 'r0' is not defined"):
        calls_of_block('r8 = find("car", nearest=r0)\n')
    # look() on a row: the SET branch is for an op ref, by shape.
    assert acts_of_block("look(r2)\n", {"r2": "road_1"}) == [
        ("crops", None, "road_1", None)]
    assert acts_of_block("look(r2)\n", {"r2": "r7"}) == [
        ("crops", "r7", None, None)]
    # ...and what the COMPILER put in a ref slot is never a name to ground:
    # intersection(r1, ...) relates to every way r1 names, which is the
    # escape hatch for a road whose name matches several.
    xsrc = 'r2 = intersection(r1, "Cross Street")\n'
    # A SET in a ref slot is its UNION, and `crossings` unwraps the wrapper
    # to the same rows the bare ref gave it -- "every way r1 names".
    assert ops_of_block(xsrc, {"r1": "r7"})[0]["args"]["refs"][0] == {
        "ref": ["r7"], "as": "r1"}
    assert grounded_in_block(xsrc, {"r1": "r7"}) == ["Cross Street"]


def test_the_items_slot_takes_a_bare_ref_and_the_place_slots_a_labelled_one():
    """Two shapes for one lookup, and the coercers are not interchangeable."""
    for src in ('r9 = r2.order(nearest="the depot")\n',
                'r9 = r2.filter(within="the depot")\n'):
        ops = ops_of_block(src, {"r2": "car_1"})
        assert ops[0]["args"]["items"] == "car_1", (src, ops)
        assert ops[0]["args"]["refs"][0]["as"] == "the depot", (src, ops)
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nr2 = r1[0]\n')],
        [run('r9 = r2.order(nearest="the depot")\n')],
        "done",
    ])
    assert "error" not in tool_results(fr)[1], tool_results(fr)[1]


def test_the_executor_never_mints_a_ref_the_model_already_uses():
    """The two rN namespaces overlap, so this program skips the model's."""
    refs = [o["ref"] for o in ops_of_block(
        'r4 = find("gate", nearest=r3)\n',
        {"r1": "r0", "r3": "car_2", "r4": "r9"}, rn=1)]
    assert "r3" not in refs and "r4" not in refs, refs
    assert refs == ["r2", "r5", "r6"], refs   # r3/r4 are the model's
    # ...and the whole chain runs rather than tripping the cycle detector.
    reply, fr, _f, _d = drive([
        [run("r1 = find('car')\nr2 = r1[0]\n")],
        [run("r3 = r1[1]\nr4 = find('gate', nearest=r3)\n")],
        "done",
    ])
    out = tool_results(fr)
    assert "error" not in out[1], out[1]
    assert "distance to r3" in str(out[1]["result"]["results"]), out[1]


def test_a_call_compiles_from_the_counter_the_flight_moved():
    """A flight mid-block takes an rN, and the next call must not reuse it."""
    results, _g, added = grow_drive([
        [run('go_to("prior_7")\n'          # flies, and mints over its rows
              'c3 = find("gate")\n')],      # ...must not land on that ref
        "done",
    ])
    assert "error" not in results[0], results[0]
    (rep,) = results[0]["flown"]["flown"]
    minted = rep["new_rows"].split(" ")[0]
    assert re.fullmatch(r"r\d+", minted), rep
    # the flight's handle is still addressable, and it is not c3's
    assert minted in results[0]["handles"], results[0]
    assert "c3" in results[0]["handles"], results[0]
    assert added[0], added
    res = results[0]["result"]["results"]
    # THE WITNESS: the find matched nothing, so a find compiled onto the
    # flight's own ref would overwrite its rows with an empty set under
    # the name the flight report just handed the model.
    assert res["c3"]["len"] == 0, res
    assert res["go_to('prior_7')"]["len"] == len(added[0]), res


def test_the_receipt_rename_never_serves_one_handle_under_another_s_name():
    """The rename is one-to-one or it does not happen."""
    reply, fr, _f, _d = drive([
        [run("r1 = find('car')\nr2 = r1[0]\nfind('gate')\n")],
        "done",
    ])
    res = tool_results(fr)[0]["result"]["results"]
    got = {k: v.get("fids") for k, v in res.items()}
    # The unassigned find is the block's shown answer and is keyed by the
    # model's own spelling; r1's envelope folds to its attributed hint --
    # never served under another name.
    assert got.get("find('gate')") == ["prior_1"], got
    assert "r1:" in (res["find('gate')"].get("hint") or ""), res


def test_a_rebind_that_failed_unbinds_the_name():
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\nr3 = r1[0]\n')],
        [run('r2 = find("gate")\nr3 = r2[99]\n')],      # r3 fails to re-bind
        [run("go_to(r3)\n")],
        "done",
    ])
    out = tool_results(fr)
    assert "did not bind" in (out[1].get("error") or ""), out[1]
    assert "r3" not in out[1]["handles"], out[1]["handles"]
    assert "car_1" not in [f["target_fid"] for f in flights], flights


def test_a_later_segment_s_refusal_keeps_the_earlier_bindings():
    """`r1.filter()` is caught by the compiler, not the parser."""
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'r2 = r1[0]\n'
              'r3 = r1.filter()\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "filter needs what to cut against" in (out[0].get("error") or ""), \
        out[0]
    assert out[0]["handles"] == ["r1", "r2"], out[0]


def test_a_flight_never_mints_a_handle_the_model_is_already_using():
    results, _g, added = grow_drive([
        [run('go_to("prior_7")\n')],              # mints r0; counter -> 1
        [run('r1 = find("black car")\n')],      # skips r1, mints r2 -> 3
        [run("r3 = r1[0]\ngo_to(r3)\n")],         # the mint lands on r3
        "done",
    ])
    (rep,) = results[2]["flown"]["flown"]
    handle = rep["new_rows"].split(" ")[0]
    assert handle != "r3", rep
    # ...and a handle the report NAMES is one the surface can resolve: it
    # is an executor ref, so it binds to itself in the code scope.
    assert handle in results[2]["handles"], results[2]
    assert added[0], added


def test_a_bare_display_line_shows_the_rows_it_promises():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nr1\n')],
        [run("r1[1]\n")],
        [run("r9\n")],
        "done",
    ])
    out = tool_results(fr)
    whole = out[0]["result"]["results"]["r1"]
    assert whole["fids"] == ["car_1", "car_2"], whole
    assert "car_1 | car" in whole["rows"], whole
    # ONE row, still rendered as a row: an int index yields a bare dict and
    # loses the ordering attribute, so the envelope downgraded to census +
    # len -- the model asked to SEE it and got a count of one.
    one = out[1]["result"]["results"]["r1[1]"]
    assert one["fids"] == ["car_2"], one
    assert "car_2 | car" in one["rows"], one
    # ...and a display of nothing is loud rather than empty.
    assert "name 'r9' is not defined" in (out[2].get("error") or ""), out[2]


def test_showing_a_one_row_handle_serves_the_row_it_stands_for():
    """A pick binds a FID, and a fid names no result to window."""
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nr2 = r1[0]\nr2\n')],
        "done",
    ])
    out = tool_results(fr)
    assert out[0]["shown"] == {"r2": "car_1"}, out[0]


def test_a_pick_says_which_row_it_got():
    xing = [{"fid": "prior_9", "xy": [1.0, 1.0], "class": "church",
             "name": "Desert Song Foursquare Church",
             "search_text": "church desert song foursquare church"}]
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\nr2 = r1[1]\n')],
        [run('r3 = find("church")\nr4 = r3[0]\ngo_to(r4)\n')],
        "done",
    ], cands=list(CANDS) + xing)
    out = tool_results(fr)
    assert out[0]["picked"] == {"r2": {"fid": "car_2"}}, out[0]
    # ...and the NAME rides it when the row has one, which is the whole
    # difference between "r4 is bound" and "r4 is the church you meant".
    assert out[1]["picked"] == {
        "r4": {"fid": "prior_9",
               "name": "Desert Song Foursquare Church"}}, out[1]
    # The inline pick's internal name is never receipted: it is this
    # block's private wiring and naming it teaches a handle the model
    # never made.
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\n')],
        [run("go_to(r1[0])\n")],
        "done",
    ])
    assert "picked" not in tool_results(fr)[1], tool_results(fr)[1]


def test_look_for_arms_the_segmenter_the_footer_already_advertises():
    reply, fr, _f, declared = drive([
        [run('look_for("pool")\n')],
        [run('look_for("car", "parking lot")\n')],
        "done",
    ])
    out = tool_results(fr)
    assert out[0]["watching"] == {"now": ["pool (target)"]}, out[0]
    assert out[1]["watching"]["now"] == [
        "pool (target)", "car (target)", "parking lot (target)"], out[1]
    assert declared[-1] == ["pool", "car", "parking lot"]
    # ...and free means loop-legal: the pre-scan cuts the MOVING calls out
    # of a loop body and nothing else.
    from maps.search.parse_compile.python_surface import scan_block
    assert scan_block('for c in r1:\n    look_for("pool")\n')


def test_the_near_form_is_the_targeted_boost_and_a_place_is_grounded():
    """`look_for("pool", at=<place>)` compiles to the boost step, and its where= is a
    GROUNDING."""
    assert calls_of_block('look_for("pool", at="the church")\n') == [
        ("ground", {"the church": "boost"}),
        ("boost", "<the church>", ["pool"])]
    # A flight to a NAME resolves the name; a flight to a fid stays literal
    # (grounding one would look its NAME up and find nothing), and so does
    # anything a binding holds -- a fid a selection resolved came OUT of a
    # row and must never go back through the find that produced it.
    assert calls_of_block('go_to("the depot")\n') == [
        ("ground", {"the depot": "move"}), ("move", "<the depot>")]
    assert calls_of_block('go_to("prior_160")\n') == [("move", "prior_160")]
    assert calls_of_block("go_to(r2)\n", {"r2": "car_1"}) == [
        ("move", "car_1")]
    # ...and end to end: the name reaches the flight as the row it named.
    reply, fr, flights, _d = drive([
        [run('go_to("the depot")\n')],
        "done",
    ])
    assert flights == [{"target_fid": "prior_7"}], flights


def test_stand_adopts_the_viewpoint_and_its_columns_ride_the_rows():
    """`stand(at=..., facing=...)` -- the sentence's from-looking prefix."""
    reply, fr, flights, _d = drive([
        [run('stand(at="prior_7", facing="north")\nr1 = find("car")\n')],
        [run('stand(at="the gate", facing="prior_2")\n')],
        [run('go_to("v1")\n')],
        "done",
    ])
    out = tool_results(fr)
    assert out[0]["stance"]["staged"][0]["handle"] == "v1", out[0]
    assert "left of v1" in out[0]["result"]["results"]["r1"]["rows"], out[0]
    # ...a compass word and a PLACE are both headings, and a place is
    # grounded like any other.
    assert out[1]["stance"]["staged"][0]["handle"] == "v2", out[1]
    # A STANCE IS NOT SOMEWHERE TO MOVE TO, and refusing that is what
    # keeps the coordinate hatch shut: the move wire resolves a vN to its
    # stand point, so stand(at=[4620, 2965]) + go_to(v1) would fly the
    # model's own estimate in two hops -- the thing go_to([x, y]) was
    # closed for. The stance keeps its point; it just cannot be flown.
    assert "not a thing on the map" in out[2]["error"], out[2]
    assert flights == [], flights

    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    from maps.search.parse_compile.python_surface import scan_block
    assert calls_of_block('stand(at="the church", facing="the gate")\n') == [
        ("ground", {"the church": "stage", "the gate": "facing"}),
        ("stand", "<the church>", "<the gate>")]
    # ...and both slots are one-place slots.
    with pytest.raises(CodeError, match="at= takes ONE place"):
        calls_of_block('stand(at=r1, facing="north")\n', {"r1": "r7"})
    # Free means loop-legal, like look_for and unlike the moving calls.
    assert scan_block('for c in r1:\n    stand(at=c, facing="north")\n')


def test_more_of_pages_a_long_result_and_stops_at_the_end():
    """`more_of(r1)` -- the sentence `more of rN.`"""
    from maps.search.analyze_scene.geom_program import BROAD_HEAD_ROWS as H
    wide = [{"fid": f"car_{i:03d}", "xy": [float(i), 0.0], "class": "car",
             "search_text": "car"} for i in range(3 * H)]
    reply, fr, _f, _d = drive([
        [run('cars = find("car")\n')],
        [run("more_of(cars)\n")],
        [run("more_of(cars)\n")],
        [run("more_of(cars)\n")],
        "done",
    ], cands=wide)
    out = tool_results(fr)
    # The truncated head names the call that pages it, never the rN.
    head = out[0]["result"]["results"]["cars"]
    assert "more_of on it serves the next" in head["shown"], head
    assert "<rN>" not in json.dumps(out[0]), out[0]
    served = [out[0]["result"]["results"]["cars"]["fids"]]
    for r in (out[1], out[2]):
        assert r.get("error") is None, r
        # Labelled with what was ASKED FOR: the rows are a WINDOW of cars,
        # not cars, so keying them "cars" would claim to be the whole set.
        env = r["result"]["results"]["more_of(cars)"]
        assert env["len"] == H and "rows" in env, env
        served.append(env["fids"])
    assert served[0] + served[1] + served[2] == [f"car_{i:03d}"
                                                 for i in range(3 * H)]
    end = out[3].get("error") or ""
    assert "already served" in end, out[3]
    # ...and the narrowing it teaches is spelled in this surface's methods,
    # on the model's own name for the set.
    assert 'of cars are already served' in end, end
    assert 'cars.order(color="<colour>")' in end, end
    assert 'cars.filter(within="<place>")' in end, end
    assert "order_by(" not in end and "narrow(" not in end, end

    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    with pytest.raises(CodeError, match="name 'r9' is not defined"):
        calls_of_block("more_of(r9)\n")
    with pytest.raises(CodeError, match="more_of pages a RESULT"):
        calls_of_block("more_of(r2)\n", {"r2": "car_1"})
    # A RESULT pages: the window opens past the rows already asked for.
    from maps.search.analyze_scene.geom_program import BROAD_HEAD_ROWS as B
    (_v, (op,), answer, page), _shown = calls_of_block(
        "more_of(r1)\n", {"r1": "r7"}, widened={"r7": B})
    assert op["args"]["items"] == f"r7[{2 * B}:{3 * B}]", op
    assert answer == op["ref"] and page == ("r7", 2 * B, "r1"), page


def test_three_ordinary_python_forms_that_were_refused_outright():
    from maps.search.parse_compile.python_surface import Row

    # 1. a literal binding (57). What a one-place slot takes -- a fid off
    #    a served row, a world point off a caption -- binds and rides; a
    #    NAME binds too and grounds where it lands, as the quoted form does.
    assert acts_of_block('h = "prior_3123"\ngo_to(h)\n') \
        == [("move", "prior_3123")]
    assert acts_of_block('depot = "the depot"\ngo_to(depot)\n') \
        == [("move", "<the depot>")]
    # ...and a POINT still reaches the moving calls only through a ROW,
    # which is what a selection binds.
    corner = Row({"fid": "end_0", "xy": [3411.0, 1468.0]})
    assert acts_of_block("go_to(corner)\n", {"corner": corner}) \
        == [("move", "[3411, 1468]")]

    # 2. indexing a call result (36) -- the call runs, the index picks a
    #    row of it, and a refusal names what the MODEL wrote.
    reply, fr, flights, _d = drive([
        [run('t = find("car")[0]\ngo_to(t)\n')],
        "done",
    ])
    out = tool_results(fr)
    assert flights == [{"target_fid": "car_1"}], flights
    assert out[0]["picked"] == {"t": {"fid": "car_1"}}, out[0]
    assert "_c" not in json.dumps(out[0]), out[0]
    reply, fr, flights, _d = drive([
        [run('t = find("nothing at all")[0]\n')],
        "done",
    ])
    err = tool_results(fr)[0].get("error") or ""
    assert "find('nothing at all')[0] did not bind" in err, err

    # 3. one level of nesting (38): the inner call is its own dispatch and
    #    the outer measures against what it left.
    ops = ops_of_block('r5 = find("house").filter(within=rd)\n',
                       {"rd": "road_1"})
    assert [o["op"] for o in ops] == ["find", "compare", "filter_items"], ops
    # A NESTED SET in a measuring slot is that set's union -- the gap to
    # the nearest house, which is what the spelling plainly means.
    ops = ops_of_block('r5 = find("car", nearest=find("house"))\n')
    assert [o["op"] for o in ops] == ["find", "find", "compare",
                                      "score_by"], ops
    assert ops[2]["args"]["refs"][0]["ref"] == [ops[0]["ref"]], ops


def test_a_failed_step_does_not_unbind_what_the_program_already_bound():
    """`r1 = find("car")` then a look that cannot render is ONE plan."""
    def _broken(_reqs):
        raise RuntimeError("renderer down")

    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nlook(r1)\n')],
        [run("go_to(r1[0])\n")],
        "done",
    ], look_renderer=_broken)
    out = tool_results(fr)
    assert "crop render failed" in (out[0].get("error") or "")
    assert out[0]["handles"] == ["r1"], out[0]
    assert "error" not in out[1], out[1]
    # ...and a grounding miss binds nothing: its program never ran.
    reply, fr, _f, _d = drive([
        [run('r2 = find("car", nearest="nowhere at all")\n')],
        "done",
    ])
    assert "handles" not in tool_results(fr)[0], tool_results(fr)[0]


def test_a_slice_frames_the_head_of_an_ordering():
    """`look(r2[0:20])` -- a badged frame over the head of an ordering."""
    import pytest

    from maps.search.parse_compile.code_surface import CodeError
    (_a, (op,), answer, _p), crops = calls_of_block(
        'look(r2[0:20])\n', {"r2": "r7"})
    assert op["args"]["items"] == "r7[0:20]", op
    # two verbs, one receipt: the window answers nothing, the crop shows it
    assert answer is None and crops == ("crops", op["ref"], None, None)
    # A window is a SET, so it binds, chains and displays like one, and an
    # open lower bound is the head of the ordering.
    (_a, (op,), _ans, _p), = calls_of_block("r5 = r2[:5]\nr5\n", {"r2": "r7"})
    assert op["args"]["items"] == "r7[0:5]", op
    # BOTH ENDS, and no step: a window addresses a range of the ordering,
    # and anything else is refused rather than silently reinterpreted.
    with pytest.raises(CodeError, match=r"\[lo:hi\] with both ends"):
        calls_of_block("r3 = r2[0:20:2]\n", {"r2": "r7"})
    with pytest.raises(CodeError, match=r"\[lo:hi\] with both ends"):
        calls_of_block("r2[10:]\n", {"r2": "r7"})
    with pytest.raises(CodeError, match="name 'r9' is not defined"):
        calls_of_block("r3 = r9[0:5]\n")

    # ...and end to end the head of the ordering is what comes back.
    from maps.search.analyze_scene.geom_program import BROAD_HEAD_ROWS as H
    wide = [{"fid": f"car_{i:03d}", "xy": [float(i), 0.0], "class": "car",
             "search_text": "car"} for i in range(3 * H)]
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nr1[0:5]\n')],
        "done",
    ], cands=wide)
    env = tool_results(fr)[0]["result"]["results"]["r1[0:5]"]
    assert env["fids"] == [f"car_{i:03d}" for i in range(5)], env


def test_a_row_of_a_window_shows_the_row_it_is():
    """w[0] and a loop over w show the window's rows, not the ref's head."""
    cands = [{"fid": f"detected_{i}", "xy": [10.0 * i, 0.0], "class": "car",
              "search_text": "car"} for i in range(6)]
    _, fr, _f, _d = drive([
        [run('r1 = find("car")\nw = r1[2:5]\nw[0]\n')],
        [run('for r in w:\n    pass\nr\n')],
        "done",
    ], cands=cands)
    head, loop = tool_results(fr)[:2]
    assert head["result"]["results"]["w[0]"]["fids"] == ["detected_2"], head
    # the loop's name holds its last row
    assert loop["shown"] == {"r": "detected_4"}, loop


def test_the_legend_documents_every_call_the_surface_serves():
    """A call the parser accepts and the legend omits is a capability the model has no
    way to learn it has."""
    import inspect

    from maps.search.parse_compile.prompts import CODE_DOC
    from maps.search.parse_compile.python_surface import (
        MissionPython,
        Rows,
        served_calls,
    )

    names = served_calls()
    assert len(names) >= 9, names                    # vacuity floor
    assert not [n for n in names if f"{n}(" not in CODE_DOC], names
    # ...and every keyword the executing functions take, bar order(reverse=),
    # which is accepted and not advertised.
    fns = [getattr(MissionPython, f"_call_{n}") for n in names] + [Rows.order, Rows.filter]
    kws = {n for f in fns for n, p in inspect.signature(f).parameters.items()
           if p.kind is p.KEYWORD_ONLY} - {"reverse"}
    assert len(kws) >= 7, kws
    assert not [k for k in kws if f"{k}=" not in CODE_DOC], sorted(kws)
    assert "TOUCHES it -- distance 0" in CODE_DOC
    # the gap cut, the union rule (a set measures from its NEAREST member),
    # the cardinal cut's order, and a blank cell as unknown
    for phrase in ("dwithin=25", "NEAREST member", "NEAREST first", "blank"):
        assert phrase in CODE_DOC, phrase
    assert "one crop PER ROW" in CODE_DOC
    # A derived scan cannot notice its disappearance; this can.
    assert "To reach a PART" in CODE_DOC
    # the visual-adjective occasion: a cue word no column carries is read
    # off crops, not off the prior
    assert "choose by what the crops show" in CODE_DOC
    # the set form: a SET is one place to cover, and a NAME is the whole
    # street
    assert "a SET is one place to cover" in CODE_DOC
    assert "the name is the whole street" in CODE_DOC
    # THE DE-ENDORSEMENT PAIR (system prompt): a sort is evidence of its column, never
    # of identity, and "know" has a standard for cue words no column carries.
    from maps.search.parse_compile.prompts import CODE_SYSTEM_PROMPT
    assert "settled by look(), not by" in CODE_SYSTEM_PROMPT
    assert "known only from a look, never from rank" in CODE_SYSTEM_PROMPT
    assert "never that its head is the thing" in CODE_SYSTEM_PROMPT
    # the facing word, the cut's compass value and the axis convention
    from maps.search.parse_compile.prompts import served_tools
    desc = served_tools()[0]["function"]["description"]
    assert '"facing <place>"' in desc, desc
    assert 'filter(facing=<place>, direction="south")' in desc, desc
    assert "+x runs EAST" in desc and "+y runs NORTH" in desc


def test_a_block_written_into_the_reply_channel_is_repaired_not_an_ending():
    reply, fr, flights, _d = drive([
        'r1 = find("car")\ngo_to(r1[0])',       # the dropped block
        [run('r1 = find("car")\n')],          # re-sent through the tool
        [run("go_to(r1[0])\n")],
        "the car by the depot.",
    ])
    assert reply.ended_by == "reply"
    assert len(flights) == 1
    nudges = [m for m in fr.messages
              if m.get("role") == "user"
              and "not through the run tool" in str(m.get("content"))]
    assert len(nudges) == 1, nudges


def test_a_second_block_in_the_reply_channel_still_ends_the_mission():
    """A stuck emitter cannot loop the mission on nudges."""
    reply, fr, flights, _d = drive(['go_to(r1[0])', 'go_to(r1[0])'])
    assert reply.ended_by == "reply"
    assert flights == []
    assert reply.turns_used == 2


def test_prose_that_narrates_a_call_is_a_real_ending():
    """Both halves of the detector, each killing what the other lets by."""
    from maps.search.parse_compile.reasoner_loop import _code_reply

    assert _code_reply('r1 = find("car")\ngo_to(r1[0])')
    assert _code_reply('```python\ngo_to(r1[0])\n```')
    assert not _code_reply("I flew to the church with go_to(r5); that is it.")
    assert not _code_reply("done")
    assert not _code_reply("The pool house is prior_7.")
    assert not _code_reply("")


def test_find_takes_the_colour_ordering_the_executor_always_had():
    """`find(class, color=word)` is one call, and it is one sort."""
    def _ops(src):
        return [(o["op"], o["args"]) for o in ops_of_block(src)]

    one = _ops('r1 = find("house", color="red")\n')
    two = _ops('r1 = find("house")\nr2 = r1.order(color="red")\n')
    assert [o for o, _a in one] == ["find", "score_by"], one
    assert [a for _o, a in one] == [a for _o, a in two], (one, two)

    # ONE score_by, colour last, distances first.
    fused = _ops('r1 = find("house", nearest="A", south_of="B", color="gray")\n')
    sorts = [a for o, a in fused if o == "score_by"]
    assert len(sorts) == 1, fused
    assert [e["key"] for e in sorts[0]["emphasis"]] == [
        "distance to a", "distance to b", "color_gray"], sorts

    # and it ORDERS, never cuts: every row the find matched is still there
    assert not [o for o, a in fused
                if o == "filter_items" and "color" in str(a.get("key"))], fused


# --- explore(set): many members, ONE chained pass -------------------------

_WILLOW_CANDS = [
    {"fid": "w1", "xy": [10.0, 0.0], "name": "Willow Avenue",
     "search_text": "willow avenue"},
    {"fid": "w2", "xy": [200.0, 0.0], "name": "Willow Avenue",
     "search_text": "willow avenue"},
    {"fid": "w3", "xy": [400.0, 0.0], "name": "Willow Avenue",
     "search_text": "willow avenue"},
    {"fid": "car_1", "xy": [20.0, 20.0], "class": "car",
     "search_text": "car"},
]

_WILLOW_STOPS = {"w1": [[0.0, 0.0], [100.0, 0.0]],
                 "w2": [[150.0, 0.0], [250.0, 0.0]],
                 "w3": [[350.0, 0.0], [450.0, 0.0]]}


def test_an_explore_of_a_NAME_covers_every_segment_bearing_it():
    """explore("Willow Avenue") is the whole street, not its first fragment."""
    import json as _json

    from maps.util.xy_token import parse_xy_token
    reply, out, flights, asked, _rend = explore_drive([
        [run('explore("Willow Avenue")\n')],
        "done",
    ], stops=_WILLOW_STOPS, cands=_WILLOW_CANDS)
    assert "error" not in out[0], out[0]
    assert {f for f, _ in asked} == {"w1", "w2", "w3"}, asked
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == [[0.0, 0.0], [100.0, 0.0], [150.0, 0.0],
                     [250.0, 0.0], [350.0, 0.0], [450.0, 0.0]], flown
    ack = out[0]["explored"]
    assert ack["segments"] == 3, ack
    assert "ended_at" not in ack, ack
    assert "across 3 segments" in (ack.get("note") or ""), ack
    assert "best name match" not in _json.dumps(out[0]), out[0]


def test_an_explore_of_a_handle_chains_members_and_reverses_a_far_end():
    """A member whose FAR end is nearer the chain tail flies reversed, so consecutive
    members join where they touch and the pass stays a line."""
    from maps.util.xy_token import parse_xy_token
    reply, out, flights, _a, _rend = explore_drive([
        [run('r1 = find("car")\n')],
        [run('explore(r1)\n')],
        "done",
    ], stops={"car_1": [[100.0, 200.0], [200.0, 200.0]],
              "car_2": [[500.0, 200.0], [300.0, 200.0]]})
    assert "error" not in out[1], out[1]
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == [[100.0, 200.0], [200.0, 200.0],
                     [300.0, 200.0], [500.0, 200.0]], flown
    assert out[1]["explored"]["segments"] == 2, out[1]["explored"]


def test_a_set_pass_covers_a_long_street_rather_than_husbanding_stops():
    """The bound is a RUNAWAY backstop, not a budget the pass husbands."""
    from maps.util.xy_token import parse_xy_token
    long_stops = {
        "w1": [[float(x), 0.0] for x in range(0, 1200, 100)],
        "w2": [[float(x), 0.0] for x in range(1200, 2400, 100)],
        "w3": [[float(x), 0.0] for x in range(2400, 3600, 100)]}
    reply, out, flights, _a, _rend = explore_drive([
        [run('explore("Willow Avenue")\n')],
        "done",
    ], stops=long_stops, cands=_WILLOW_CANDS)
    assert "error" not in out[0], out[0]
    # (The drive harness has a 32-flight backstop of its own, so the exact count is its
    # limit and not the pass's.)
    assert len(flights) > 24, len(flights)
    note = out[0]["explored"].get("note") or ""
    # NO BUDGET FACT: not the ceiling, not "one pass", not advice to do less.
    for banned in ("SAMPLED", "Narrow it", "one pass flies"):
        assert banned not in note, note
    assert "covered" in note, note
    # The PASS plans the whole street end to end; the drive harness's own
    # 32-flight backstop is what stops short of it, not the pass.
    assert out[0]["explored"]["stops"] > 24, out[0]["explored"]
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown[0] == [0.0, 0.0], flown


def test_a_set_pass_dedupes_the_shared_endpoint_of_touching_members():
    """Two segments that join share a stop; the pass flies it once."""
    from maps.util.xy_token import parse_xy_token
    reply, out, flights, _a, _rend = explore_drive([
        [run('explore("Willow Avenue")\n')],
        "done",
    ], stops={"w1": [[0.0, 0.0], [100.0, 0.0]],
              "w2": [[110.0, 0.0], [210.0, 0.0]],
              "w3": [[320.0, 0.0], [420.0, 0.0]]},
        cands=_WILLOW_CANDS)
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    # [110, 0] sits within half a spacing (50) of the kept [100, 0]
    assert [110.0, 0.0] not in flown, flown
    assert len(flown) == 5, flown


def test_go_to_of_a_set_is_refused_toward_explore():
    """go_to still lands on ONE thing; the remedy names the set form."""
    reply, fr, flights, _d = drive([
        [run('r1 = find("car")\ngo_to(r1)\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "error" in out[0], out[0]
    assert "explore(r1) to cover the whole set" in out[0]["error"], out[0]
    assert flights == [], flights


def test_an_explore_of_a_LIST_of_rows_covers_their_union():
    reply, out, flights, _a, _rend = explore_drive([
        [run('r1 = find("car")\n')],
        [run('explore([r1[0], r1[1]])\n')],
        "done",
    ], stops={"car_1": [[100.0, 200.0], [200.0, 200.0]],
              "car_2": [[500.0, 200.0], [300.0, 200.0]]})
    assert "error" not in out[1], out[1]
    assert out[1]["explored"]["segments"] == 2, out[1]["explored"]
    assert len(flights) == 4, flights


@pytest.mark.parametrize("move,want", [
    ("explore(e)", ["[80, 100]", "[120, 100]"]),
    ('explore(["end_0", "end_1"])', ["[80, 100]", "[120, 100]"]),
    ('go_to("end_0")', ["[80, 100]"]),
    ("go_to(end_1)", ["[120, 100]"]),
])
def test_an_explore_of_ends_of_flies_each_terminus_as_its_point(move, want):
    """A terminus is nowhere on the surface; the pass flies its point, as
    go_to(e[0]) does, and so does its fid typed."""
    from maps.search.analyze_scene.surface import build_surface_from_graph
    from maps.search.tests.fakes import ROADS, priors
    cands, anchors = build_surface_from_graph(
        {**priors(), **ROADS}.items(), (0.0, 0.0), 1.0)
    reply, out, flights, _a, _rend = explore_drive([
        [run(f'e = ends_of("Long Road")\n{move}\n')],
        "done",
    ], stops=None, cands=cands + anchors, obs_span_m=40.0)
    assert "error" not in out[0], out[0]
    assert [f["target_fid"] for f in flights] == want


def test_a_set_pass_over_footprintless_members_says_it_has_none():
    reply, out, flights, _a, _rend = explore_drive([
        [run('r1 = find("car")\n')],
        [run('explore(r1)\n')],
        "done",
    ], stops=None, obs_span_m=20.0)
    note = out[1]["explored"]["note"]
    assert note.startswith(
        "r1 has no footprint to cover, so this pass is its 2 stops."), note


# --- pixels are earned by flying: the look coverage gate ------------------

_GATE_CANDS = [
    {"fid": "car_1", "xy": [110.0, 205.0], "class": "car",
     "search_text": "car"},                    # near the pass's stops
    {"fid": "car_2", "xy": [500.0, 900.0], "class": "car",
     "search_text": "car"},                    # far from any stop
]


def test_a_look_before_any_flight_refuses_toward_explore():
    import json as _json
    reply, out, flights, _a, rendered = explore_drive([
        [run('r1 = find("car")\nlook(r1[0])\n')],
        "done",
    ], cands=_GATE_CANDS, obs_minor_m=100.0, with_renderer=True)
    assert flights == [], flights
    assert all(not r for r in rendered), rendered
    blob = _json.dumps(out[0])
    assert "has never been under the sensor" in blob, out[0]
    assert "explore the place" in blob, out[0]


def test_a_pass_earns_exactly_the_pixels_it_covered():
    """After a pass, a point inside a flown view square renders; ground no flight
    covered still refuses."""
    import json as _json
    reply, out, flights, _a, rendered = explore_drive([
        [run('explore("prior_7")\n')],
        [run('look([110, 205])\n')],  # 11 m from the pass's first stop
        [run('look([500, 900])\n')],  # no stop near
        "done",
    ], cands=_GATE_CANDS, obs_minor_m=100.0, with_renderer=True)
    served = [r for reqs in rendered for r in reqs]
    assert len(served) == 1, served
    blob = _json.dumps(out[2])
    assert "has never been under the sensor" in blob, out[2]


def test_without_a_stated_footprint_the_gate_stays_off():
    """Sensorless runs state no short side, and a gate keyed on an invented number would
    refuse what the engine could serve."""
    reply, out, flights, _a, rendered = explore_drive([
        [run('r1 = find("car")\nlook(r1[0])\n')],
        "done",
    ], cands=_GATE_CANDS, obs_minor_m=None, with_renderer=True)
    served = [r for reqs in rendered for r in reqs]
    assert [r["center_fid"] for r in served] == ["car_1"], served


# --- strata: the find's match tiers as live attribute views ---------------

_TIER_CANDS = (
    [{"fid": f"d{i}", "xy": [float(100 * i), 0.0], "class": "parking",
      "search_text": "parking"} for i in range(3)]
    + [{"fid": f"p{i}", "xy": [float(100 * i), 500.0],
        "search_text": "amenity parking",
        "tags": {"amenity": "parking"}} for i in range(2)])


def test_strata_are_live_views_and_the_census_says_so():
    """r1.exact / r1.tagged split the find's own tiers; the receipt's census names them
    as structure, not a float column."""
    import json as _json
    reply, out, flights, _a, _r = explore_drive([
        [run('parks = find("parking")\n'
              'print(len(parks), len(parks.exact), len(parks.tagged))\n')],
        "done",
    ], cands=_TIER_CANDS)
    assert "error" not in out[0], out[0]
    assert out[0]["printed"] == "5 3 2", out[0]["printed"]
    assert "tiers: 3 exact, 2 tagged" in _json.dumps(out[0]), out[0]


def test_explore_of_a_stratum_covers_exactly_its_rows():
    """explore(parks.exact) is the workflow's own spelling: the pass walks the rows that
    ARE the thing and none of the tag-carriers."""
    stops = {f"d{i}": [[float(100 * i), 0.0]] for i in range(3)}
    reply, out, flights, asked, _r = explore_drive([
        [run('parks = find("parking")\nexplore(parks.exact)\n')],
        "done",
    ], cands=_TIER_CANDS, stops=stops)
    assert "error" not in out[0], out[0]
    assert out[0]["explored"]["segments"] == 3, out[0]["explored"]
    assert sorted({f for f, _ in asked}) == ["d0", "d1", "d2"], asked


def test_a_stratum_of_tierless_rows_refuses_with_the_teaching():
    """A browse carries no match tiers; the refusal says where strata come from instead
    of serving a silent empty set."""
    reply, out, flights, _a, _r = explore_drive([
        [run('everything = find()\nprint(len(everything.exact))\n')],
        "done",
    ], cands=_TIER_CANDS)
    assert "error" in out[0], out[0]
    assert "no match tiers" in out[0]["error"], out[0]["error"]


# --- the pass shows its catch ---------------------------------------------

def test_a_pass_serves_its_catch_as_crops():
    results, _graph, added = grow_drive([
        [run('explore("prior_7")\n')],
        "done",
    ])
    ex = results[0].get("explored") or {}
    assert ex.get("catch"), results[0]
    assert ex["catch"].get("delivered"), ex["catch"]


def test_a_looks_marks_reach_the_renderer_on_every_crop_of_a_set():
    # The mark has to be IN FRAME to be drawn -- the renderer drops a mark
    # whose footprint never reached the pixels -- so the fixture puts the
    # cars and the thing marked inside one crop window.
    _close = [
        {"fid": "car_1", "xy": [110.0, 205.0], "class": "car",
         "search_text": "car"},
        {"fid": "car_2", "xy": [118.0, 209.0], "class": "car",
         "search_text": "car"},
        {"fid": "lamp_1", "xy": [114.0, 207.0], "class": "lamp",
         "search_text": "lamp"},
    ]
    reply, out, flights, _a, rendered = explore_drive([
        [run('r1 = find("car")\nm = find("lamp")\n'
              'look(r1[0:2], marks=m[0])\n')],
        "done",
    ], cands=_close, obs_minor_m=None, with_renderer=True)
    def _context(reqs):
        # A crop ALWAYS marks its own subject, so the question is whether a
        # mark names something ELSE -- the anchor the caller asked for.
        return [r for r in reqs
                if [m for m in (r.get("marks") or [])
                    if str(m.get("fid")) != str(r.get("center_fid"))]]

    served = [r for reqs in rendered for r in reqs]
    # EVERY crop of the set carries it, not just a lone one.
    assert len(served) >= 2, served
    assert len(_context(served)) == len(served), [
        (r.get("center_fid"), r.get("marks")) for r in served]

    # VACUITY FLOOR: the same look WITHOUT marks= must carry none, or a
    # renderer that marks everything would pass the assertion above.
    _r2, _o2, _f2, _a2, rendered2 = explore_drive([
        [run('r1 = find("car")\nlook(r1[0:2])\n')],
        "done",
    ], cands=_close, obs_minor_m=None, with_renderer=True)
    served2 = [r for reqs in rendered2 for r in reqs]
    assert len(served2) >= 2, served2
    assert not _context(served2), [
        (r.get("center_fid"), r.get("marks")) for r in served2]


def test_a_pass_hands_every_stop_the_stops_behind_it():
    """Each item an explore flies carries ``pass_ahead``, the stops the pass still has
    to fly after it, so the flight mission can post the whole pass to the vehicle
    from the first leg."""
    _r, _o, flights, _a, _rend = explore_drive([
        [run('explore("prior_7")\n')],
        "done",
    ])
    assert len(flights) == len(STOPS) > 1
    for k, f in enumerate(flights):
        assert f["pass_ahead"] == STOPS[k + 1:], (k, f)
    _r, out, flights, _a, _rend = explore_drive([
        [run('go_to("prior_7")\n')],
        "done",
    ])
    assert len(flights) == 1 and flights[0]["target_fid"] == "prior_7"
    assert "pass_ahead" not in flights[0], flights
    assert "explored" not in out[0], out[0]
    assert out[0]["flown"]["flown"][0]["target"] == "prior_7", out[0]


def test_a_pass_takes_a_stop_the_recording_holds_first():
    from maps.util.xy_token import parse_xy_token

    asked: list[list] = []

    def held_now(stops):
        asked.append([list(xy) for _, xy in stops])
        return [list(xy) == STOPS[2] for _, xy in stops]

    _r, _o, flights, _a, _rend = explore_drive([
        [run('explore("prior_7")\n')],
        "done",
    ], held_now=held_now)
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == [STOPS[2], STOPS[0], STOPS[1]]
    assert [f["pass_ahead"] for f in flights] == [[STOPS[0], STOPS[1]], [STOPS[1]], []]
    assert asked == [[STOPS[0], STOPS[1], STOPS[2]], [STOPS[0], STOPS[1]]]


def test_two_unnamed_results_with_one_label_stay_two_results():
    """``find(A).filter(within=find(A))``: both sets carry the label
    "find('A')", and each slot still reads its own set."""
    ops = ops_of_block('r1 = find("car").filter(within=find("car"))\n')
    first, second, cmp_op = ops[0], ops[1], ops[2]
    assert cmp_op["args"]["items"] == first["ref"], ops
    assert cmp_op["args"]["refs"] == [
        {"ref": [second["ref"]], "as": "find('car')"}], ops


def test_more_of_on_a_one_block_handle_says_to_show_it():
    """A handle that fits one block has no page past it: more_of says how
    to see it, never that its rows are already served (a bound find that
    was never shown has served none)."""
    reply, fr, _f, _d = drive([
        [run('cars = find("car")\nd = find("depot")\nd')],
        [run("more_of(cars)\n")],
        "done",
    ])
    err = tool_results(fr)[1]["error"]
    assert err == "cars holds 2 row(s), one block: show it with a bare `cars` line.", err
