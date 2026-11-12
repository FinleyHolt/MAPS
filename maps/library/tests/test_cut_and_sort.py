"""``filter`` cuts at a STATED gap as well as at contact, and ``order`` sorts."""

from __future__ import annotations

import pytest

from maps.library.op_compiler import CodeError
from maps.mission.tests.fakes import (
    drive, quiet_env, raising_ops_of_block, run, tool_results)


def _ops(src: str) -> list[dict]:
    # The surface has no parser: a block becomes ops only by EXECUTING.
    return raising_ops_of_block(src)


def _cut(ops: list[dict]) -> dict:
    cuts = [o for o in ops if o["op"] == "filter_items"]
    assert len(cuts) == 1, [o["op"] for o in ops]
    return cuts[0]["args"]


# --------------------------------------------------------------------------
# The gap
# --------------------------------------------------------------------------


def test_ddwithin_cuts_the_distance_column_at_the_stated_metres():
    ops = _ops('r1 = find("car")\n'
               'r2 = r1.filter(near="Poppy Boulevard", dwithin=25)\n')
    assert [o["op"] for o in ops] == ["find", "compare", "filter_items"], \
        [o["op"] for o in ops]
    a = _cut(ops)
    assert a["op"] == "le", a
    assert a["value"] == 25.0, a
    assert a["key"].startswith("distance to "), a


def test_within_still_cuts_at_contact():
    ops = _ops('r1 = find("car")\n'
               'r2 = r1.filter(within="Poppy Boulevard")\n')
    a = _cut(ops)
    assert (a["op"], a["value"]) == ("le", 0), a


def test_a_whole_class_is_one_place_to_measure_from():
    """The union reading, end to end: ``near=`` takes the find, not a row index of it,
    and still compiles to a single distance cut."""
    ops = _ops('houses = find("house")\n'
               'cars = find("car")\n'
               'outside = cars.filter(near=houses, dwithin=25)\n')
    assert [o["op"] for o in ops] == \
        ["find", "find", "compare", "filter_items"], [o["op"] for o in ops]
    assert _cut(ops)["value"] == 25.0


def test_a_float_gap_survives_as_a_float():
    assert _cut(_ops('r1 = find("car")\n'
                     'r2 = r1.filter(near="Poppy Boulevard", dwithin=7.5)\n')
                )["value"] == 7.5


# --------------------------------------------------------------------------
# The refusals -- each names the spelling that would have worked
# --------------------------------------------------------------------------


@pytest.mark.parametrize("src,phrase", [
    ('r1 = find("car")\nr2 = r1.filter(within="A", near="B")\n',
     "not both"),
    ('r1 = find("car")\nr2 = r1.filter()\n',
     "filter needs what to cut against"),
    ('r1 = find("car")\nr2 = r1.filter(within="A", dwithin=25)\n',
     "dwithin= is the gap of a near= cut"),
    ('r1 = find("car")\nr2 = r1.filter(east_of="A", dwithin=25)\n',
     "dwithin= is the gap of a near= cut"),
    ('r1 = find("car")\nr2 = r1.filter(facing="A", direction="south", dwithin=25)\n',
     "dwithin= is the gap of a near= cut"),
    ('r1 = find("car")\nr2 = r1.filter(near="A")\n',
     "needs the gap to keep"),
    ('r1 = find("car")\nr2 = r1.filter(near="A", dwithin="soon")\n',
     "not a number"),
    ('r1 = find("car")\nr2 = r1.filter(near="A", dwithin=-5)\n',
     "cannot be negative"),
    ('r1 = find("car")\nr2 = r1.filter(lambda r: True, north_of="A")\n',
     "not both"),
])
def test_the_refusal_names_the_exit(src, phrase):
    with pytest.raises(CodeError) as e:
        _ops(src)
    assert phrase in str(e.value), str(e.value)


_CUTS = ("near=/within=/dwithin=/facing= with direction=, "
         "east_of=/north_of=/south_of=/west_of=")
_PRED_OR_CUT = f"filter takes a predicate OR {_CUTS}, not both"


def test_an_unknown_filter_keyword_reads_the_same_cut_list_as_the_predicate_refusal():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = r1.filter(beside="A")\n')
    assert f"filter takes {_CUTS} or a predicate; beside= is none of them." in str(e.value), str(e.value)


def test_a_bare_filter_reads_one_refusal_on_every_receiver():
    for src in ('r1 = find("car")\nr2 = r1.filter()\n', 'r2 = a.filter()\n',
                'r2 = a.filter(lambda r: True).filter()\n'):
        with pytest.raises(CodeError) as e:
            raising_ops_of_block(src, bound={"a": "detected_1"})
        assert "filter needs what to cut against" in str(e.value), (src, str(e.value))


def test_a_predicate_and_a_cut_kwarg_read_one_refusal_on_every_receiver():
    from maps.library.rows import Row, RowList
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = r1.filter(lambda r: True, facing="A")\n')
    assert _PRED_OR_CUT in str(e.value), str(e.value)
    row = Row({"fid": "detected_1"})
    for rows in (row, RowList([row])):
        with pytest.raises(CodeError) as e:
            rows.filter(lambda r: True, facing="A")
        assert str(e.value) == _PRED_OR_CUT, str(e.value)



@pytest.mark.parametrize("call", [
    '.filter("x")', '.order("x")', '.order(direction="north", reverse=True)',
    '.order(lambda r: 1, nearest="A")', '.order(direction="north", nearest="A")'])
def test_a_row_and_a_list_refuse_a_cut_or_sort_with_the_sets_own_text(call):
    from maps.library.rows import Row, RowList
    with pytest.raises(CodeError) as e:
        _ops(f'r1 = find("car")\nr1{call}\n')
    on_set = str(e.value)
    row = Row({"fid": "detected_1"})
    for rows in (row, RowList([row])):
        with pytest.raises(CodeError) as e:
            eval(f"rows{call}")
        assert str(e.value) in on_set, (rows, str(e.value), on_set)


def test_a_gapless_near_cut_points_at_an_ordering_the_surface_accepts():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = r1.filter(near="A")\n')
    # find(near=) without dwithin= is refused too, so the exit is order()
    assert "find(near=" not in str(e.value), str(e.value)
    assert "rows.order(nearest=<place>)" in str(e.value), str(e.value)
    _ops('r1 = find("car")\nr2 = r1.order(nearest="A")\n')


# --------------------------------------------------------------------------
# One ordering, many cuts
# --------------------------------------------------------------------------


def test_order_refuses_two_keys_because_a_table_sorts_by_one():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\n'
             'r2 = r1.order(nearest="Poppy Boulevard", color="red")\n')
    assert "ONE key" in str(e.value), str(e.value)
    # and the refusal names the exit: cut on one, order by the other
    assert ".filter(" in str(e.value), str(e.value)


def test_a_cut_and_a_sort_chain_into_one_expression():
    ops = _ops('cars = find("car")\n'
               'houses = find("house")\n'
               'best = cars.filter(near=houses, dwithin=25)'
               '.order(nearest="Poppy Boulevard")\n')
    kinds = [o["op"] for o in ops]
    # the cut is compare+filter_items; the sort is the score_by after it
    assert kinds.count("filter_items") == 1, kinds
    assert kinds[-1] == "score_by", kinds
    assert kinds.index("filter_items") < len(kinds) - 1, kinds


def test_the_retired_free_calls_name_where_they_went():
    """A model reaching for narrow() gets the method, not a bare NameError."""
    from maps.library.block_scan import _MOVED_TO_METHOD

    assert set(_MOVED_TO_METHOD) == {"narrow", "order_by", "gap"}
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = narrow(r1, near="A", dwithin=25)\n')
    assert "rows.filter(" in str(e.value), str(e.value)


# --------------------------------------------------------------------------
# The spellings the model reaches for
# --------------------------------------------------------------------------


def test_find_cuts_at_a_stated_gap_where_the_model_puts_it():
    ops = _ops('r1 = find("car", near="Poppy Boulevard", dwithin=25)\n')
    kinds = [o["op"] for o in ops]
    assert kinds[0] == "find", kinds
    assert "filter_items" in kinds, kinds
    cut = [o for o in ops if o["op"] == "filter_items"][0]["args"]
    assert cut["op"] == "le" and cut["value"] == 25.0, cut


def test_ddwithin_on_find_needs_the_place_it_measures_from():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car", dwithin=25)\n')
    assert "near=" in str(e.value), str(e.value)


def test_colour_never_cuts_and_the_refusal_names_the_sort():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = r1.filter(color="red")\n')
    msg = str(e.value)
    assert "never CUTS" in msg and 'order(color=' in msg, msg

    # ...and a single Row cuts through the same guard, not a bare TypeError.
    from maps.library.rows import Row
    with pytest.raises(CodeError) as e:
        Row({"fid": "detected_1"}).filter(color="red")
    assert "never CUTS" in str(e.value), str(e.value)
    # ...and so does a list a predicate filter returned
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = r1.filter(lambda r: True).filter(color="red")\n')
    assert "never CUTS" in str(e.value), str(e.value)


# --------------------------------------------------------------------------
# One word, one operation: near= CUTS, nearest= ORDERS
# --------------------------------------------------------------------------


def test_near_on_find_cuts_and_refuses_to_be_an_ordering():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car", near="Poppy Boulevard")\n')
    msg = str(e.value)
    assert "CUTS" in msg and "dwithin=" in msg, msg
    assert "nearest=" in msg, msg          # the refusal names the ordering


def test_nearest_orders_and_drops_nothing():
    ops = _ops('r1 = find("car", nearest="Poppy Boulevard")\n')
    kinds = [o["op"] for o in ops]
    assert "score_by" in kinds, kinds
    assert "filter_items" not in kinds, kinds


def test_the_two_spellings_are_one_cut_and_one_sort():
    """A cut and a sort in one find: near=/dwithin= drops, nearest= orders what
    survives."""
    ops = _ops('r1 = find("car", near="Poppy Boulevard", dwithin=25,\n'
               '         nearest="Poppy Boulevard")\n')
    kinds = [o["op"] for o in ops]
    assert "filter_items" in kinds and "score_by" in kinds, kinds
    assert kinds.index("filter_items") < kinds.index("score_by"), kinds


def test_a_row_serves_no_second_point_to_subtract():
    """There is ONE distance here and it is the footprint gap, so a row must not hand
    over a second point the model can difference by hand."""
    from maps.library.rows import Row

    r = Row({"fid": "detected_1", "xy": [10.0, 20.0],
             "mask_centroid": [11.0, 21.0], "class": "car"})
    assert "mask_centroid" not in r.keys(), r.keys()
    assert r.get("mask_centroid") is None
    with pytest.raises(AttributeError):
        r.mask_centroid
    # the point that IS a place stays reachable
    assert r.xy == [10.0, 20.0] and r.x == 10.0


def test_a_predicate_filter_is_not_a_one_way_door():
    from maps.library.rows import Row, RowList

    seen = []

    class _MP:
        def rows_order(self, items, **kw):
            seen.append(("order", list(items), kw)); return "ordered"

        def rows_filter(self, items, **kw):
            seen.append(("filter", list(items), kw)); return "cut"

    mp = _MP()
    rows = RowList([Row({"fid": "detected_1", "class": "car"}, mp=mp),
                    Row({"fid": "detected_2", "class": "car"}, mp=mp)])

    assert rows.order(color="red") == "ordered"
    assert seen[-1][0] == "order" and seen[-1][2] == {"nearest": None, "color": "red"}
    assert len(seen[-1][1]) == 2          # the materialised subset, by row

    assert rows.filter(near="Poppy Boulevard", dwithin=25) == "cut"
    assert seen[-1][0] == "filter"
    assert seen[-1][2] == {"near": "Poppy Boulevard", "dwithin": 25}

    # the predicate forms still work and still chain
    assert isinstance(rows.filter(lambda r: r["fid"] == "detected_1"), RowList)
    assert len(rows.order(lambda r: r["fid"])) == 2


def test_an_empty_list_cuts_and_orders_to_an_empty_list():
    _r, fr, _f, _d = drive([[run(
        'p = find("car").filter(lambda r: r.get("class") == "pool")\n'
        'x = p.order(nearest="prior_7").filter(near="prior_7", dwithin=5)\n'
        'print("after", len(x))\n')], "done"])
    out = tool_results(fr)[0]
    assert "error" not in out and out["printed"] == "after 0", out


def test_ddwithin_cuts_against_whichever_anchor_was_named():
    """``find("parking", nearest=[X], dwithin=500)`` means rank by the gap to X and drop
    past 500 -- coherent, unambiguous, and what anyone would write."""
    ops = _ops('r1 = find("parking", nearest="Willow Avenue", dwithin=500)\n')
    kinds = [o["op"] for o in ops]
    assert "filter_items" in kinds, kinds          # it cut
    assert "score_by" in kinds, kinds              # and it still ordered
    cut = [o for o in ops if o["op"] == "filter_items"][0]["args"]
    assert cut["value"] == 500.0, cut


def test_a_list_in_near_cuts_against_its_nearest_member_as_filter_does():
    ops = _ops('r1 = find("car", near=["prior_7", "prior_1"], dwithin=500)\n')
    assert [o["op"] for o in ops] == ["find", "compare", "filter_items"], ops
    cmp = [o for o in ops if o["op"] == "compare"][0]["args"]
    assert cmp["refs"][0]["ref"] == ["prior_7", "prior_1"], cmp
    assert _cut(ops)["value"] == 500.0


def test_dwithin_cuts_against_an_x_y_point_in_near_or_nearest():
    for kw in ("near", "nearest"):
        ops = _ops(f'r1 = find("car", {kw}=[7900, 5600], dwithin=300)\n')
        assert _cut(ops)["value"] == 300.0, ops


def test_a_list_in_nearest_with_dwithin_refuses_and_names_near():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car", nearest=["prior_7", "prior_1"], dwithin=500)\n')
    msg = str(e.value)
    assert "a LIST in nearest= ranks by several distances" in msg, msg
    assert "near=<the set>" in msg, msg


def test_ddwithin_with_no_anchor_at_all_still_refuses_and_names_both():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("parking", dwithin=500)\n')
    msg = str(e.value)
    assert "near=" in msg and "nearest=" in msg, msg


def test_a_set_handed_to_go_to_is_told_to_index_not_told_it_is_a_coordinate():
    from maps.library.rows import Row
    rows = [Row({"fid": "detected_68"}), Row({"fid": "detected_69"})]
    with pytest.raises(CodeError) as e:
        _ops('go_to(x)\n')          # unbound -> different path; use the API
    from maps.library import interpreter as ps
    mp = ps.MissionPython(quiet_env())
    with pytest.raises(CodeError) as e:
        mp.library.go_to(rows)
    msg = str(e.value)
    assert "set of 2" in msg and "[0]" in msg, msg
    assert "not a coordinate" not in msg, msg


# --------------------------------------------------------------------------
# Direction, text to text
# --------------------------------------------------------------------------


def test_a_cardinal_cuts_on_a_set_in_hand_not_only_inside_find():
    ops = _ops('r1 = find("house")\n'
               'r2 = r1.filter(west_of="88th Street")\n')
    kinds = [o["op"] for o in ops]
    assert "compare" in kinds and "filter_items" in kinds, kinds
    cut = [o for o in ops if o["op"] == "filter_items"][0]["args"]
    assert cut["op"] == "on_side", cut
    assert "west of" in cut["key"], cut


def test_cardinal_cuts_chain():
    ops = _ops('r1 = find("house")\n'
               'r2 = r1.filter(west_of="88th Street", north_of="Poppy Boulevard")\n')
    cuts = [o for o in ops if o["op"] == "filter_items"]
    assert len(cuts) == 2, [o["op"] for o in ops]
    assert all(c["args"]["op"] == "on_side" for c in cuts), cuts


def test_a_cardinal_is_its_own_cut_and_says_so_beside_a_gap():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("house")\n'
             'r2 = r1.filter(near="A", dwithin=25, west_of="B")\n')
    assert "ONE relation at a time" in str(e.value), str(e.value)


@pytest.mark.parametrize("src", [
    'r1 = find("house", within=25)\n',
    'r1 = find("house")\nr2 = r1.filter(within=25)\n',
    'r1 = find("house")\nr2 = r1.filter(within=25.0)\n',
])
def test_a_distance_in_within_is_refused_and_names_dwithin(src):
    with pytest.raises(CodeError) as e:
        _ops(src)
    msg = str(e.value)
    assert "dwithin=" in msg, msg
    assert "TOUCHES" in msg, msg


def test_the_two_spellings_still_do_their_own_jobs():
    """Vacuity floor on the guard above: it must refuse the swap without refusing either
    correct spelling."""
    touch = _cut(_ops('r1 = find("house")\nr2 = r1.filter(within="G Ave")\n'))
    assert touch["op"] == "le" and touch["value"] == 0, touch
    gap = _cut(_ops('r1 = find("house")\n'
                    'r2 = r1.filter(near="G Ave", dwithin=25)\n'))
    assert gap["op"] == "le" and gap["value"] == 25, gap


def _served_error(code: str) -> str:
    _r, fr, _f, _d = drive([[run(code)], "done"])
    return tool_results(fr)[0]["error"]


def test_a_set_read_as_a_row_names_the_row_to_read_it_off():
    err = _served_error('cars = find("car")\nprint(cars.xy)\n')
    assert "cars is a set of 2 row(s)" in err and "cars[0].xy" in err, err


def test_a_keyword_spelled_as_a_method_names_the_call():
    err = _served_error('cars = find("car")\ncars.nearest("prior_7")\n')
    assert "cars.order(nearest=...)" in err, err


def test_filter_nearest_names_the_order_it_is():
    err = _served_error('find("car").filter(nearest="prior_7")\n')
    assert "nearest= ORDERS and drops nothing: rows.order(nearest=<place>)" in err, err
