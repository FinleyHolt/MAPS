"""``narrow`` cuts at a STATED gap, not only at contact.

``within=`` compiles to a ``distance to <anchor>`` cut at 0 -- touching. A
great many cues mean *beside* rather than *touching*, and a thing parked
outside a house touches nothing, so containment could not express them and
the only spelling left (``find(nearest=...)``) ORDERS and drops nothing.

calcity::200#198 is the measured case. Its cue is "A red parked car outside
a house off of Poppy Boulevard", and in one run's world state the served
candidates were:

    track    d(Poppy Blvd)   d(nearest house)   red
    wt_166        3.2 m           44.0 m        0.57
    wt_76         6.4 m           19.9 m        0.38   <- ground truth
    wt_172       10.5 m           10.5 m        0.37   <- committed, 114 m off
    wt_43        27.1 m           13.0 m        0.20

Every column the agent could cut on ranked the wrong row first: redness put
wt_166 top, and proximity to the ONE house it picked put wt_172 top. What it
never had was a way to DROP wt_166 for being 44 m from any house. With the
gap cut it is one line -- cars.filter(near=find("house"), dwithin=25) -- and
the ordering is then free to say which of the survivors is on the boulevard,
where the ground truth leads at 6.4 m.

The set reading is the other half: a set in ``near=`` is its union and the
distance is the gap to its NEAREST member, so ``find("house")`` is one place
to measure from. That was already true in the executor and appeared nowhere
in the served bytes, which is why the model kept picking a single house.
"""

from __future__ import annotations

import pytest

from maps.search.parse_compile.python_surface import CodeError
from maps.search.tests.test_code_surface_arm import raising_ops_of_block


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
    """The contract that existed before the gap did. ``within=`` is
    ``dwithin=0`` on the same machinery, and a regression here would read as
    "narrow got more permissive" on every episode at once."""
    ops = _ops('r1 = find("car")\n'
               'r2 = r1.filter(within="Poppy Boulevard")\n')
    a = _cut(ops)
    assert (a["op"], a["value"]) == ("le", 0), a


def test_a_whole_class_is_one_place_to_measure_from():
    """The union reading, end to end: ``near=`` takes the find, not a row
    index of it, and still compiles to a single distance cut."""
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
     "takes no dwithin"),
    ('r1 = find("car")\nr2 = r1.filter(near="A")\n',
     "needs the gap to keep"),
    ('r1 = find("car")\nr2 = r1.filter(near="A", dwithin="soon")\n',
     "not a number"),
    ('r1 = find("car")\nr2 = r1.filter(near="A", dwithin=-5)\n',
     "cannot be negative"),
])
def test_the_refusal_names_the_exit(src, phrase):
    with pytest.raises(CodeError) as e:
        _ops(src)
    assert phrase in str(e.value), str(e.value)


# --------------------------------------------------------------------------
# The served bytes -- pinned BY NAME, because a scan cannot notice its own
# disappearance and an affordance the model is never told about is one it
# does not use. That is the whole mechanism this file exists for: the union
# rule was true in the executor and absent from the prompt, and the model
# narrowed to a single house every time.
# --------------------------------------------------------------------------


def _served() -> str:
    from maps.search.parse_compile import prompts
    return "\n".join(
        str(v) for k, v in vars(prompts).items()
        if k.isupper() and isinstance(v, str))


def test_the_legend_serves_the_gap_and_the_union_rule():
    doc = _served()
    assert "dwithin=25" in doc, "the gap form is not in the served bytes"
    assert "NEAREST member" in doc, \
        "the union rule is not served -- the model cannot use what it is " \
        "not told, which is how 200#198 was lost"
    # The vacuity floor: this scan passes trivially if the doc ever shrinks
    # to nothing, so hold it to a plausible size and to the sibling verbs
    # it sits beside.
    assert len(doc) > 4000, len(doc)
    for verb in (".filter(", "find(", "go_to(", "explore(", "look_for("):
        assert verb in doc, verb


# --------------------------------------------------------------------------
# One ordering, many cuts
# --------------------------------------------------------------------------


def test_order_refuses_two_keys_because_a_table_sorts_by_one():
    """The fused ranking is DELETED, and this is the refusal that replaces it.

    Measured on calcity::200#198: the target sits 6.4 m off the boulevard
    and 19.9 m from a house, and five ways of combining those with redness
    (min-max sum, z-score sum, rank sum, product, distances only) each rank
    a wrong car above it -- it is second on everything and first on nothing,
    which is exactly what averaging discards. The fused form was also never
    once written: 587 order_by sites across three flown arms passed two
    kwargs zero times.
    """
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\n'
             'r2 = r1.order(nearest="Poppy Boulevard", color="red")\n')
    assert "ONE key" in str(e.value), str(e.value)
    # and the refusal names the exit: cut on one, order by the other
    assert ".filter(" in str(e.value), str(e.value)


def test_a_cut_and_a_sort_chain_into_one_expression():
    """The whole of calcity::200#198, in the spelling the model reached for."""
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
    """A model reaching for narrow() gets the method, not a bare NameError.

    Nothing defines the old name -- this is the refusal text, not an alias.
    Without it the arm measures whether the model can guess a rename.
    """
    from maps.search.parse_compile.python_surface import _MOVED_TO_METHOD

    assert set(_MOVED_TO_METHOD) == {"narrow", "order_by", "gap"}
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = narrow(r1, near="A", dwithin=25)\n')
    assert "rows.filter(" in str(e.value), str(e.value)


# --------------------------------------------------------------------------
# The spellings the model reached for and the surface did not serve
# (calcity::200#198 on e898fdc9: six of fourteen turns lost to these)
# --------------------------------------------------------------------------


def test_find_cuts_at_a_stated_gap_where_the_model_puts_it():
    """``find(near=X, dwithin=M)`` -- reached for THREE times in one episode.

    find(nearest=X) orders and find(within=X) cuts at contact, so a stated gap
    on the same call is the only reading left. Same compare+filter_items
    within= already compiled, at a value instead of at 0.
    """
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


def test_a_row_and_a_set_both_measure_distance_as_a_method():
    """``row.gap(other)`` answered "carries no 'gap'" -- which reads as a
    missing COLUMN on a surface whose cut and sort are already methods. The
    measurement is spelled `distance`, the word the served column uses."""
    from maps.search.parse_compile.python_surface import Row, Rows
    assert callable(Row.distance) and callable(Rows.distance)
    assert not hasattr(Rows, "gap") and not hasattr(Row, "gap")


def test_colour_never_cuts_and_the_refusal_names_the_sort():
    """A bare TypeError cost a turn. Colour is mask-extent dependent -- the
    same car reads 0.63 through a tight mask and 0.38 through its full one
    -- so a colour threshold drops the right row on the width of its mask.
    """
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("car")\nr2 = r1.filter(color="red")\n')
    msg = str(e.value)
    assert "never CUTS" in msg and 'order(color=' in msg, msg

    # ...and a single Row cuts through the same guard, not a bare TypeError.
    from maps.search.parse_compile.python_surface import Row
    with pytest.raises(CodeError) as e:
        Row({"fid": "detected_1"}).filter(color="red")
    assert "never CUTS" in str(e.value), str(e.value)


# --------------------------------------------------------------------------
# One word, one operation: near= CUTS, nearest= ORDERS
# --------------------------------------------------------------------------


def test_near_on_find_cuts_and_refuses_to_be_an_ordering():
    """``find(near=X)`` ORDERED and dropped nothing while reading exactly
    like a filter, and the model's own variable names recorded the false
    belief: `cars_near_poppy`, `houses_near_poppy`, `houses_near_red_car`,
    `red_cars` -- bound on five straight arms of calcity::200#198 and then
    treated as though the constraint had been applied. It never had.
    """
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
    """A cut and a sort in one find: near=/dwithin= drops, nearest= orders
    what survives. Two operations, two words, neither doing the other."""
    ops = _ops('r1 = find("car", near="Poppy Boulevard", dwithin=25,\n'
               '         nearest="Poppy Boulevard")\n')
    kinds = [o["op"] for o in ops]
    assert "filter_items" in kinds and "score_by" in kinds, kinds
    assert kinds.index("filter_items") < kinds.index("score_by"), kinds


def test_look_for_names_where_it_segments_with_at():
    """``near=`` on look_for meant a third thing -- where to segment extra
    -- so it is ``at=``, the word stand() already uses for a location."""
    import inspect

    from maps.search.parse_compile.python_surface import MissionPython
    sig = inspect.signature(MissionPython._call_look_for)
    assert "at" in sig.parameters and "near" not in sig.parameters, sig


def test_a_row_serves_no_second_point_to_subtract():
    """There is ONE distance here and it is the footprint gap, so a row must
    not hand over a second point the model can difference by hand.

    ``mask_centroid`` rides ``geom_program._MODEL_HIDDEN_ROW_FIELDS`` on the
    op path and was NOT hidden on the live-surface path, so ``row_of_fid``
    served it beside ``xy``. Centroid-to-centroid is the measurement this
    surface exists to make impossible: Poppy Boulevard way 5 is 7.0 m from
    the car parked against it by footprint and 408.4 m by centroid.

    ``xy`` stays -- points are places (``find(near=[x, y])``,
    ``show_map(around=[x, y])``) and removing it would cost real capability
    to close a hazard the census can watch instead.
    """
    from maps.search.parse_compile.python_surface import Row, _ROW_HIDDEN

    assert "mask_centroid" in _ROW_HIDDEN
    r = Row({"fid": "detected_1", "xy": [10.0, 20.0],
             "mask_centroid": [11.0, 21.0], "class": "car"})
    assert "mask_centroid" not in r.keys(), r.keys()
    assert r.get("mask_centroid") is None
    with pytest.raises(AttributeError):
        r.mask_centroid
    # the point that IS a place stays reachable
    assert r.xy == [10.0, 20.0] and r.x == 10.0


def test_a_predicate_filter_is_not_a_one_way_door():
    """A lambda must not cost the served cut and sort.

    calcity::200#198, nrst r4: the model narrowed cars with a lambda, was
    refused ``order(color="red")`` because the result was a RowList, fell
    back to reading colour strings off fifty printed rows by hand, and
    committed a car 2,501 m from target. A list of Row rides the items slot
    element-wise, so the served form works on a materialised subset exactly
    as it does on a handle -- there was never a reason to refuse it.
    """
    from maps.search.parse_compile.python_surface import Row, RowList

    seen = []

    class _MP:
        def _rows_order(self, items, label, **kw):
            seen.append(("order", list(items), kw)); return "ordered"

        def _rows_filter(self, items, label, **kw):
            seen.append(("filter", list(items), kw)); return "cut"

    mp = _MP()
    rows = RowList([Row({"fid": "detected_1", "class": "car"}, mp=mp),
                    Row({"fid": "detected_2", "class": "car"}, mp=mp)])

    assert rows.order(color="red") == "ordered"
    assert seen[-1][0] == "order" and seen[-1][2] == {"color": "red"}
    assert len(seen[-1][1]) == 2          # the materialised subset, by row

    assert rows.filter(near="Poppy Boulevard", dwithin=25) == "cut"
    assert seen[-1][0] == "filter"
    assert seen[-1][2] == {"near": "Poppy Boulevard", "dwithin": 25}

    # the predicate forms still work and still chain
    assert isinstance(rows.filter(lambda r: r["fid"] == "detected_1"), RowList)
    assert len(rows.order(lambda r: r["fid"])) == 2


def test_an_empty_predicate_result_says_so_rather_than_indexing_nothing():
    from maps.search.parse_compile.python_surface import RowList
    with pytest.raises(CodeError, match="empty"):
        RowList().order(color="red")


def test_ddwithin_cuts_against_whichever_anchor_was_named():
    """``find("parking", nearest=[X], dwithin=500)`` means rank by the gap to
    X and drop past 500 -- coherent, unambiguous, and what anyone would
    write. Refusing it was a spelling objection to a correct idea, and it
    cost canary 139#137 a turn on 237952f9.
    """
    ops = _ops('r1 = find("parking", nearest="Willow Avenue", dwithin=500)\n')
    kinds = [o["op"] for o in ops]
    assert "filter_items" in kinds, kinds          # it cut
    assert "score_by" in kinds, kinds              # and it still ordered
    cut = [o for o in ops if o["op"] == "filter_items"][0]["args"]
    assert cut["value"] == 500.0, cut


def test_ddwithin_with_no_anchor_at_all_still_refuses_and_names_both():
    with pytest.raises(CodeError) as e:
        _ops('r1 = find("parking", dwithin=500)\n')
    msg = str(e.value)
    assert "near=" in msg and "nearest=" in msg, msg


def test_a_set_handed_to_go_to_is_told_to_index_not_told_it_is_a_coordinate():
    from maps.search.parse_compile.python_surface import Row
    rows = [Row({"fid": "detected_68"}), Row({"fid": "detected_69"})]
    with pytest.raises(CodeError) as e:
        _ops('go_to(x)\n')          # unbound -> different path; use the API
    from maps.search.parse_compile import python_surface as ps
    mp = ps.MissionPython(ps.MissionEnv(
        run_plan=lambda p, s: {}, rows_of_ref=lambda r: [], plan_rn=[0],
        discourse={}, minted=[], say_err=lambda a, m: {**a, "error": m},
        row_of_fid=lambda f: None))
    with pytest.raises(CodeError) as e:
        mp._call_go_to(rows)
    msg = str(e.value)
    assert "set of 2" in msg and "[0]" in msg, msg
    assert "not a coordinate" not in msg, msg


# --------------------------------------------------------------------------
# Direction, text to text
# --------------------------------------------------------------------------


def test_a_cardinal_cuts_on_a_set_in_hand_not_only_inside_find():
    """find(west_of=X) has always cut; a NARROWED set could not ask the same
    question, so a candidate set lost the relation the moment it was cut."""
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


def test_facing_is_a_word_and_it_is_already_inverted():
    """A facing question is a READ, not a second cut family.

    ``direction=`` served ``facing <place>`` in DEGREES and told the model to
    apply ``(b + 180) % 360`` itself. Zero calls across two 93-episode arms,
    and the one episode that read the column took a bearing of 0 to mean "on
    the north side" -- the un-inverted answer -- then abandoned the surface
    and hand-rolled ``p.xy[1] > 1305`` (calcity::480#478, a baseline hit the
    arm lost). A ``faces_<dir>=`` cut family replaced it and reached 3 of 45
    episodes; it was also the argument-swapped inverse of ``<dir>_of=``,
    which is the shape SpatialML dropped from its own schema because it
    confuses the reader.

    So: ONE cut family, and the facing fact rides the compare that
    ``near=``/``nearest=`` already runs, as a compass WORD that cannot be
    read backwards.

    ``direction=`` IS BACK, and it is not the kwarg this retired. The dead
    one asked for a raw DEGREES column and pushed ``(b + 180) % 360`` onto
    the model. The live one is the compass WORD as the value of a cut,
    beside ``facing=<place>``: no degrees, no arithmetic, no column
    request, and the word stays out of the kwarg NAME, which is the whole
    lesson ``faces_east=`` paid for. What has to stay gone is the degrees
    reading and the inversion instruction, so that is what is asserted.
    """
    from maps.search.analyze_scene.geom_program import run_program

    road = {"fid": "prior_1", "name": "88th Street", "xy": [0.0, 0.0],
            "mask": [[[0.0, 0.0], [0.0, 400.0]]], "tags": {"highway": "x"},
            "search_text": ""}

    def house(fid, x):
        return {"fid": fid, "name": fid, "class": "house", "xy": [x, 200.0],
                "mask": [[[x - 5, 195.0], [x + 5, 195.0],
                          [x + 5, 205.0], [x - 5, 205.0]]],
                "tags": {}, "search_text": "house"}

    out = run_program([house("detected_1", 40.0), house("detected_2", -40.0)],
                      [road],
                      {"ops": [
                          {"ref": "r1", "op": "find",
                           "args": {"query": "house"}},
                          {"ref": "r2", "op": "compare",
                           "args": {"items": "r1",
                                    "refs": [{"ref": ["prior_1"],
                                              "as": "88th Street"}]}}]})
    rows = {r["fid"]: r for r in out["results"]["r2"]}
    key = "facing 88th_street"
    # detected_1 sits EAST of the road, so its front points WEST at it.
    assert rows["detected_1"][key] == "west", rows["detected_1"]
    assert rows["detected_2"][key] == "east", rows["detected_2"]
    # and the legend says so, or the column is unreachable in practice
    from maps.search.parse_compile import prompts
    desc = prompts.CODE_TOOLS[0]["function"]["description"]
    assert '"facing <place>"' in desc, desc
    # the word never rides the kwarg NAME: that shape reached 3 of 45
    assert "faces_east" not in desc and "faces_south" not in desc
    # and the degrees reading with its inversion stays retired
    assert "edge bearing" not in desc and "180) % 360" not in desc
    # direction= is served only as the cut's compass WORD, next to facing=
    assert 'filter(facing=<place>, direction="south")' in desc, desc
    # the axis convention is stated, because the model re-derives it
    # otherwise: "x is east y is" ran 24 windows over the 1,021-episode arm.
    assert "+x runs EAST" in desc and "+y runs NORTH" in desc


@pytest.mark.parametrize("src", [
    'r1 = find("house", within=25)\n',
    'r1 = find("house")\nr2 = r1.filter(within=25)\n',
    'r1 = find("house")\nr2 = r1.filter(within=25.0)\n',
])
def test_a_distance_in_within_is_refused_and_names_dwithin(src):
    """The two names are Shapely's, and they used to be swapped here.

    ``a.within(b)`` is containment; ``dwithin(a, b, d)`` is a distance. This
    surface spelled the distance one ``within=`` until the rename, so the old
    spelling still reads as ordinary Python and would otherwise cut against a
    "place" called 25 -- a kwarg that does nothing, which is the failure mode
    this tree has been bitten by twice (``toward=`` was built, accepted, and
    compiled to nothing; a deleted compile-result key left four live readers
    that silently disarmed every reasoner).

    The rename is only safe because the two arguments have different TYPES,
    so the mistake is catchable rather than silent. That is what this pins.
    """
    with pytest.raises(CodeError) as e:
        _ops(src)
    msg = str(e.value)
    assert "dwithin=" in msg, msg
    assert "TOUCHES" in msg, msg


def test_the_two_spellings_still_do_their_own_jobs():
    """Vacuity floor on the guard above: it must refuse the swap without
    refusing either correct spelling."""
    touch = _cut(_ops('r1 = find("house")\nr2 = r1.filter(within="G Ave")\n'))
    assert touch["op"] == "le" and touch["value"] == 0, touch
    gap = _cut(_ops('r1 = find("house")\n'
                    'r2 = r1.filter(near="G Ave", dwithin=25)\n'))
    assert gap["op"] == "le" and gap["value"] == 25, gap


def test_further_is_gone_from_every_channel():
    """``order(further=<dir>)`` is DELETED, and a clean cutover means gone
    from the surface, the compiler's kwarg table and the served legend at
    once -- a spelling that survives in one of the three is the "present but
    always empty" hazard wearing a kwarg.

    Why it went. It was off-executor (a plain ``sorted()`` on ``r.xy``), so
    it served no column, dropped the ref, and read the row's recorded POINT
    rather than its footprint. Its own refusal named an impossible remedy --
    "chain the two you mean" -- where ``order`` refuses two orderings and a
    second ``order`` call overwrites the first, so a diagonal had no
    expressible form at all. It reached 1 of 29 episodes and its arm scored
    17/87 against the 25/87 of the arm it was built on.

    The corpus does not miss it: ``<dir>ernmost`` is 12 of the 1,023 cues and
    ``<dir> end of`` is 34. The commit's "61% carry an end / ordinal / corner
    word" was counting *corner* (227) and *intersection* (246), which are
    ``intersection()`` and ``ends_of()``, not an ordering.

    What says it instead is the key ``order`` already takes, over a column
    the rows already carry.
    """
    from maps.search.parse_compile import code_surface, prompts
    from maps.search.parse_compile import python_surface as ps

    assert not hasattr(ps, "_FURTHER")
    assert not hasattr(ps, "_further_key")
    assert "further" not in code_surface.CALL_KWARGS["order"]
    desc = prompts.CODE_TOOLS[0]["function"]["description"]
    # the KWARG, not the English word: the axis line still reads "a larger y
    # is further north", which is prose about the frame and not a spelling.
    assert "further=" not in desc, desc

    # ...and no ordering entry point still ACCEPTS the name. Every kwarg path
    # funnels to _rows_order, whose signature is keyword-only, so a surviving
    # `further=` would raise TypeError naming itself rather than compiling to
    # nothing -- which is how `toward=` lived here once, accepted and inert.
    import inspect
    for fn in (ps.Rows.order, ps.MissionPython._rows_order):
        assert "further" not in inspect.signature(fn).parameters, fn


def test_a_directional_ordering_is_a_key_over_a_served_column():
    """The replacement, and the reason no kwarg replaced it.

    Nothing in the ecosystem ships a directional ordering verb -- no
    ``northernmost`` in Shapely, GeoPandas, rasterio or xarray. The
    documented idiom is a COLUMN then a SORT (``gdf.bounds`` ->
    ``sort_values("maxy", ascending=False)``), which is exactly the shape
    that already works on this surface: the ``facing <ref>`` word column read
    in the model's own lambda took facing clauses from 15% to 65%.
    """
    from maps.search.parse_compile.python_surface import RowList

    class _R:
        def __init__(self, fid, xy, east_of=None):
            self.fid, self.xy = fid, xy
            self._c = {"east of poppy_boulevard": east_of}

        def __getitem__(self, k):
            return self._c[k]

        @property
        def y(self):
            return self.xy[1]

    rows = RowList([_R("a", [10.0, 100.0], 5.0), _R("b", [50.0, 20.0], 90.0),
                    _R("c", [30.0, 60.0], 40.0)])
    # global: which end of the street, off the axis convention the legend states
    assert [r.fid for r in rows.order(lambda r: -r.y)] == ["a", "c", "b"]
    # anchored: furthest east OF a named place, off the signed column
    assert [r.fid for r in
            rows.order(lambda r: -r["east of poppy_boulevard"])] == \
        ["b", "c", "a"]
