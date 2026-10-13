"""The real-Python run surface: what Python buys, and the guards that stay."""
from __future__ import annotations

import pytest

from maps.search.parse_compile.python_surface import (
    CodeError,
    Row,
    scan_block,
)
from maps.search.tests.fakes import (
    quiet_env,
    drive,
    explore_drive,
    grow_drive,
    run,
    tool_results,
)

# --------------------------------------------------------------------------
# An error costs its line, never the look_for before it
# --------------------------------------------------------------------------


def test_the_147_145_block_arms_the_watch_despite_the_map_refusal():
    reply, fr, _flights, declared = drive([
        [run('look_for("house", "pool")\n'
              'street = find("car")\n'
              'show_map(around=street)\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "around= takes ONE place" in (out.get("error") or ""), out
    # The two lines before the refusal RAN: the watch is armed (with its
    # runtime detection count riding the trailer) and the handle is bound.
    assert declared and declared[-1] == [
        {"class": "house", "role": "target"},
        {"class": "pool", "role": "target"}], declared
    assert out["watching"] == {"now": ["house (target)", "pool (target)"]}, \
        out["watching"]
    assert "street" in (out.get("handles") or []), out


def test_a_list_where_one_place_goes_is_refused_by_its_slot():
    _, fr, _, _ = drive([[run('show_map(around=["prior_7", "prior_1"])')],
                         [run('stand(at="prior_7", facing=["a", "b", "c"])')],
                         [run("show_map(around=[3, 4])")], "done"])
    two, three, point = tool_results(fr)
    assert ("around= takes ONE place and that is a list of 2 -- index the one "
            "you mean, or give one [x, y] point.") in two["error"], two
    assert "facing= takes ONE place and that is a list of 3" in three["error"], three
    assert "error" not in point, point


def test_a_block_flies_and_then_queries_what_the_pass_found():
    reply, out, flights, _asked, _rend = explore_drive([
        [run('look_for("pool")\n'
              'explore("prior_160")\n'
              'r2 = find("pool")\n'
              'r2\n')],
        "done",
    ])
    assert flights, "the pass flew"
    assert "explored" in out[0], out[0]
    assert "error" not in out[0], out[0]


# --------------------------------------------------------------------------
# Ordinary Python
# --------------------------------------------------------------------------


def test_print_len_and_fstrings_come_back_in_the_receipt():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'print(f"got {len(r1)} cars")\n'
              'n = len(r1) + 1\n'
              'n\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "got 2 cars" in (out.get("printed") or ""), out
    assert "n = 3" in (out.get("printed") or ""), out


def test_a_row_is_data_the_python_way():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'r = r1[0]\n'
              'print(r.xy, r["class"], r.get("nope", "-"))\n'
              'print(hasattr(r, "xy"), sorted(r.keys())[:2])\n'
              'both = r1 + find("car")\n'
              'print(len(both))\n'
              'print(r)\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    printed = out.get("printed") or ""
    assert "[20.0, 20.0] car -" in printed, printed
    assert "True" in printed and "4" in printed, printed
    assert "fid='car_1'" in printed, printed


def test_a_conditional_runs_and_a_loop_may_look():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'if len(r1) > 1:\n'
              '    fids = [r.fid for r in r1]\n'
              'print(fids)\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "car_1" in (out.get("printed") or ""), out


def test_a_python_error_keeps_what_already_ran():
    """NameError at line 2; the find at line 1 stands, envelope included."""
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nnope + 1\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "line 2: NameError" in (out.get("error") or ""), out
    assert "r1" in (out.get("handles") or []), out
    assert out.get("result"), out


# --------------------------------------------------------------------------
# The pre-scan: what is refused, with the reasons served
# --------------------------------------------------------------------------


def test_the_prescan_refusals():
    with pytest.raises(CodeError, match="cannot import os"):
        scan_block("import os\n")
    with pytest.raises(CodeError, match="cannot import random"):
        scan_block("from random import choice\n")
    # statistics is OUT: NormalDist.samples draws ambient random.gauss,
    # and two identical blocks must never produce two different results.
    with pytest.raises(CodeError, match="cannot import statistics"):
        scan_block("import statistics\n")
    with pytest.raises(CodeError, match="math is a loaded module"):
        scan_block("math = 5\n")
    with pytest.raises(CodeError, match="no underscore names"):
        scan_block("_x = 1\n")
    with pytest.raises(CodeError, match="there is no eval"):
        scan_block("eval('1')\n")
    with pytest.raises(CodeError, match="cannot go inside a loop"):
        scan_block("for c in r1:\n    go_to(c)\n")
    with pytest.raises(CodeError, match="cannot go inside a loop"):
        scan_block("while x:\n    explore(x)\n")
    for comp in ("[go_to(r) for r in r1]", "{explore(r) for r in r1}",
                 "{r: go_to(r) for r in r1}", "list(go_to(r) for r in r1)"):
        with pytest.raises(CodeError, match="cannot go inside a loop"):
            scan_block(f"{comp}\n")
    with pytest.raises(CodeError, match="syntax error"):
        scan_block("r1 = find(\n")
    # ...and ordinary Python is not refused.
    scan_block("xs = sorted([3, 1, 2])\n"
               "if xs:\n"
               "    n = len(xs)\n"
               "for x in xs:\n"
               "    print(x)\n")


def test_a_runaway_loop_dies_on_the_line_ceiling():
    reply, fr, _f, _d = drive([
        [run("n = 0\nwhile True:\n    n = n + 1\n")],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "unbounded loop" in (out.get("error") or ""), out


def test_printing_a_whole_city_is_capped_like_a_notebook():
    reply, fr, _f, _d = drive([
        [run('for i in range(500):\n    print(f"row {i}")\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    printed = out.get("printed") or ""
    assert "row 0" in printed and "row 499" in printed, printed[:200]
    assert "printed lines suppressed" in printed, printed[:200]
    assert printed.count("\n") < 70, printed.count("\n")



def test_print_takes_prints_own_keywords():
    from maps.search.parse_compile.python_surface import MissionPython

    out = MissionPython(quiet_env()).run_block(
        'print("a", end="")\nprint("b", flush=True)\nprint("c", "d", sep="-")\n')
    assert "error" not in out, out
    assert out["printed"] == "ab\nc-d", out["printed"]


def test_a_move_returns_the_rows_it_arrived_over():
    results, _g, added = grow_drive([
        [run('look_for("car")\n')],
        [run('arrivals = go_to("prior_7")\n'
              'print(len(arrivals) if arrivals else "none")\n')],
        "done",
    ])
    assert "error" not in results[1], results[1]
    assert (results[1].get("printed") or "").strip() != "none", results[1]


def test_binding_over_a_mission_call_is_refused():
    import pytest

    with pytest.raises(CodeError, match="intersection is a mission call"):
        scan_block('intersection = intersection("A", "B")\n')


def test_math_is_loaded_and_import_math_works():
    reply, fr, _f, _d = drive([
        [run('import math\n'
              'print(math.dist([0, 0], [3, 4]))\n'
              'print(math.sqrt(16))\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    printed = out.get("printed") or ""
    assert "5.0" in printed and "4.0" in printed, printed


def test_a_syntax_error_echoes_the_offending_line():
    with pytest.raises(CodeError) as e:
        scan_block("d = (a - b)**2 + **(c - f)2\n")
    msg = str(e.value)
    assert "syntax error" in msg
    assert "**(c - f)2" in msg and "^" in msg, msg


def test_a_compile_stage_syntax_error_reaches_the_model():
    with pytest.raises(CodeError) as e:
        scan_block('r = find("car", near="a", near="b")\n')
    assert "keyword argument repeated" in str(e.value)
    reply, fr, _f, _d = drive([
        [run('r = find("car", near="a", near="b")\n')],
        "done",
    ])
    err = tool_results(fr)[0].get("error") or ""
    assert "keyword argument repeated" in err, err


def test_an_unknown_call_gets_a_did_you_mean():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nnarow(r1, within="the depot")\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    err = out.get("error") or ""
    assert "did you mean" in err and "narrow" in err, out
    assert "r1" in (out.get("handles") or []), out


def test_try_except_is_ordinary_python():
    reply, fr, _f, _d = drive([
        [run('try:\n'
              '    x = [1][9]\n'
              'except IndexError:\n'
              '    x = "caught"\n'
              'try:\n'
              '    y = 1 / 0\n'
              'except Exception:\n'
              '    y = "caught too"\n'
              'print(x, y)\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "caught caught too" in (out.get("printed") or ""), out


def test_an_unbound_name_says_why_it_is_gone():
    reply, fr, _f, _d = drive([
        [run('r3 = find("car")[99]\n')],
        [run('go_to(r3)\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "did not bind" in (out[0].get("error") or ""), out[0]
    err = out[1].get("error") or ""
    assert "r3 is unbound" in err and "re-derive" in err, out[1]


def test_kwarg_typeerrors_name_the_call_not_the_internals():
    reply, fr, _f, _d = drive([
        [run('find("car", fid="car_1")\n')],
        "done",
    ])
    err = (tool_results(fr)[0].get("error") or "")
    assert "find()" in err and "'fid'" in err, err
    assert "find keywords:" in err and "near=" in err, err
    assert "_call_" not in err, err


def test_a_bare_fid_reads_as_its_row_and_still_flies():
    reply, fr, flights, _d = drive([
        [run('print(prior_7.xy)\ngo_to(prior_7)\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "[" in (out.get("printed") or ""), out
    assert flights and flights[0]["target_fid"] == "prior_7", flights


def test_row_get_resolves_aliases_instead_of_defaulting():
    from maps.search.parse_compile.python_surface import Row

    r = Row({"fid": "f1", "xy": [1.5, 2.5], "distance to p0": 5.0})
    assert r.get("distance_to_p0", 9999) == 5.0
    assert r["distance_to_p0"] == 5.0
    assert r.x == 1.5 and r.y == 2.5
    assert r["x"] == 1.5 and r.get("y") == 2.5   # all three access forms
    assert r.get("truly_absent", "-") == "-"


def test_a_ref_column_is_spelled_by_its_places_real_name():
    """The model spells a column with the name the CUE gave it; the producer spells it
    with the name it sanitized."""
    from maps.search.analyze_scene.geom_program import _sanitize_token
    from maps.search.parse_compile.python_surface import Row

    lbl = _sanitize_token("Poppy Boulevard")
    assert lbl == "poppy_boulevard", lbl        # the producer's own rule
    r = Row({"fid": "prior_1", "xy": [10.0, 20.0],
             f"facing {lbl}": "south",
             f"distance to {lbl}": 7.0,
             f"north of {lbl}": 42.5})

    # the cue's spelling, the served spelling and the shouty one all land
    for k, want in ((f"facing {lbl}", "south"),
                    ("facing Poppy Boulevard", "south"),
                    ("FACING POPPY BOULEVARD", "south"),
                    ("distance to Poppy Boulevard", 7.0),
                    ("north of Poppy Boulevard", 42.5)):
        assert r[k] == want, k
        assert r.get(k) == want, k              # .get may not differ

    # ...and a genuinely absent column is still absent, so the normalisation
    # widened the SPELLING and not the surface.
    assert r.get("facing Some Other Road") is None
    assert r.get("facing Poppy Boulevard", 9999) == "south"


def test_an_ambiguous_column_spelling_resolves_to_nothing():
    """Two served columns that normalise alike is a silent bind waiting to happen, so
    this refuses rather than picking one."""
    from maps.search.parse_compile.python_surface import Row

    r = Row({"fid": "prior_1", "facing a_b": "north", "facing a b": "south"})
    assert r.get("facing A B") is None
    assert r["facing a b"] == "south"           # an EXACT spelling still wins


def test_builtins_survive_the_did_you_mean():
    reply, fr, _f, _d = drive([
        [run('print(list((1, 2)), dict(a=1)["a"], sorted({3, 1}), '
              'repr(set()))\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "[1, 2] 1 [1, 3] set()" in (out.get("printed") or ""), out


def test_bare_fid_rows_compare_by_value():
    """Two references to the same bare fid mint two Row views; model dedup (`if cand not
    in seen`) must compare the PLACE, not the object identity -- and a Row equals its
    own fid string."""
    reply, fr, _f, _d = drive([
        [run('print(prior_7 == prior_7)\n'
              'print(prior_7 in [prior_7])\n'
              'print(prior_7 == "prior_7")\n'
              'seen = {prior_7, prior_7}\n'
              'print(len(seen))\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    printed = out.get("printed") or ""
    assert printed.splitlines() == ["True", "True", "True", "1"], printed


def test_the_caret_lands_right_under_tabs():
    with pytest.raises(CodeError) as e:
        scan_block("if True:\n\td = (a - b)**2 + **(c - f)2\n")
    msg = str(e.value)
    body, caret_line = msg.splitlines()[-2:]
    assert "\t" not in body, repr(body)     # rendered, not raw tab
    assert caret_line.rstrip().endswith("^")
    assert body[len(caret_line.rstrip()) - 1] == "*", (body, caret_line)


def test_a_failed_unpack_unbinds_every_target():
    reply, fr, _f, _d = drive([
        [run('a, b = 1, 2\n')],
        [run('a, b = [1, 2, 3][99], 5\n')],
        [run('print(a)\n')],
        "done",
    ])
    out = tool_results(fr)
    assert "IndexError" in (out[1].get("error") or ""), out[1]
    assert "a is unbound" in (out[2].get("error") or ""), out[2]


def test_a_crop_token_falls_back_to_the_rows_own_xy():
    from maps.search.parse_compile.views import _crop_token_of

    live_row = {"fid": "detected_9", "xy": [10.0, 20.0]}
    end_row = {"fid": "end_0", "xy": [3.0, 4.5]}
    assert _crop_token_of(live_row, True) == "detected_9"
    assert _crop_token_of(end_row, False) == "[3, 4.5]"
    assert _crop_token_of({"fid": "end_1"}, False) == "end_1"
    # An EVICTED detection is not a derived row: it keeps its fid and the
    # skip path reports it gone, never a crop of its stale xy.
    assert _crop_token_of({"fid": "detected_3", "xy": [9.0, 9.0]},
                          False) == "detected_3"


def test_a_far_anchored_find_says_nothing_is_near():
    far = [
        {"fid": "prior_9", "xy": [5000.0, 5000.0], "class": "pool",
         "search_text": "pool"},
        {"fid": "road_1", "name": "Mill Lane", "xy": [0.0, 0.0],
         "class": "road", "centerline": [[0.0, -50.0], [0.0, 50.0]],
         "search_text": "mill lane"},
    ]
    reply, fr, _f, _d = drive([
        [run('pools = find("pool", nearest="Mill Lane")\n')],
        "done",
    ], cands=far)
    out = tool_results(fr)[0]
    assert "lies near" in (out.get("note") or ""), out
    assert "only when a flight covers it" in out["note"], out["note"]


def test_order_by_fuses_two_anchors_like_find_near():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'r2 = r1.order(nearest=["the depot", "the gate"])\n'
              'r2\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    rows = out["result"]["results"]["r2"]["rows"]
    assert "distance to the_depot" in rows and \
        "distance to the_gate" in rows, rows


def test_a_slice_rides_slots_and_slices_again():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\n'
              'head = r1[0:2]\n'
              'sub = head[0:1]\n'
              'print(len(sub), sub[0].fid)\n'
              'near_head = find("gate", nearest=r1[0:2])\n'
              'near_head\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "1 car_1" in (out.get("printed") or ""), out


def test_a_chained_index_refusal_names_what_it_indexed():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nr1[0:2][999]["fid"]\n')],
        "done",
    ])
    err = tool_results(fr)[0].get("error") or ""
    assert "r1[0:2][999] did not bind: r1[0:2] served 2 row(s)" in err, err


def test_an_explore_names_the_uncovered_sibling_segments():
    """A named street is many fragments; a pass covers one."""
    cands = [
        {"fid": "prior_1", "name": "85th Street", "xy": [0.0, 0.0],
         "class": "road", "centerline": [[0.0, -50.0], [0.0, 50.0]],
         "search_text": "85th street"},
        {"fid": "prior_2", "name": "85th Street", "xy": [0.0, 200.0],
         "class": "road", "centerline": [[0.0, 150.0], [0.0, 250.0]],
         "search_text": "85th street"},
        {"fid": "prior_3", "name": "85th Street", "xy": [0.0, 400.0],
         "class": "road", "centerline": [[0.0, 350.0], [0.0, 450.0]],
         "search_text": "85th street"},
    ]
    _reply, out, _flights, _asked, _rend = explore_drive(
        [[run('look_for("pool")\nexplore(prior_1)\n')], "done"],
        cands=cands, stops=[[0.0, -50.0], [0.0, 50.0]])
    note = (out[0].get("explored") or {}).get("note") or out[0].get("note") or ""
    assert "2 more segment(s) no pass has covered" in note, out[0]
    assert "prior_2" in note and "prior_3" in note, note


def test_a_near_miss_fid_names_the_correction():
    reply, fr, _f, _d = drive([
        [run('go_to(prio_7)\n')],
        "done",
    ])
    err = (tool_results(fr)[0].get("error") or "")
    assert "no prio_7 on this surface" in err and "prior_7" in err, err


def test_a_get_near_miss_notes_the_real_column():
    """.get keeps returning the default on a miss, and the receipt names the near column."""
    reply, fr, _f, _d = drive([
        [run('r1 = find("car", nearest="the depot")\n'
              'v = r1[0].get("distance_to_depot", 9999)\n'
              'print(v)\n')],
        "done",
    ])
    out = tool_results(fr)[0]
    assert "error" not in out, out
    assert "9999" in (out.get("printed") or ""), out
    note = out.get("note") or ""
    assert "distance_to_depot" in note and "distance to the_depot" in note, out


def test_rows_repr_names_its_rows():
    reply, fr, _f, _d = drive([
        [run('r1 = find("car")\nprint(r1)\n')],
        "done",
    ])
    printed = (tool_results(fr)[0].get("printed") or "")
    assert "car_1 [" in printed and "row(s):" in printed, printed


def test_a_bound_handle_beats_a_coincidental_fid_suggestion():
    """corner_5 typo'd from a bound corner_4 must suggest the model's own handle, never
    a coincidental prior_5 -- the fid correction fires only when the PREFIX
    near-misses a fid family (detect_59 -> detected_59)."""
    reply, fr, _f, _d = drive([
        [run('corner_4 = [10, 20]\nprint(corner_5)\n')],
        "done",
    ])
    err = (tool_results(fr)[0].get("error") or "")
    assert "corner_4" in err, err
    assert "prior_5" not in err, err


def test_a_tied_view_is_not_subscriptable_the_way_dict_keys_is_not():
    """The ways of ONE name tie, so no row is its first."""
    from maps.search.parse_compile.python_surface import MissionPython, Rows

    rows = [{"fid": f"prior_{i}", "name": "Neuralia Road"} for i in range(12)]
    env = quiet_env(rows_of_ref=lambda ref: rows, tied_of_ref=lambda ref: True)
    view = Rows("r1", MissionPython(env))

    assert len(view) == 12
    assert len(list(view)) == 12, "a tied view still iterates"

    for bad in (0, -1, 5):
        with pytest.raises(CodeError, match="not subscriptable"):
            view[bad]
    with pytest.raises(CodeError, match="not subscriptable"):
        view[20:40]


def test_an_ordered_result_stays_subscriptable():
    """The refusal is about TIES, not about find results."""
    from maps.search.parse_compile.python_surface import MissionPython, Rows

    rows = [{"fid": f"detected_{i}", "class": "car"} for i in range(6)]
    env = quiet_env(rows_of_ref=lambda ref: rows, tied_of_ref=lambda ref: False)
    view = Rows("r2", MissionPython(env))
    assert view[0].get("fid") == "detected_0"
    assert len(view[0:3]) == 3


def test_a_tied_views_repr_shows_its_shape_and_leaks_no_point():
    """print(road) must not hand back what the head withholds."""
    from maps.search.parse_compile.python_surface import MissionPython, Rows

    rows = [{"fid": f"prior_{5775 + i}", "name": "Neuralia Road",
             "xy": [2242.27 + i, 3048.07 + i]} for i in range(64)]

    def _env(tied):
        return quiet_env(rows_of_ref=lambda ref: rows,
                         tied_of_ref=lambda ref: tied)

    tied = repr(Rows("r1", MissionPython(_env(True))))
    assert "Neuralia Road" in tied and "64" in tied, tied
    assert "prior_5775" not in tied, tied
    assert "2242" not in tied, tied

    # VACUITY FLOOR: an UNTIED result still names its rows and points, or
    # this would pass on a repr that had stopped saying anything at all.
    untied = repr(Rows("r1", MissionPython(_env(False))))
    assert "prior_5775" in untied, untied
    assert "2242" in untied, untied


def test_a_relation_key_names_the_relate_instead_of_a_column_list():
    """A relation column is not MISSING, it is UNASKED-FOR."""
    r = Row({"fid": "prior_3213", "xy": [1.0, 2.0], "class": "house",
             "distance to poppy_boulevard": 12.0}, None)

    # a relate column names the relate, and names the PLACE it was asked
    # about rather than the ones it already carries
    for k in ("distance to redwood", "east of quezon", "facing nipa",
              "bearing from great_circle"):
        with pytest.raises(KeyError) as e:
            r[k]
        msg = str(e.value)
        assert "never related to" in msg, msg
        assert "order(nearest=" in msg, msg
        assert k[k.index(" ") + 1:].split(" ")[-1] in msg, msg
    # no relate writes an "along" column, so none is offered for one
    with pytest.raises(KeyError) as e:
        r["along airway"]
    assert "never related to" not in str(e.value), str(e.value)
    # attribute access answers the same way; one resolver, one message
    with pytest.raises(AttributeError, match="never related to"):
        r.__getattr__("distance to redwood")

    # a VIEWPOINT side is not a relate, and must not claim to be
    with pytest.raises(KeyError) as e:
        r["left of v1"]
    assert "VIEWPOINT" in str(e.value) and "stand(" in str(e.value)
    assert "never related to" not in str(e.value)

    # a colour is a palette word, and an ordinary miss is still a column
    # list -- the message is only special where a remedy exists
    for k in ("color_red", "tags"):
        with pytest.raises(KeyError) as e:
            r[k]
        assert "its columns are" in str(e.value), k

    # and a column the rows DO carry still just reads
    assert r["distance to poppy_boulevard"] == 12.0
    # .get keeps its contract: a miss returns the default, silently
    assert r.get("distance to redwood", 99) == 99


def test_a_non_callable_predicate_teaches_the_method_the_surface_serves():
    for call in ("r1.filter(5)", "r1.order(5)"):
        _reply, fr, _f, _d = drive([[run(f'r1 = find("car")\n{call}\n')], "done"])
        err = tool_results(fr)[0].get("error") or ""
        assert "r.distance(x)" in err and "gap(" not in err, err


def test_every_filter_refuses_an_unknown_kwarg_in_one_text():
    errs = []
    for rows in ("r1", "r1[0]", "r1.filter(lambda r: True)"):
        _reply, fr, _f, _d = drive([[run(f'r1 = find("car")\n{rows}.filter(toward=1)\n')],
                                     "done"])
        errs.append(tool_results(fr)[0].get("error") or "")
    assert errs[0] == errs[1] == errs[2] and "toward=" in errs[0], errs


def test_a_list_plus_rows_is_a_rowlist_either_way_round():
    _reply, fr, _f, _d = drive([[run('r1 = find("car")\n'
                                       'print(hasattr(r1 + [], "filter"),'
                                       ' hasattr([] + r1, "filter"))\n')], "done"])
    out = tool_results(fr)[0]
    assert (out.get("printed") or "").strip() == "True True", out


def test_a_slice_of_one_row_chains_like_every_other_slice():
    r = Row({"fid": "f1", "xy": [1.0, 2.0]})
    assert r[0:1].order(lambda x: x.x) == [r]
