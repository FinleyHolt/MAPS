"""The real-Python run surface: what Python buys, and the guards that stay.

The seam tests live in ``test_code_surface_arm`` (a `run` block reaches the
plan body `say` reaches). These pin what is NEW here: eager source-order
execution, Python's own failure semantics, the ordinary-Python affordances
(print, len, if, loops), and the pre-scan refusals -- including the block
shape that lost calcity::147#145, verbatim.
"""
from __future__ import annotations

import pytest

from maps.search.parse_compile.python_surface import (
    _AUTOATTACH_PREFIXES, CodeError, Row, looks_like_code, scan_block)
from maps.search.tests.test_code_surface_arm import _explore_drive, _run
from maps.search.tests.test_mission_loop import _drive, _tool_results


# --------------------------------------------------------------------------
# The 147#145 block: an error costs its line, never the look_for before it
# --------------------------------------------------------------------------


def test_the_147_145_block_arms_the_watch_despite_the_map_refusal():
    """The block that lost calcity::147#145, verbatim.

    On the fold-order surface the `around=<set>` refusal fired at parse
    time and took the whole block with it -- including the valid
    `look_for` on line 1 -- so the first explore flew with an empty
    watchlist and detected nothing (3,728 m miss). Here the refusal costs
    ITS line: the watch is armed, the street is bound, and the error
    still says exactly what it always said.
    """
    reply, fr, _flights, declared = _drive([
        [_run('look_for("house", "pool")\n'
              'street = find("car")\n'
              'show_map(around=street)\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "around= takes ONE place" in (out.get("error") or ""), out
    # The two lines before the refusal RAN: the watch is armed (with its
    # runtime detection count riding the trailer) and the handle is bound.
    assert declared and declared[-1] == [
        {"class": "house", "role": "target"},
        {"class": "pool", "role": "target"}], declared
    assert out["watching"] == {"now": ["house (target)", "pool (target)"]}, \
        out["watching"]
    assert "street" in (out.get("handles") or []), out


def test_a_block_flies_and_then_queries_what_the_pass_found():
    """Eager source order, end to end: the find AFTER the explore reads
    the rows that pass just detected -- the shape the fold-order surface
    refused and the legend now teaches."""
    reply, out, flights, _asked, _rend = _explore_drive([
        [_run('look_for("pool")\n'
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
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'print(f"got {len(r1)} cars")\n'
              'n = len(r1) + 1\n'
              'n\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "got 2 cars" in (out.get("printed") or ""), out
    assert "n = 3" in (out.get("printed") or ""), out


def test_a_row_is_data_the_python_way():
    """r.xy, r["class"], r.get, print(r): the first pysurface fleet drew
    150+ AttributeErrors over 69 episodes from a fid-only Row -- a row IS
    its served columns to a Python reader."""
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'r = r1[0]\n'
              'print(r.xy, r["class"], r.get("nope", "-"))\n'
              'print(hasattr(r, "xy"), sorted(r.keys())[:2])\n'
              'both = r1 + find("car")\n'
              'print(len(both))\n'
              'print(r)\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    printed = out.get("printed") or ""
    assert "[20.0, 20.0] car -" in printed, printed
    assert "True" in printed and "4" in printed, printed
    assert "fid='car_1'" in printed, printed


def test_find_refuses_two_queries_with_the_remedy():
    reply, fr, _f, _d = _drive([
        [_run('find("89th Street", "Fuchsia Avenue")\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "find takes ONE query" in (out.get("error") or ""), out
    assert "intersection(" in out["error"], out


def test_a_conditional_runs_and_a_loop_may_look():
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'if len(r1) > 1:\n'
              '    fids = [r.fid for r in r1]\n'
              'print(fids)\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "car_1" in (out.get("printed") or ""), out


def test_a_python_error_keeps_what_already_ran():
    """NameError at line 2; the find at line 1 stands, envelope included."""
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nnope + 1\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
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
    with pytest.raises(CodeError, match="syntax error"):
        scan_block("r1 = find(\n")
    # ...and ordinary Python is not refused.
    scan_block("xs = sorted([3, 1, 2])\n"
               "if xs:\n"
               "    n = len(xs)\n"
               "for x in xs:\n"
               "    print(x)\n")


def test_a_runaway_loop_dies_on_the_line_ceiling():
    reply, fr, _f, _d = _drive([
        [_run("n = 0\nwhile True:\n    n = n + 1\n")],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "unbounded loop" in (out.get("error") or ""), out


def test_the_code_reply_detector_reads_python():
    assert looks_like_code("go_to(prior_4302)")
    assert looks_like_code('r1 = find("car")\ngo_to(r1[0])')
    assert not looks_like_code("The pool house is prior_7.")
    assert not looks_like_code("done")


def test_a_bare_fid_name_is_that_fid():
    """go_to(detected_192) unquoted: 541 NameErrors over 149 episodes on
    the first fleet. A served fid reads as a binding."""
    reply, fr, flights, _d = _drive([
        [_run("go_to(prior_7)\n")],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert flights and flights[0]["target_fid"] == "prior_7", flights


def test_printing_a_whole_city_is_capped_like_a_notebook():
    """One episode printed 3,414 rows, blew the context, and died with
    zero flights. The printed channel serves head + tail + a suppression
    note, notebook-style."""
    reply, fr, _f, _d = _drive([
        [_run('for i in range(500):\n    print(f"row {i}")\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    printed = out.get("printed") or ""
    assert "row 0" in printed and "row 499" in printed, printed[:200]
    assert "printed lines suppressed" in printed, printed[:200]
    assert printed.count("\n") < 70, printed.count("\n")


def test_a_move_returns_the_rows_it_arrived_over():
    from maps.search.tests.test_code_surface_arm import _fly_drive
    results, _g, added = _fly_drive([
        [_run('look_for("car")\n')],
        [_run('arrivals = go_to("prior_7")\n'
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
    """The top kicked-back family on both fleets was hand-rolled
    Euclidean distance ("without math module", the model's own comment)
    with a mangled exponent. The pure deterministic stdlib is loaded, and
    importing it is a no-op rather than a refusal."""
    reply, fr, _f, _d = _drive([
        [_run('import math\n'
              'print(math.dist([0, 0], [3, 4]))\n'
              'print(math.sqrt(16))\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    printed = out.get("printed") or ""
    assert "5.0" in printed and "4.0" in printed, printed


def test_a_syntax_error_echoes_the_offending_line():
    with pytest.raises(CodeError) as e:
        scan_block("d = (a - b)**2 + **(c - f)2\n")
    msg = str(e.value)
    assert "syntax error" in msg
    assert "**(c - f)2" in msg and "^" in msg, msg


def test_an_unknown_call_gets_a_did_you_mean():
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nnarow(r1, within="the depot")\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    err = out.get("error") or ""
    assert "did you mean" in err and "narrow" in err, out
    assert "r1" in (out.get("handles") or []), out


def test_try_except_is_ordinary_python():
    """`except Exception:` raised NameError on the first pysurface3
    fleet (113#112 b32): the standard exception names ride the
    builtins like the rest of the language."""
    reply, fr, _f, _d = _drive([
        [_run('try:\n'
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
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "caught caught too" in (out.get("printed") or ""), out


def test_show_is_the_bare_expression_renderer():
    """show(r1) was reached 25x on arm 1 and only earned an error; the
    model's own display verb now renders exactly what `r1` on its own
    line does."""
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nshow(r1)\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "r1" in ((out.get("result") or {}).get("results") or {}), out


def test_an_unbound_name_says_why_it_is_gone():
    reply, fr, _f, _d = _drive([
        [_run('r3 = find("car")[99]\n')],
        [_run('go_to(r3)\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "did not bind" in (out[0].get("error") or ""), out[0]
    err = out[1].get("error") or ""
    assert "r3 is unbound" in err and "re-derive" in err, out[1]


def test_kwarg_typeerrors_name_the_call_not_the_internals():
    reply, fr, _f, _d = _drive([
        [_run('find("car", fid="car_1")\n')],
        "done",
    ])
    err = (_tool_results(fr)[0].get("error") or "")
    assert "find()" in err and "'fid'" in err, err
    assert "find keywords:" in err and "near=" in err, err
    assert "_call_" not in err, err


def test_a_bare_fid_reads_as_its_row_and_still_flies():
    """detected_192.xy was the 'str' object AttributeError family: a bare
    fid name now resolves to a Row view of the live row, and a slot still
    takes it (the flight target is the fid verbatim)."""
    reply, fr, flights, _d = _drive([
        [_run('print(prior_7.xy)\ngo_to(prior_7)\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "[" in (out.get("printed") or ""), out
    assert flights and flights[0]["target_fid"] == "prior_7", flights


def test_row_get_resolves_aliases_instead_of_defaulting():
    """The silent hole: r.get('distance_to_p0', 9999) fed its sentinel
    into a real filter whenever the spelling missed the spaced column."""
    from maps.search.parse_compile.python_surface import Row

    r = Row({"fid": "f1", "xy": [1.5, 2.5], "distance to p0": 5.0})
    assert r.get("distance_to_p0", 9999) == 5.0
    assert r["distance_to_p0"] == 5.0
    assert r.x == 1.5 and r.y == 2.5
    assert r["x"] == 1.5 and r.get("y") == 2.5   # all three access forms
    assert r.get("truly_absent", "-") == "-"


def test_a_ref_column_is_spelled_by_its_places_real_name():
    """The model spells a column with the name the CUE gave it; the producer
    spells it with the name it sanitized. Those have to be one lookup.

    Every ref label rides ``geom_program._sanitize_token``, so relating rows
    to Poppy Boulevard writes ``"facing poppy_boulevard"`` -- while the
    legend's own worked example said ``r["facing Poppy Boulevard"]``, which
    raised KeyError. That is the largest directional family in the corpus
    (318 of 1,023 cues carry a facing clause) reading a served example that
    could not run.

    Fixed once, in the one resolver, so all three ref column families move
    together rather than one alias per prefix.
    """
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
    """Two served columns that normalise alike is a silent bind waiting to
    happen, so this refuses rather than picking one."""
    from maps.search.parse_compile.python_surface import Row

    r = Row({"fid": "prior_1", "facing a_b": "north", "facing a b": "south"})
    assert r.get("facing A B") is None
    assert r["facing a b"] == "south"           # an EXACT spelling still wins


def test_builtins_survive_the_did_you_mean():
    """LOAD_GLOBAL falls through to __builtins__ only on a plain
    KeyError: review found `list` did-you-meant `bisect` and died."""
    reply, fr, _f, _d = _drive([
        [_run('print(list((1, 2)), dict(a=1)["a"], sorted({3, 1}), '
              'repr(set()))\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "[1, 2] 1 [1, 3] set()" in (out.get("printed") or ""), out


def test_bare_fid_rows_compare_by_value():
    """Two references to the same bare fid mint two Row views; model
    dedup (`if cand not in seen`) must compare the PLACE, not the
    object identity -- and a Row equals its own fid string."""
    reply, fr, _f, _d = _drive([
        [_run('print(prior_7 == prior_7)\n'
              'print(prior_7 in [prior_7])\n'
              'print(prior_7 == "prior_7")\n'
              'seen = {prior_7, prior_7}\n'
              'print(len(seen))\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    printed = out.get("printed") or ""
    assert printed.splitlines() == ["True", "True", "True", "1"], printed


def test_the_caret_lands_right_under_tabs():
    """The pad is measured on the tab-EXPANDED prefix: with a raw-column
    pad the caret drifted left of the offending character on every
    tab-indented block."""
    with pytest.raises(CodeError) as e:
        scan_block("if True:\n\td = (a - b)**2 + **(c - f)2\n")
    msg = str(e.value)
    body, caret_line = msg.splitlines()[-2:]
    assert "\t" not in body, repr(body)     # rendered, not raw tab
    assert caret_line.rstrip().endswith("^")
    assert body[len(caret_line.rstrip()) - 1] == "*", (body, caret_line)


def test_a_failed_unpack_unbinds_every_target():
    reply, fr, _f, _d = _drive([
        [_run('a, b = 1, 2\n')],
        [_run('a, b = [1, 2, 3][99], 5\n')],
        [_run('print(a)\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "IndexError" in (out[1].get("error") or ""), out[1]
    assert "a is unbound" in (out[2].get("error") or ""), out[2]


def test_a_crop_token_falls_back_to_the_rows_own_xy():
    from maps.search.parse_compile.reasoner_loop import _crop_token_of

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
    """The 147#145 fact, served at the moment of the query: every match
    kilometres from the named anchor is said in one sentence."""
    far = [
        {"fid": "prior_9", "xy": [5000.0, 5000.0], "class": "pool",
         "search_text": "pool"},
        {"fid": "road_1", "name": "Mill Lane", "xy": [0.0, 0.0],
         "class": "road", "centerline": [[0.0, -50.0], [0.0, 50.0]],
         "search_text": "mill lane"},
    ]
    reply, fr, _f, _d = _drive([
        [_run('pools = find("pool", nearest="Mill Lane")\n')],
        "done",
    ], cands=far)
    out = _tool_results(fr)[0]
    assert "lies near" in (out.get("note") or ""), out
    assert "only when a flight covers it" in out["note"], out["note"]


def test_order_by_fuses_two_anchors_like_find_near():
    """houses.order(nearest=[A, B]) was refused on the first fleets;
    the executor fuses the two normalized distances exactly as
    find(nearest=[A, B]) does, so the other spelling now compiles too."""
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'r2 = r1.order(nearest=["the depot", "the gate"])\n'
              'r2\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    rows = out["result"]["results"]["r2"]["rows"]
    assert "distance to the_depot" in rows and \
        "distance to the_gate" in rows, rows


def test_a_slice_rides_slots_and_slices_again():
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'head = r1[0:2]\n'
              'sub = head[0:1]\n'
              'print(len(sub), sub[0].fid)\n'
              'near_head = find("gate", nearest=r1[0:2])\n'
              'near_head\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "1 car_1" in (out.get("printed") or ""), out


def test_an_explore_names_the_uncovered_sibling_segments():
    """A named street is many fragments; a pass covers one. The ack says
    how much of the NAME remains un-flown (147#145's method-correct retry
    explored 2 of 10 segments and missed by 573 m)."""
    import json as _json
    from types import SimpleNamespace
    from maps.search.parse_compile.reasoner_loop import (
        MissionSpec, advance_reasoner_one_round)
    from maps.search.tests.test_mission_loop import FakeReasoner

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
    flights: list = []

    def _fly(item):
        flights.append(item)
        return {"stop_id": f"stop_{len(flights):03d}",
                "target": item["target_fid"], "pose_xy": [1.0, 2.0],
                "new_detections": 0, "cue": "the cue"}

    fr = FakeReasoner([
        [("run", {"code": 'look_for("pool")\nexplore(prior_1)\n'})],
        "done",
    ])
    advance_reasoner_one_round(
        reasoner=fr, user_message="Find it.", ctx=SimpleNamespace(
            graph=None, cell_size_m=1.0, tile_origin_xy=(0.0, 0.0),
            grid_shape=(10, 10), robot_pose=None),
        grounded_fids=set(), rid_tag="m0",
        compute_namespace={"cands": cands, "anchors": [], "cell_m": 1.0},
        diag={}, obs_span_m=200.0,
        mission_spec=MissionSpec(
            fly=_fly, declare_concepts=lambda i: {"declared": []},
            coverage_stops=lambda fid, spacing_m: [[0.0, -50.0], [0.0, 50.0]],
            flight_budget=8, turn_cap=16),
    )
    out = [_json.loads(m["content"]) for m in fr.messages
           if m.get("role") == "tool"]
    note = (out[0].get("explored") or {}).get("note") or out[0].get("note") or ""
    assert "2 more segment(s) no pass has covered" in note, out[0]
    assert "prior_2" in note and "prior_3" in note, note


def test_a_near_miss_fid_names_the_correction():
    """go_to(detect_59) got a bare NameError while detected_59 sat in the
    previous block's own print (284#282 b8)."""
    reply, fr, _f, _d = _drive([
        [_run('go_to(prio_7)\n')],
        "done",
    ])
    err = (_tool_results(fr)[0].get("error") or "")
    assert "no prio_7 on this surface" in err and "prior_7" in err, err


def test_a_get_near_miss_notes_the_real_column():
    """.get keeps returning the default on a miss, and the receipt names
    the near column the spelling missed (363#361 burned three blocks on
    'distance to Airway Boulevard' vs the served 'along airway')."""
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car", nearest="the depot")\n'
              'v = r1[0].get("distance_to_depot", 9999)\n'
              'print(v)\n')],
        "done",
    ])
    out = _tool_results(fr)[0]
    assert "error" not in out, out
    assert "9999" in (out.get("printed") or ""), out
    note = out.get("note") or ""
    assert "distance_to_depot" in note and "distance to the_depot" in note, out


def test_rows_repr_names_its_rows():
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nprint(r1)\n')],
        "done",
    ])
    printed = (_tool_results(fr)[0].get("printed") or "")
    assert "car_1 [" in printed and "row(s):" in printed, printed


def test_a_bound_handle_beats_a_coincidental_fid_suggestion():
    """corner_5 typo'd from a bound corner_4 must suggest the model's own
    handle, never a coincidental prior_5 -- the fid correction fires only
    when the PREFIX near-misses a fid family (detect_59 -> detected_59)."""
    reply, fr, _f, _d = _drive([
        [_run('corner_4 = [10, 20]\nprint(corner_5)\n')],
        "done",
    ])
    err = (_tool_results(fr)[0].get("error") or "")
    assert "corner_4" in err, err
    assert "prior_5" not in err, err


def test_a_tied_view_is_not_subscriptable_the_way_dict_keys_is_not():
    """The ways of ONE name tie, so no row is its first.

    Python refuses ``d.keys()[0]`` for exactly this reason -- a view over
    something with no order -- and the model has met that refusal ten
    thousand times. It is not a rule about care: calcity::284#282 took
    ``street_88[0]`` and swept 1 of 10 segments, calcity::26#26 took
    ``road[20:40]`` and covered a third of Neuralia Road, and both read as
    a choice while being a coin flip.

    Iteration and len still work, so the SET is fully usable -- explore it,
    order it, materialise it. Only the arbitrary pick is unsayable.
    """
    from maps.search.parse_compile.python_surface import (
        CodeError, MissionEnv, MissionPython, Rows)

    rows = [{"fid": f"prior_{i}", "name": "Neuralia Road"} for i in range(12)]
    env = MissionEnv(
        run_plan=lambda *a, **k: {},
        rows_of_ref=lambda ref: rows,
        tied_of_ref=lambda ref: True,
        plan_rn=[0], discourse={}, minted=[],
        say_err=lambda *a, **k: None, row_of_fid=lambda f: None)
    view = Rows("r1", MissionPython(env))

    assert len(view) == 12
    assert len(list(view)) == 12, "a tied view still iterates"

    for bad in (0, -1, 5):
        with pytest.raises(CodeError, match="not subscriptable"):
            view[bad]
    with pytest.raises(CodeError, match="not subscriptable"):
        view[20:40]


def test_an_ordered_result_stays_subscriptable():
    """The refusal is about TIES, not about find results.

    ``order_by`` gives a real order, and rank 0 of a real order is a
    meaningful row -- refusing it would delete the ranking the agent just
    asked for.
    """
    from maps.search.parse_compile.python_surface import (
        MissionEnv, MissionPython, Rows)

    rows = [{"fid": f"detected_{i}", "class": "car"} for i in range(6)]
    env = MissionEnv(
        run_plan=lambda *a, **k: {},
        rows_of_ref=lambda ref: rows,
        tied_of_ref=lambda ref: False,
        plan_rn=[0], discourse={}, minted=[],
        say_err=lambda *a, **k: None, row_of_fid=lambda f: None)
    view = Rows("r2", MissionPython(env))
    assert view[0].get("fid") == "detected_0"
    assert len(view[0:3]) == 3


def test_a_tied_views_repr_shows_its_shape_and_leaks_no_point():
    """print(road) must not hand back what the head withholds.

    The name-set head serves len/name/length_m and deliberately no fids and
    no points. The repr handed back both, and calcity::26#26 read a way's
    coordinate off it and flew go_to([2242.27, 3048.07]) -- one way of a
    64-way road, reached by a point the surface had refused to serve.
    """
    from maps.search.parse_compile.python_surface import (
        MissionEnv, MissionPython, Rows)

    rows = [{"fid": f"prior_{5775 + i}", "name": "Neuralia Road",
             "xy": [2242.27 + i, 3048.07 + i]} for i in range(64)]

    def _env(tied):
        return MissionEnv(
            run_plan=lambda *a, **k: {}, rows_of_ref=lambda ref: rows,
            tied_of_ref=lambda ref: tied, plan_rn=[0], discourse={},
            minted=[], say_err=lambda *a, **k: None,
            row_of_fid=lambda f: None)

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
    """A relation column is not MISSING, it is UNASKED-FOR.

    `distance to <ref>` / `facing <ref>` / `<dir> of <ref>` exist only
    where a compare named that ref, so a block asking for one has not
    misspelled a column -- it has named a place these rows were never
    related to, and the executor computes exactly that on request
    (geom_program._auto_attach). Answering with twelve column names that
    are not the one asked for sent the block hunting a spelling: 104
    blocks over 77 episodes died this way across six 500-episode roots,
    95 of them on `distance to <ref>`.
    """
    r = Row({"fid": "prior_3213", "xy": [1.0, 2.0], "class": "house",
             "distance to poppy_boulevard": 12.0}, None)

    # a relate column names the relate, and names the PLACE it was asked
    # about rather than the ones it already carries
    for k in ("distance to redwood", "east of quezon", "facing nipa",
              "along airway", "bearing from great_circle"):
        with pytest.raises(KeyError) as e:
            r[k]
        msg = str(e.value)
        assert "never related to" in msg, msg
        assert "order(nearest=" in msg, msg
        assert k[k.index(" ") + 1:].split(" ")[-1] in msg, msg
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


def test_the_prefix_table_has_one_home():
    """Retyping it is how the two halves drift: the Row would name a
    remedy for a key the executor cannot attach, or stay silent on one it
    can."""
    from maps.search.analyze_scene import geom_program as gp
    assert _AUTOATTACH_PREFIXES is gp._AUTOATTACH_PREFIXES
    # the three families the Row branches on are all covered by it
    assert set(gp._STANDING_SIDE_PREFIXES) <= set(_AUTOATTACH_PREFIXES)
    for p in ("distance to ", "facing ", "east of ", "along ", "color_"):
        assert p in _AUTOATTACH_PREFIXES, p
