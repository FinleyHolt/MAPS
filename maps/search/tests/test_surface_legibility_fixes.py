from __future__ import annotations

from maps.query.program import (
    compact_results,
    run_program,
)

_ANCHORS = [
    {"fid": "road_1", "name": "Elm Street", "xy": [50.0, 0.0],
     "class": "road", "centerline": [[0.0, 0.0], [100.0, 0.0]]},
    # A near-square parking polygon: the load path PCA-derives a centerline
    # for EVERY >=2-cell node, so this shape carries one despite being areal.
    {"fid": "lot_1", "name": "Corner Lot", "xy": [20.0, 20.0],
     "class": "parking", "centerline": [[15.0, 20.0], [25.0, 20.0]],
     "footprint_area_m2": 1875.0},
]


def _run(ops, cands, anchors=None):
    return run_program(
        [dict(c) for c in cands],
        [dict(a) for a in (anchors if anchors is not None else _ANCHORS)],
        {"ops": ops})


_COLOR_CANDS = [
    {"fid": "d0", "xy": [10.0, 2.0], "class": "car",
     "color": "white", "color_conf": 0.9},
    {"fid": "d1", "xy": [90.0, -8.0], "class": "car",
     "color": "black", "color_conf": 0.9},
    {"fid": "d2", "xy": [50.0, 3.0], "class": "car"},  # colorless
]


# --------------------------------------------------------------------------- #
# HEAD-TRUNCATION-STATED                                                      #
# --------------------------------------------------------------------------- #


def test_truncated_tied_head_states_both_facts():
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "car",
              "search_text": "car"}
             for i in range(60)]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "car"}}],
               cands)
    env = compact_results(out["results"])["r0"]
    hint = env.get("hint") or ""
    assert "showing the first" in env["shown"], env
    assert "showing the first" not in hint, hint
    assert "equally good matches" in hint, hint
    assert "arbitrary" in hint, hint
    assert env["len"] == 60


# --------------------------------------------------------------------------- #
# LINE-ASPECT-GATE                                                            #
# --------------------------------------------------------------------------- #


def test_near_square_centerline_ref_is_areal():
    out = _run([
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["lot_1"]}},
    ], _COLOR_CANDS)
    row = out["results"]["r1"][0]
    assert "distance to lot_1" in row, sorted(row)                 # areal family attaches
    assert "along_lot_1" not in row, sorted(row)


def test_distance_to_ranks_by_magnitude_either_side():
    cands = [
        {"fid": "d0", "xy": [10.0, 2.0], "class": "car"},   # dist 2
        {"fid": "d1", "xy": [90.0, -8.0], "class": "car"},  # dist 8
        {"fid": "d2", "xy": [50.0, 3.0], "class": "car"},   # dist 3
    ]
    road = dict(_ANCHORS[0], mask=[[[0.0, 0.0], [100.0, 0.0]]])
    magnitude = _run([
        {"ref": "rc", "op": "compare",
         "args": {"items": ["d0", "d1", "d2"], "refs": ["road_1"]}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "rc",
                  "emphasis": [{"key": "distance to road_1", "w": -1}]}},
    ], cands, [road])
    assert [r["fid"] for r in magnitude["results"]["r1"]] == \
        ["d0", "d2", "d1"]


# --------------------------------------------------------------------------- #
# WHERE-THE-CUT-STANDS                                                        #
# --------------------------------------------------------------------------- #


def _cut_hint(xs) -> str:
    """A truncated, ORDERED head's render note, over rows at the given x
    offsets."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(x), 0.0],
              "class": "house", "search_text": "house"}
             for i, x in enumerate(xs)]
    anchors = [{"fid": "road_9", "name": "Glade Avenue", "xy": [0.0, 0.0],
                "class": "road", "centerline": [[0.0, -1.0], [0.0, 1.0]]}]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "house"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "r0", "refs": ["road_9"]}},
                {"ref": "r2", "op": "score_by",
                 "args": {"items": "r1",
                          "emphasis": [{"key": "distance to road_9",
                                        "w": -1}]}}],
               cands, anchors)
    return compact_results(out["results"])["r2"].get("shown") or ""


def _scores(hint):
    """(head_lo, head_hi, next) off the served note."""
    import re
    m = re.search(r"score runs (\S+)\.\.(\S+), and the next row scores (\S+)",
                  hint)
    assert m, hint
    return tuple(float(g) for g in m.groups())


def test_a_truncated_head_says_where_its_ordering_stood_at_the_cut():
    """The one fact a served head structurally cannot show."""
    hint = _cut_hint([float(i) for i in range(60)])
    assert "showing the first" in hint, hint
    assert "ranked on 'distance to road_9'" in hint, hint
    assert "this head's score runs" in hint, hint
    assert "the next row scores" in hint, hint


def test_it_reports_the_COMPOSITE_not_a_column():
    """A multi-key sort is ordered by the SUM; no column is monotone in it."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "house",
              "search_text": "house",
              "color": "white" if i % 2 else "black", "color_conf": 0.9}
             for i in range(60)]
    anchors = [{"fid": "road_9", "name": "Glade Avenue", "xy": [0.0, 0.0],
                "class": "road", "centerline": [[0.0, -1.0], [0.0, 1.0]]}]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "house"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "r0", "refs": ["road_9"]}},
                {"ref": "r2", "op": "score_by",
                 "args": {"items": "r1", "emphasis": [
                     {"key": "distance to road_9", "w": -1},
                     {"key": "color_white", "w": 1}]}}],
               cands, anchors)
    res = out["results"]["r2"]
    assert res.op_cols[0] == "color_white", res.op_cols   # the trap
    hint = compact_results(out["results"])["r2"].get("shown") or ""
    # both keys NAMED, the numbers from the composite
    assert "'color_white'" in hint and "'distance to road_9'" in hint, hint
    lo, _hi, nxt = _scores(hint)
    assert nxt <= lo, hint     # the next row never outranks the page


def test_a_plateau_and_a_real_gap_read_differently():
    """A FACT, not a verdict: the numbers separate the two cases."""
    flat = _scores(_cut_hint([float(i) for i in range(60)]))
    gap = _scores(_cut_hint([float(i) for i in range(24)]
                            + [900.0 + i for i in range(36)]))
    flat_drop = flat[0] - flat[2]
    gap_drop = gap[0] - gap[2]
    assert gap_drop > 10 * flat_drop, (flat, gap)
    # and no verdict rides along
    hint = _cut_hint([float(i) for i in range(60)])
    tail = hint.split("ranked on")[-1]
    for word in ("arbitrary", "meaningless", "should", "try", "narrow"):
        assert word not in tail, (word, hint)


def test_an_unordered_dump_has_no_cut_to_stand_at():
    """No ordering op ran, so there is no column the cut is against."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "car",
              "search_text": "car"} for i in range(60)]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "car"}}], cands)
    env = compact_results(out["results"])["r0"]
    assert "showing the first" in env["shown"], env
    assert "ranked on" not in env["shown"] + env.get("hint", ""), env


# --------------------------------------------------------------------------- #
# SAME-NAME-TIE-COVERING-OCCASION                                             #
# --------------------------------------------------------------------------- #


def test_a_uniform_class_tie_keeps_the_generic_hint():
    """Sixty tied UNNAMED cars: a whole-town pass is a thinned sample, not the cue's
    remedy, so the covering clause stays off the generic tie."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "car",
              "search_text": "car"}
             for i in range(60)]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "car"}}],
               cands)
    env = compact_results(out["results"])["r0"]
    hint = env.get("hint") or ""
    assert "arbitrary" in hint, hint
    assert "pieces of ONE named thing" not in hint, hint


def test_a_folded_result_never_says_how_a_grid_it_does_not_serve_renders():
    """Only an envelope that serves rows says how they render; a folded
    result keeps its hint, and nothing names the executor's rN."""
    import json

    from maps.search.tests.fakes import CANDS, drive, run, tool_results
    cands = CANDS + [{"fid": f"car_{i}", "xy": [5.0 * i, 7.0], "class": "car",
                      "search_text": "car"} for i in range(3, 70)]
    _, fr, _, _ = drive([[run('cars = find("car")\n'
                              'x = cars.order(nearest="prior_1")\nx')],
                         "done"], cands=cands)
    env = tool_results(fr)[0]["result"]["results"]["x"]
    assert "showing the first" not in env["hint"], env["hint"]
    assert env["hint"].startswith("cars: 69 of 72 rows matched"), env["hint"]
    assert env["shown"].startswith("showing the first 24 of 69"), env
    assert "<rN>" not in json.dumps(env), env
