#!/usr/bin/env python
"""The expressiveness-audit legibility fixes, pinned.

  * COLOR-BROWSE-ORDERED: a query-less find carrying ``color`` orders by
    its own attached ``color_<word>`` column (ordering only — every row,
    colorless included, stays in the rN) and renders rows instead of the
    census-only null that read as "no results".
  * HEAD-TRUNCATION-STATED: a rendered head shorter than the result says
    "showing the first K of N", and a tie-degenerate ordering owns its
    arbitrariness instead of posing as best-first.
  * LINE-ASPECT-GATE: a near-square ref with a stored (PCA-derived)
    centerline is AREAL — d_/dc_/brg_ attach; only an elongated ref (or a
    footprint-less road prior) is a LINE with along_/off_.

Pure (no GPU / tile / vLLM). ``pytest maps/search/tests/test_surface_legibility_fixes.py``.
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    ProgramError,
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
     "aspect_ratio": 1.4, "footprint_area_m2": 1875.0},
]


def _run(ops, cands, anchors=None):
    return run_program(
        [dict(c) for c in cands],
        [dict(a) for a in (anchors if anchors is not None else _ANCHORS)],
        {"ops": ops})


# --------------------------------------------------------------------------- #
# COLOR-BROWSE-ORDERED                                                        #
# --------------------------------------------------------------------------- #

_COLOR_CANDS = [
    {"fid": "d0", "xy": [10.0, 2.0], "class": "car",
     "color": "white", "color_conf": 0.9},
    {"fid": "d1", "xy": [90.0, -8.0], "class": "car",
     "color": "black", "color_conf": 0.9},
    {"fid": "d2", "xy": [50.0, 3.0], "class": "car"},  # colorless
]


def test_color_browse_orders_by_color_column():
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"color": "black", "provenance": "runtime"}}],
               _COLOR_CANDS)
    rows = list(out["results"]["r0"])
    assert [r["fid"] for r in rows][0] == "d1"          # blackest first
    assert [r["fid"] for r in rows][-1] == "d2"         # colorless last
    assert len(rows) == 3                                # nothing dropped
    env = compact_results(out["results"])["r0"]
    assert env.get("fids") or env.get("rows"), env       # rows RENDER
    hint = env.get("hint") or ""
    assert "ordered by color_black" in hint, hint
    assert "nothing dropped" in hint, hint


# --------------------------------------------------------------------------- #
# HEAD-TRUNCATION-STATED                                                      #
# --------------------------------------------------------------------------- #


def test_truncated_tied_head_states_both_facts():
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "car",
              "search_text": "car"}
             for i in range(60)]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "car", "provenance": "runtime"}}],
               cands)
    env = compact_results(out["results"])["r0"]
    hint = env.get("hint") or ""
    assert "showing the first" in hint, hint
    # columnless tie wording: name_match no longer renders as a grid column
    # (audit 7/8 not-observed), so the clause states uniform match quality
    # without naming the column or quoting an invisible value
    assert "equally good matches" in hint, hint
    assert "arbitrary" in hint, hint
    assert env["len"] == 60


# --------------------------------------------------------------------------- #
# LINE-ASPECT-GATE                                                            #
# --------------------------------------------------------------------------- #


def test_near_square_centerline_ref_is_areal():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0", "refs": ["lot_1"]}},
    ], _COLOR_CANDS)
    row = out["results"]["r1"][0]
    assert "distance to lot_1" in row, sorted(row)                 # areal family attaches
    assert "along_lot_1" not in row, sorted(row)


def test_distance_to_ranks_by_magnitude_either_side():
    # sides are presence columns; 'distance to <line>' is the either-side
    # proximity sort (the retired abs idiom, now a named column).
    cands = [
        {"fid": "d0", "xy": [10.0, 2.0], "class": "car"},   # dist 2
        {"fid": "d1", "xy": [90.0, -8.0], "class": "car"},  # dist 8
        {"fid": "d2", "xy": [50.0, 3.0], "class": "car"},   # dist 3
    ]
    magnitude = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0",
                  "emphasis": [{"key": "distance to road_1", "w": -1}]}},
    ], cands)
    assert [r["fid"] for r in magnitude["results"]["r1"]] == \
        ["d0", "d2", "d1"]


def test_distance_to_on_a_line_charts_and_orders():
    # 'distance to <line>' is the lateral magnitude (either side): naming
    # it in an emphasis auto-attaches the chart and orders nearest-line-
    # first — the old d_-on-a-line refusal is retired with the rename.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0",
                  "emphasis": [{"key": "distance to road_1", "w": -1}]}},
    ], _COLOR_CANDS)
    rows = out["results"]["r1"]
    vals = [r["distance to road_1"] for r in rows]
    assert vals == sorted(vals), vals


# --------------------------------------------------------------------------- #
# WHERE-THE-CUT-STANDS                                                        #
# --------------------------------------------------------------------------- #


def _cut_hint(xs) -> str:
    """A truncated, ORDERED head over rows at the given x offsets."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(x), 0.0],
              "class": "house", "search_text": "house"}
             for i, x in enumerate(xs)]
    anchors = [{"fid": "road_9", "name": "Glade Avenue", "xy": [0.0, 0.0],
                "class": "road", "centerline": [[0.0, -1.0], [0.0, 1.0]]}]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "house", "provenance": "runtime"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "r0", "refs": ["road_9"]}},
                {"ref": "r2", "op": "score_by",
                 "args": {"items": "r1",
                          "emphasis": [{"key": "distance to road_9",
                                        "w": -1}]}}],
               cands, anchors)
    return compact_results(out["results"])["r2"].get("hint") or ""


def _scores(hint):
    """(head_lo, head_hi, next) off the served note."""
    import re
    m = re.search(r"score runs (\S+)\.\.(\S+), and the next row scores (\S+)",
                  hint)
    assert m, hint
    return tuple(float(g) for g in m.groups())


def test_a_truncated_head_says_where_its_ordering_stood_at_the_cut():
    """The one fact a served head structurally cannot show.

    Every rendered row carries its own values and the row just past the cut
    carries none, so a head of 24 of 3,414 says nothing about whether the
    answer is at rank 25. The episodes that never got a crop onto the
    target queried into a match set of median 3,080 rows against 146 for
    the ones that did, and their correct row sat at rank 0 one time in
    eight against six in ten.
    """
    hint = _cut_hint([float(i) for i in range(60)])
    assert "showing the first" in hint, hint
    assert "ranked on 'distance to road_9'" in hint, hint
    assert "this head's score runs" in hint, hint
    assert "the next row scores" in hint, hint


def test_it_reports_the_COMPOSITE_not_a_column():
    """A multi-key sort is ordered by the SUM; no column is monotone in it.

    `op_cols` is `sorted(emphasis)`, so on the legend's own
    `near=<place>, color=<word>` shape the alphabetically-first key is
    `color_white` -- and the row past the cut carries its BEST possible
    value, which read as "the cut discarded a row better than everything
    shown" on a head that had separated cleanly.
    """
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "house",
              "search_text": "house",
              "color": "white" if i % 2 else "black", "color_conf": 0.9}
             for i in range(60)]
    anchors = [{"fid": "road_9", "name": "Glade Avenue", "xy": [0.0, 0.0],
                "class": "road", "centerline": [[0.0, -1.0], [0.0, 1.0]]}]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "house", "provenance": "runtime"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "r0", "refs": ["road_9"]}},
                {"ref": "r2", "op": "score_by",
                 "args": {"items": "r1", "emphasis": [
                     {"key": "distance to road_9", "w": -1},
                     {"key": "color_white", "w": 1}]}}],
               cands, anchors)
    res = out["results"]["r2"]
    assert res.op_cols[0] == "color_white", res.op_cols   # the trap
    hint = compact_results(out["results"])["r2"].get("hint") or ""
    # both keys NAMED, the numbers from the composite
    assert "'color_white'" in hint and "'distance to road_9'" in hint, hint
    lo, _hi, nxt = _scores(hint)
    assert nxt <= lo, hint     # the next row never outranks the page


def test_a_plateau_and_a_real_gap_read_differently():
    """A FACT, not a verdict: the numbers separate the two cases.

    The distinction is a JUMP at the cut, not a wider spacing -- the score
    is min-max normalised, so uniformly-spread rows look the same at any
    scale, which is correct: "is rank 25 as good as rank 24" is a relative
    question.
    """
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
                 "args": {"query": "car", "provenance": "runtime"}}], cands)
    hint = compact_results(out["results"])["r0"].get("hint") or ""
    assert "showing the first" in hint, hint
    assert "this head runs" not in hint, hint


def test_an_untruncated_head_stays_silent():
    """Stated only when it BITES -- the whole set is on the page."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "car",
              "search_text": "car"} for i in range(5)]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "car", "provenance": "runtime"}}], cands)
    hint = compact_results(out["results"])["r0"].get("hint") or ""
    assert "this head runs" not in hint, hint


# --------------------------------------------------------------------------- #
# SAME-NAME-TIE-COVERING-OCCASION                                             #
# --------------------------------------------------------------------------- #



def test_a_uniform_class_tie_keeps_the_generic_hint():
    """Sixty tied UNNAMED cars: a whole-town pass is a thinned sample,
    not the cue's remedy, so the covering clause stays off the generic
    tie."""
    cands = [{"fid": f"d{i:03d}", "xy": [float(i), 0.0], "class": "car",
              "search_text": "car"}
             for i in range(60)]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"query": "car", "provenance": "runtime"}}],
               cands)
    env = compact_results(out["results"])["r0"]
    hint = env.get("hint") or ""
    assert "arbitrary" in hint, hint
    assert "pieces of ONE named thing" not in hint, hint


