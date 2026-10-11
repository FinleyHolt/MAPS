#!/usr/bin/env python
"""ORDER-FREEDOM contract: the agent tool-calls in ANY order."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    ProgramError,
    _grouped_keys,
    compact_results,
    run_program,
)
from maps.search.parse_compile.reasoner_loop import (
    _PREFLIGHT_COLOR_NOTE,
    _PREFLIGHT_RUNTIME_NOTE,
    run_geom_program_tool,
)

# Three cars strung along a road with a stored centerline, plus a plain
# building anchor (no centerline) and a landmark the cars are away from.
_CANDS = [
    {"fid": "d0", "xy": [10.0, 2.0], "class": "car", "color": "white"},
    {"fid": "d1", "xy": [90.0, -8.0], "class": "car", "color": "white"},
    {"fid": "d2", "xy": [50.0, 3.0], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "road_1", "name": "Elm Street", "xy": [50.0, 0.0],
     "class": "road", "centerline": [[0.0, 0.0], [100.0, 0.0]]},
    {"fid": "church", "name": "St Church", "xy": [200.0, 200.0],
     "class": "building"},
]


def _run(ops, cands=None, anchors=None):
    return run_program(
        [dict(c) for c in (cands if cands is not None else _CANDS)],
        [dict(a) for a in (anchors if anchors is not None else _ANCHORS)],
        {"ops": ops})


def _err(ops, cands=None, anchors=None) -> str:
    try:
        _run(ops, cands, anchors)
    except ProgramError as e:
        return str(e)
    raise AssertionError(f"expected ProgramError for {ops!r}")


_FIND_CARS = {"ref": "r0", "op": "find",
              "args": {"provenance": "runtime"}}


# --------------------------------------------------------------------------- #
# ALONG/OFF-LAZY: the chart parameter rides the key                            #
# --------------------------------------------------------------------------- #


def test_along_emphasis_auto_attaches_without_compare():
    out = _run([
        _FIND_CARS,
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0",
                  "emphasis": [{"key": "along road_1", "w": -1}]}},
    ])
    fids = [r["fid"] for r in out["results"]["r1"]]
    assert fids == ["d0", "d2", "d1"], fids  # arc order along the road
    assert out["results"]["r1"][0]["along road_1"] is not None


def test_off_filter_auto_attaches_without_compare():
    out = _run([
        _FIND_CARS,
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "$r0", "key": "distance to road_1",
                  "op": "le", "value": 5.0}},
    ])
    fids = {r["fid"] for r in out["results"]["r1"]}
    assert fids == {"d0", "d2"}, fids  # d1 sits 8 m off the road


def test_along_matches_explicit_frame_chart():
    auto = _run([
        _FIND_CARS,
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0",
                  "emphasis": [{"key": "along road_1", "w": -1}]}},
    ])["results"]["r1"]
    explicit = _run([
        _FIND_CARS,
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0", "frame": {"along": "road_1"}}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "$r1",
                  "emphasis": [{"key": "along road_1", "w": -1}]}},
    ])["results"]["r2"]
    assert [r["fid"] for r in auto] == [r["fid"] for r in explicit]
    assert auto[0]["along road_1"] == explicit[0]["along road_1"]


def test_along_on_non_centerline_fid_falls_through_loud():
    # 'church' is a live fid but carries no centerline -> no chart attaches
    # and the guard fires with the along_/off_ family remedy.
    msg = _err([
        _FIND_CARS,
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0",
                  "emphasis": [{"key": "along church", "w": -1}]}},
    ])
    assert "along church" in msg, msg


# --------------------------------------------------------------------------- #
# FAMILY-REMEDY: the missing key's OWN producing idiom, in BOTH guards         #
# --------------------------------------------------------------------------- #


def test_name_match_remedy_names_find_query_in_both_guards():
    for msg in (
        _err([_FIND_CARS,
              {"ref": "r1", "op": "score_by",
               "args": {"items": "$r0",
                        "emphasis": [{"key": "name_match", "w": 1}]}}]),
        _err([_FIND_CARS,
              {"ref": "r1", "op": "filter_items",
               "args": {"items": "$r0", "key": "name_match",
                        "op": "ge", "value": 0.5}}]),
    ):
        assert "find{query" in msg, msg
        assert "compare" not in msg.split("numeric keys present")[0], msg


def test_frame_column_remedy_names_compare_frame_in_both_guards():
    for msg in (
        _err([_FIND_CARS,
              {"ref": "r1", "op": "score_by",
               "args": {"items": "$r0",
                        "emphasis": [{"key": "ahead of road_1", "w": 1}]}}]),
        _err([_FIND_CARS,
              {"ref": "r1", "op": "filter_items",
               "args": {"items": "$r0", "key": "behind road_1",
                        "op": "le", "value": 7}}]),
    ):
        assert "frame" in msg and "compare" in msg, msg
    msg = _err([_FIND_CARS,
                {"ref": "r1", "op": "filter_items",
                 "args": {"items": "$r0", "key": "ord_rl",
                          "op": "eq", "value": 7}}])
    assert "no item carries" in msg and "keys present" in msg, msg
    assert "frame" not in msg, msg


def test_coordinate_shaped_ref_remedy_names_the_point_fact():
    # A number-shaped <ref> extends the fid idiom to a coordinate -- the remedy states
    # the FACT (a number pair is a POINT; a point carries columns only under the label
    # it is given) and never "not a fid on this surface" alone. It reaches the SENTENCE
    # channel: a spoken "near 1246 1100" mints the label 1246_1100, and a later "the
    # nearest one." orders by 'distance to 1246_1100' over rows that sentence never
    # related to it.
    msg = _err([_FIND_CARS,
                {"ref": "r1", "op": "score_by",
                 "args": {"items": "$r0",
                          "emphasis": [{"key": "distance to 1593_5", "w": -1}]}}])
    assert "is not a fid" in msg and "POINT" in msg, msg
    assert "distance to <label>" in msg, msg
    assert "refs:[" not in msg and "score_by" not in msg, msg


def test_key_census_prints_each_key_once():
    # longest-prefix family assignment: brg_gap_* members never re-list
    # under brg_, grp_size never re-lists under grp.
    present = ["facing xy", "bearing from xy", "grp", "grp_size", "distance to road_1"]
    tokens = _grouped_keys(present).split(", ")
    for k in present:
        assert tokens.count(k) == 1, (k, tokens)


# --------------------------------------------------------------------------- #
# CONTACT-FACT: brg_gap_ absent where d_=0 states the geometry                 #
# --------------------------------------------------------------------------- #


def test_brg_gap_on_all_contact_pool_states_the_contact_fact():
    # one candidate exactly ON the anchor point: d_=0 everywhere, brg_gap_
    # legitimately absent. The error must state contact + the brg_ sibling,
    # never "not a fid on this surface" (the fid IS live; the compare ran).
    cands = [{"fid": "d0", "xy": [200.0, 200.0], "class": "car"}]
    for msg in (
        _err([_FIND_CARS,
              {"ref": "r1", "op": "score_by",
               "args": {"items": "$r0",
                        "emphasis": [{"key": "facing church", "w": 1}]}}],
             cands=cands),
        _err([_FIND_CARS,
              {"ref": "r1", "op": "filter_items",
               "args": {"items": "$r0", "key": "facing church",
                        "op": "ge", "value": 0.0}}],
             cands=cands),
    ):
        assert "contact" in msg and "bearing from church" in msg, msg
        assert "not a fid" not in msg, msg


# --------------------------------------------------------------------------- #
# VALUE-REQUIRED: filter_items without value is loud, never TypeError/dead     #
# --------------------------------------------------------------------------- #


def test_filter_omitted_value_is_loud_and_names_op_and_arg():
    msg = _err([_FIND_CARS,
                {"ref": "r1", "op": "filter_items",
                 "args": {"items": "$r0", "key": "distance to church", "op": "lt"}}])
    assert "'lt'" in msg and "'value'" in msg and "truthy" in msg, msg


def test_filter_default_eq_without_value_is_loud_not_a_dead_filter():
    msg = _err([_FIND_CARS,
                {"ref": "r1", "op": "filter_items",
                 "args": {"items": "$r0", "key": "class"}}])
    assert "'eq'" in msg and "'value'" in msg, msg


def test_filter_truthy_without_value_still_works():
    out = _run([_FIND_CARS,
                {"ref": "r1", "op": "filter_items",
                 "args": {"items": "$r0", "key": "class", "op": "truthy"}}])
    assert len(out["results"]["r1"]) == 3


def test_filter_numeric_op_requires_numeric_value():
    msg = _err([_FIND_CARS,
                {"ref": "r1", "op": "filter_items",
                 "args": {"items": "$r0", "key": "distance to church",
                          "op": "lt", "value": "near"}}])
    assert "NUMERIC" in msg and "'lt'" in msg, msg


# --------------------------------------------------------------------------- #
# EMPTY-ANCHOR-NOTES: the tag census survives the prior-surface marker         #
# --------------------------------------------------------------------------- #


def test_empty_anchor_find_keeps_the_tag_census_note():
    # Anchors carrying real OSM tags: a query matching nothing ships the
    # surface census (top tag key=value counts + name samples) on the miss.
    anchors = [
        {"fid": "prior_0", "name": "Elm Street", "xy": [50.0, 0.0],
         "tags": {"highway": "residential"},
         "centerline": [[0.0, 0.0], [100.0, 0.0]]},
        {"fid": "prior_1", "name": "St Church", "xy": [200.0, 200.0],
         "tags": {"building": "church"}},
    ]
    out = _run([{"ref": "r0", "op": "find",
                 "args": {"provenance": "prior", "query": "zzzqx"}}],
               anchors=anchors)
    env = compact_results(out["results"])["r0"]
    assert env["len"] == 0
    hint = env.get("hint") or ""
    assert "surface census" in hint, hint          # the census note SHIPPED
    assert "highway=residential(1)" in hint, hint  # tag key=value counts
    assert "Elm Street" in hint, hint              # name samples
    # the truthful searched-scope fact rides on top of the census
    assert "searched the prior map rows (provenance:'prior')" in hint, hint


# --------------------------------------------------------------------------- #
# PREFLIGHT-NOTES: phase facts stated where they are read                      #
# --------------------------------------------------------------------------- #

_PRIOR_NS = {"cands": [],
             "anchors": [dict(a) for a in _ANCHORS], "cell_m": 1.0}


def test_preflight_color_note_on_emphasis_only_attach():
    # The NEW lazy path -- no find{color} anywhere: a pre-takeoff emphasis
    # naming color_red must state the phase fact, not just "no color".
    out = run_geom_program_tool(
        {"program": {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
            {"ref": "r1", "op": "score_by",
             "args": {"items": "$r0",
                      "emphasis": [{"key": "color_red", "w": 1}]}},
        ]}}, namespace=_PRIOR_NS, preflight=True)
    assert _PREFLIGHT_COLOR_NOTE in str(out.get("error")), out


def test_preflight_color_note_on_find_color():
    out = run_geom_program_tool(
        {"program": {"ops": [
            {"ref": "r0", "op": "find",
             "args": {"provenance": "prior", "color": "red"}},
        ]}}, namespace=_PRIOR_NS, preflight=True)
    assert "error" not in out, out
    assert _PREFLIGHT_COLOR_NOTE in str(out["results"]["r0"].get("hint")), out


def test_preflight_runtime_find_states_the_detection_fact():
    out = run_geom_program_tool(
        {"program": {"ops": [
            {"ref": "r0", "op": "find",
             "args": {"provenance": "runtime", "query": "car"}},
        ]}}, namespace=_PRIOR_NS, preflight=True)
    assert "error" not in out, out
    assert _PREFLIGHT_RUNTIME_NOTE in str(out["results"]["r0"].get("hint")), out


def test_per_stop_round_serves_no_phase_note():
    # the notes are PREFLIGHT-only: the same programs per-stop stay clean.
    ns = {"cands": [dict(c) for c in _CANDS],
          "anchors": [dict(a) for a in _ANCHORS], "cell_m": 1.0}
    out = run_geom_program_tool(
        {"program": {"ops": [
            {"ref": "r0", "op": "find",
             "args": {"provenance": "runtime", "color": "red"}},
        ]}}, namespace=ns)
    assert _PREFLIGHT_COLOR_NOTE not in str(out), out
    assert _PREFLIGHT_RUNTIME_NOTE not in str(out), out
