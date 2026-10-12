"""A truthy cut, the census on an empty find, and the pre-takeoff phase notes."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    compact_results,
    run_program,
)
from maps.search.parse_compile.carry import (
    _preflight_color_note,
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


# --------------------------------------------------------------------------- #
# TRUTHY: a key-presence cut takes no value                                    #
# --------------------------------------------------------------------------- #


def test_filter_truthy_without_value_still_works():
    out = _run([{"ref": "r1", "op": "filter_items",
                 "args": {"items": ["d0", "d1", "d2"], "key": "class",
                          "op": "truthy"}}])
    assert len(out["results"]["r1"]) == 3


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
                 "args": {"query": "zzzqx"}}],
               anchors=anchors)
    env = compact_results(out["results"])["r0"]
    assert env["len"] == 0
    hint = env.get("hint") or ""
    assert "surface census" in hint, hint          # the census note SHIPPED
    assert "highway=residential(1)" in hint, hint  # tag key=value counts
    assert "Elm Street" in hint, hint              # name samples
    # the truthful searched-scope fact rides on top of the census
    assert "searched every surface row" in hint, hint


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
            {"ref": "r0", "op": "find", "args": {}},
            {"ref": "r1", "op": "score_by",
             "args": {"items": "r0",
                      "emphasis": [{"key": "color_red", "w": 1}]}},
        ]}}, namespace=_PRIOR_NS, preflight=True)
    assert _preflight_color_note() in str(out.get("error")), out


def test_per_stop_round_serves_no_phase_note():
    # the note is PREFLIGHT-only: the same program per-stop stays clean.
    ns = {"cands": [dict(c) for c in _CANDS],
          "anchors": [dict(a) for a in _ANCHORS], "cell_m": 1.0}
    out = run_geom_program_tool(
        {"program": {"ops": [
            {"ref": "r0", "op": "find", "args": {}},
            {"ref": "r1", "op": "score_by",
             "args": {"items": "r0",
                      "emphasis": [{"key": "color_red", "w": 1}]}},
        ]}}, namespace=ns)
    assert _preflight_color_note() not in str(out), out
