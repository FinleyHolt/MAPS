"""The tool surface is loud on malformed shapes and self-describing about
its own state (a clean RL target). The pinned guards:

- Fix 1: compare frame with {reference, from} (viewer-form 'from' supplied
  alongside declared-view 'reference') raises a FORM-NAMING error instead of
  silently dropping 'from' and misdirecting to a 'needs side' error.
- Fix 4: no result ever mentions the deleted arrangement family; a road
  chart attaches its own along_/off_ columns.

The order-free tests here overlap ``test_order_freedom.py``'s surface (kept
separate: these pin the self-description of order-free resolution errors).
"""
from __future__ import annotations

import re

import pytest

from maps.search.analyze_scene.geom_program import (
    ProgramError,
    compact_results,
    run_program,
)

# a small runtime pool + a point anchor (shed) and a road (centerline)
_GRID = [
    {"fid": "c00", "xy": [10.0, 10.0], "class": "car", "color": "red"},
    {"fid": "c01", "xy": [15.0, 10.4], "class": "car", "color": "blue"},
    {"fid": "c02", "xy": [20.0, 10.8], "class": "car", "color": "white"},
    {"fid": "c10", "xy": [10.0, 22.0], "class": "car", "color": "red"},
    {"fid": "c11", "xy": [15.0, 22.4], "class": "car", "color": "blue"},
    {"fid": "c12", "xy": [20.0, 22.8], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "road_1", "name": "Main Street", "xy": [50.0, 0.0], "class": "road",
     "centerline": [[0.0, 0.0], [100.0, 0.0]]},
    {"fid": "shed_1", "name": "Shed", "xy": [15.0, -30.0], "class": "building"},
]


def _run(ops):
    return run_program([dict(c) for c in _GRID],
                       [dict(a) for a in _ANCHORS], {"ops": ops})["results"]


def _hint_text(env) -> str:
    h = env.get("hint") or env.get("hints") or ""
    return " ".join(h) if isinstance(h, list) else str(h)


# --- Fix 1 -----------------------------------------------------------------
def test_fix1_frame_reference_with_from_raises():
    with pytest.raises(ProgramError) as e:
        _run([{"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
              {"ref": "r1", "op": "compare",
               "args": {"items": "$r0", "refs": ["shed_1"],
                        "frame": {"reference": "shed_1", "from": "shed_1"}}}])
    msg = str(e.value)
    assert "VIEWER form" in msg
    assert "from" in msg and "reference" in msg
    # it must NOT misdirect to the areal 'needs side' remedy
    assert "needs its 'side'" not in msg


def test_fix1_viewer_form_toward_from_still_valid():
    # the honest form {toward, from} must remain accepted (no over-fire)
    res = _run([{"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "$r0", "refs": ["shed_1"],
                          "frame": {"toward": "road_1", "from": "shed_1"}}}])
    cols = compact_results(res)["r1"].get("columns", [])
    assert any(c.startswith(("left of ", "right of ", "ahead of ", "behind "))
               for c in cols)


# --- Fix 4 -----------------------------------------------------------------
def test_fix4_bare_compare_serves_no_arrangement_note():
    # the arrangement family is DELETED: a bare compare's hint never
    # mentions it (the absent-family note died with the columns).
    res = _run([{"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "$r0", "refs": ["shed_1"]}}])
    env = compact_results(res)["r1"]
    assert "arrangement" not in _hint_text(env)


def test_fix4_road_chart_attaches_its_columns():
    res = _run([{"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
                {"ref": "r1", "op": "compare",
                 "args": {"items": "$r0", "refs": ["road_1"],
                          "frame": {"along": "road_1"}}}])
    env = compact_results(res)["r1"]
    assert "arrangement" not in _hint_text(env)
    cols = env.get("columns", [])
    assert any(c.startswith("along ") for c in cols), cols
    assert not any(c in ("ord_lr", "ord_rl", "row_lo") for c in cols), cols


def test_census_near_cell_serves_stored_precision():
    # a same-named ref fanning to many instances reduces to near_/spread_ census
    # cells; near_ IS the stored d_ value and must serve VERBATIM (2 dp), never a
    # render-time 1 dp re-round (which collapsed a 0.04 m gap to 0.0 beside the
    # row's own d_ column -- the _fmt_float "quantize once" law).
    from maps.search.analyze_scene.geom_program import _anchor_fanout_census
    car = [{"fid": "d0", "xy": [0.0, 0.0], "class": "car"}]
    anch = [{"fid": "t0", "name": "Terrace", "xy": [14.34, 0.0],
             "class": "building"}] + [
        {"fid": f"t{i}", "name": "Terrace", "xy": [40.0 + i * 5, 0.0],
         "class": "building"} for i in range(1, 10)]
    out = run_program(car, anch, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "Terrace", "provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": ["d0"], "refs": ["$r0"]}}]})
    rrows, _ = _anchor_fanout_census(out["results"]["r1"])
    near = [row["near_terrace"] for row in rrows if "near_terrace" in row]
    assert near and abs(near[0] - 14.34) < 1e-9, near  # 14.34 verbatim, not 14.3


# --- Order freedom -----------------------------------------------------------
def test_order_free_forward_ref_resolves():
    # an op may reference a handle defined LATER; the topological sort runs the
    # producer first, so what used to be a forward_ref ERROR now resolves.
    res = _run([
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["shed_1"]}},          # uses r0 ...
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},  # ... defined AFTER
    ])
    assert isinstance(res.get("r1"), list) and res["r1"]  # resolved, no error


def test_order_free_outcome_identical_across_permutations():
    # SAME emitted-last op (score_by r2); only the upstream find/compare order
    # differs (B forward-refs r0). Pure dataflow => byte-identical result.
    find = {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}}
    comp = {"ref": "r1", "op": "compare",
            "args": {"items": "r0", "refs": ["shed_1"]}}
    sort = {"ref": "r2", "op": "score_by",
            "args": {"items": "r1", "emphasis": [{"key": "distance to shed_1", "w": -1}]}}
    a = [x["fid"] for x in _run([find, comp, sort])["r2"][:6]]
    b = [x["fid"] for x in _run([comp, find, sort])["r2"][:6]]  # forward ref
    assert a == b and a


def test_order_free_duplicate_handle_errors():
    with pytest.raises(ProgramError) as e:
        _run([{"ref": "r0", "op": "find", "args": {"query": "car"}},
              {"ref": "r0", "op": "find", "args": {"query": "road"}}])
    assert "two ops" in str(e.value)


def test_order_free_cycle_errors():
    with pytest.raises(ProgramError) as e:
        _run([{"ref": "r0", "op": "score_by",
               "args": {"items": "r1", "emphasis": [{"key": "name_match", "w": 1}]}},
              {"ref": "r1", "op": "score_by",
               "args": {"items": "r0", "emphasis": [{"key": "name_match", "w": 1}]}}])
    assert "cycle" in str(e.value)
