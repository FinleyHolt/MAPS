"""The staged-viewpoint surface in the executor (act.stage's vN handles)."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    _attach_standing_columns,
    run_program,
)

# Viewer at the origin looking EAST: east is ahead, south is the viewer's
# right (right = cw(up); up = (1, 0) -> right = (0, -1)).
_V1 = {"handle": "v1", "from_xy": [0.0, 0.0], "up_unit": [1.0, 0.0]}
_STAGED = {"v1": _V1}

_CANDS = [
    {"fid": "c_e", "xy": [10.0, 0.0], "class": "car"},    # dead ahead
    {"fid": "c_w", "xy": [-10.0, 0.0], "class": "car"},   # behind
    {"fid": "c_n", "xy": [0.0, 10.0], "class": "car"},    # viewer's LEFT
    {"fid": "c_s", "xy": [0.0, -10.0], "class": "car"},   # viewer's RIGHT
]


def _rows_by_fid(rows):
    return {r["fid"]: r for r in rows}


def test_attach_signs_are_the_cw_handedness_law():
    out = _rows_by_fid(_attach_standing_columns(_CANDS, _V1))
    assert out["c_e"]["ahead of v1"] == 10.0
    assert "behind v1" not in out["c_e"]
    assert out["c_w"]["behind v1"] == 10.0
    assert out["c_n"]["left of v1"] == 10.0
    assert "right of v1" not in out["c_n"]
    assert out["c_s"]["right of v1"] == 10.0
    # on-axis lateral 0.0 lands on the right/ahead member (_side_pair law)
    assert out["c_e"]["right of v1"] == 0.0


def test_attach_copies_rows_never_mutates_the_surface():
    before = [dict(c) for c in _CANDS]
    _attach_standing_columns(_CANDS, _V1)
    assert _CANDS == before


def test_standing_columns_ride_every_find_result():
    out = run_program(
        [dict(c) for c in _CANDS], [],
        {"ops": [{"ref": "r0", "op": "find",
                  "args": {"provenance": "runtime"}}]},
        staged=_STAGED, standing="v1")
    rows = _rows_by_fid(out["results"]["r0"])
    assert rows["c_n"]["left of v1"] == 10.0
    assert rows["c_s"]["right of v1"] == 10.0


def test_staged_handle_ref_attaches_the_stance_columns():
    out = run_program(
        [dict(c) for c in _CANDS], [],
        {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0", "refs": ["v1"]}},
        ]},
        staged=_STAGED)
    rows = _rows_by_fid(out["results"]["r1"])
    assert rows["c_e"]["distance to v1"] == 10.0
    assert rows["c_e"]["ahead of v1"] == 10.0
    assert rows["c_n"]["left of v1"] == 10.0
    assert "bearing from v1" not in rows["c_e"]


def test_no_stage_means_no_columns_anywhere():
    # The vacuity floor: without staged/standing the surface is untouched
    # and the columns exist nowhere (offline-teacher byte parity).
    out = run_program(
        [dict(c) for c in _CANDS], [],
        {"ops": [{"ref": "r0", "op": "find",
                  "args": {"provenance": "runtime"}}]})
    for r in out["results"]["r0"]:
        assert not any(k.endswith(" v1") for k in r)


def test_null_form_args_behave_as_absent():
    # The required-form schema fills unused slots with null; the executor
    # strips them, so the null form and the bare form are ONE program.
    out = run_program(
        [{"fid": "c1", "xy": [1.0, 2.0], "class": "car",
          "search_text": "car"}], [],
        {"ops": [{"ref": "r0", "op": "find",
                  "args": {"provenance": "runtime", "query": None,
                           "color": None}}]})
    assert len(out["results"]["r0"]) == 1


def test_find_query_naming_a_fid_returns_that_row():
    out = run_program(
        [{"fid": "detected_3", "xy": [1.0, 2.0], "class": "car",
          "search_text": "car"}], [],
        {"ops": [{"ref": "r0", "op": "find",
                  "args": {"query": "detected_3"}}]})
    rows = out["results"]["r0"]
    assert len(rows) == 1 and rows[0]["fid"] == "detected_3"


def test_side_columns_render_contiguous_in_fixed_order():
    from maps.search.analyze_scene.geom_program import _grid_columns
    rows = [
        {"fid": "a", "xy": [1, 2], "right of v1": 3.0, "behind v1": 1.0,
         "distance to lot": 9.0},
        {"fid": "b", "xy": [2, 3], "left of v1": 4.0,
         "distance to lot": 2.0, "ahead of v1": 6.0},
    ]
    cols = _grid_columns(rows)
    fam = [c for c in cols if c.endswith(" v1")]
    assert fam == ["left of v1", "right of v1", "ahead of v1", "behind v1"]
    i = [cols.index(c) for c in fam]
    assert i == list(range(i[0], i[0] + 4))  # contiguous
