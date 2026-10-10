"""The standing ANCHOR in the executor: the model's own discourse anchor
({label, ref}, minted by a near-PP or anchor-superlative) rides every find
result as 'distance to <label>' — the staged-viewpoint law applied to the
discourse anchor (item 2: relational
columns re-anchored to the instruction's landmark; the "the nearest one."
stuck-loop over fresh finds, 11-40 burned turns in >=7/96 episodes, is the
measured target).

Attach-only semantics: never a gate, never a drop; an unresolvable ref
skips without error; no anchor = no column (offline-teacher byte parity).
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    compact_results,
    run_program,
)

_CANDS = [
    {"fid": "d0", "xy": [10.0, 0.0], "class": "car", "search_text": "car"},
    {"fid": "d1", "xy": [40.0, 0.0], "class": "car", "search_text": "car"},
]
_ANCHORS = [
    {"fid": "z_solo", "name": "Manhattan House", "xy": [0.0, 0.0],
     "class": "building"},
    {"fid": "a1", "xy": [0.0, 0.0], "class": "building"},
    {"fid": "a2", "xy": [30.0, 0.0], "class": "building"},
    {"fid": "road_1", "xy": [25.0, 0.0], "class": "road",
     "centerline": [[0.0, 0.0], [50.0, 0.0]]},
]

_FIND = {"ref": "r0", "op": "find",
         "args": {"query": "car", "provenance": "runtime"}}


def _run(standing_anchor=None, cands=None):
    return run_program([dict(c) for c in (cands or _CANDS)],
                       [dict(a) for a in _ANCHORS], {"ops": [_FIND]},
                       standing_anchor=standing_anchor)


def test_standing_anchor_column_rides_every_find_row():
    out = _run({"label": "gate", "ref": "z_solo"})
    rows = out["results"]["r0"]
    assert len(rows) == 2                              # vacuity floor
    assert all(r.get("distance to gate") is not None for r in rows), rows
    assert "distance to gate" in getattr(rows, "op_cols", [])
    header = compact_results(out["results"])["r0"]["rows"].splitlines()[0]
    assert "distance to gate" in header, header


def test_no_anchor_means_no_column():
    out = _run(None)
    for r in out["results"]["r0"]:
        assert not any(k.startswith("distance to ") for k in r), r


def test_unresolvable_ref_skips_without_error():
    # The anchor's fid left this surface (detection eviction): the find is
    # otherwise identical, no error, no column.
    out = _run({"label": "gate", "ref": "z_gone"})
    base = _run(None)
    assert out["results"]["r0"] == base["results"]["r0"]


def test_union_ref_is_the_exact_min_gap_under_one_label():
    # {ref: [fid,...], as: lab} is the labeled UNION ref — the same shape
    # _sub_anchors substitutes for a grounded anchor tie, so the standing
    # column and a sentence's own compare mint IDENTICAL semantics.
    out = _run({"label": "gate", "ref": ["a1", "a2"]})
    direct = run_program(
        [dict(c) for c in _CANDS], [dict(a) for a in _ANCHORS],
        {"ops": [_FIND,
                 {"ref": "r1", "op": "compare",
                  "args": {"items": "r0",
                           "refs": [{"ref": ["a1", "a2"], "as": "gate"}]}}]})
    got = {r["fid"]: r["distance to gate"] for r in out["results"]["r0"]}
    want = {r["fid"]: r["distance to gate"] for r in direct["results"]["r1"]}
    assert got == want and len(got) == 2, (got, want)


def test_attach_is_idempotent_on_rows_already_carrying_the_column():
    carrying = [dict(c, **{"distance to gate": 5.0}) for c in _CANDS]
    out = _run({"label": "gate", "ref": "z_solo"}, cands=carrying)
    base = _run(None, cands=carrying)
    assert out["results"]["r0"] == base["results"]["r0"]
