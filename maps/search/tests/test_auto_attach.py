"""A relation column named in filter_items or a score_by emphasis attaches without
an explicit compare, through the same compare op."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program

# three cars + one landmark ("church" at [100,100]); d0 nearest the church.
_CANDS = [
    {"fid": "d0", "xy": [95.0, 98.0], "class": "car", "color": "white"},
    {"fid": "d1", "xy": [10.0, 10.0], "class": "car", "color": "white"},
    {"fid": "d2", "xy": [50.0, 50.0], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "church", "name": "St Church", "xy": [100.0, 100.0], "class": "building"},
]


def _run(ops):
    return run_program([dict(c) for c in _CANDS], [dict(a) for a in _ANCHORS], {"ops": ops})


def test_single_key_emphasis_sign_orders_without_compare():
    # score_by{emphasis:[{key:'distance to church', w:±1}]} with NO preceding compare —
    # the d_church column auto-attaches, and the emphasis SIGN is the
    # direction: w:-1 = nearest-first, w:+1 = farthest-first.
    asc = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])
    fids = [r["fid"] for r in asc["results"]["r1"]]
    assert fids[0] == "d0", fids  # nearest the church ranks first
    desc = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0", "emphasis": [{"key": "distance to church", "w": 1}]}},
    ])
    fids_desc = [r["fid"] for r in desc["results"]["r1"]]
    assert fids_desc[0] == "d1", fids_desc  # farthest first under w:+1
    assert fids_desc == fids[::-1], (fids, fids_desc)


def test_matches_explicit_compare_then_sort():
    # the auto-attach path is equivalent to writing the explicit compare+sort.
    auto = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])["results"]["r1"]
    explicit = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": {"items": "r0", "refs": ["church"]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])["results"]["r2"]
    assert [r["fid"] for r in auto] == [r["fid"] for r in explicit]
    # and the attached column value matches too
    assert auto[0]["distance to church"] == explicit[0]["distance to church"]


def test_filter_items_relation_column_without_compare():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0", "key": "distance to church", "op": "le", "value": 20.0}},
    ])
    fids = {r["fid"] for r in out["results"]["r1"]}
    assert fids == {"d0"}, fids  # only d0 within 20m of the church


def test_facing_auto_attaches_as_a_word():
    # 'facing <ref>' attaches the same way, as a compass WORD on every
    # off-contact row.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0", "key": "facing church", "op": "in_or_blank",
                  "value": ["north", "northeast", "east"]}},
    ])
    rows = out["results"]["r1"]
    assert {r["fid"] for r in rows} == {"d0", "d1", "d2"}
    assert all(r["facing church"] in ("north", "northeast", "east") for r in rows)
