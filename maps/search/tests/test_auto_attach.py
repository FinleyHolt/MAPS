"""Auto-attach relation columns: the model can reference a relation column
(d_/brg_gap_/brg_/dc_<ref>) in filter_items or a score_by emphasis WITHOUT
first running an explicit compare — the executor attaches it by reusing the
same compare op. No compare-first ceremony; the absent-key guard fires only
for a genuinely unresolvable ref. (The along_/off_ chart members of the
lazy family are pinned in test_order_freedom.py.)
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program, ProgramError

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
         "args": {"items": "$r0", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])
    fids = [r["fid"] for r in asc["results"]["r1"]]
    assert fids[0] == "d0", fids  # nearest the church ranks first
    desc = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0", "emphasis": [{"key": "distance to church", "w": 1}]}},
    ])
    fids_desc = [r["fid"] for r in desc["results"]["r1"]]
    assert fids_desc[0] == "d1", fids_desc  # farthest first under w:+1
    assert fids_desc == fids[::-1], (fids, fids_desc)


def test_matches_explicit_compare_then_sort():
    # the auto-attach path is equivalent to writing the explicit compare+sort.
    auto = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])["results"]["r1"]
    explicit = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": ["church"]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "$r1", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])["results"]["r2"]
    assert [r["fid"] for r in auto] == [r["fid"] for r in explicit]
    # and the attached column value matches too
    assert auto[0]["distance to church"] == explicit[0]["distance to church"]


def test_filter_items_relation_column_without_compare():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "$r0", "key": "distance to church", "op": "le", "value": 20.0}},
    ])
    fids = {r["fid"] for r in out["results"]["r1"]}
    assert fids == {"d0"}, fids  # only d0 within 20m of the church


def test_score_by_emphasis_relation_column_without_compare():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])
    fids = [r["fid"] for r in out["results"]["r1"]]
    assert fids[0] == "d0", fids


def test_already_attached_is_byte_identical():
    # when the model DID compare first, auto-attach must be a no-op (short-circuit
    # on the already-present column) — the explicit program is unchanged.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": ["church"]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "$r1", "emphasis": [{"key": "distance to church", "w": -1}]}},
    ])
    assert [r["fid"] for r in out["results"]["r2"]][0] == "d0"


def test_dc_and_brg_gap_auto_attach():
    # the centroid-distance and min-gap-bearing columns auto-attach the same
    # way d_ does (brg_gap_ strips its own longer prefix, never brg_'s).
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0", "emphasis": [{"key": "center distance to church", "w": -1}]}},
    ])
    fids = [r["fid"] for r in out["results"]["r1"]]
    assert fids[0] == "d0", fids
    # 'facing <ref>' auto-attaches the same way, and it is a WORD, so it
    # answers truthy/eq and REFUSES the ordering ops -- a compass word has no
    # >=. That refusal is the column doing its job: ranking rows by a compass
    # reading sorts a circle by where you cut it.
    out2 = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "$r0", "key": "facing church", "op": "truthy"}},
    ])
    assert len(out2["results"]["r1"]) == 3  # attached on every off-contact row
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "$r0", "key": "facing church",
                      "op": "ge", "value": 0.0}},
        ])
    except ProgramError as e:
        assert "SCALAR key" in str(e), str(e)
    else:
        raise AssertionError("a compass word must refuse an ordering op")


def test_unresolvable_ref_still_errors():
    # the absent-key guard survives for a genuinely-unresolvable / guessed key:
    # 'ghost' is not a fid on the surface, so nothing attaches and the loud error
    # (the model typo'd) still fires.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "score_by",
             "args": {"items": "$r0", "emphasis": [{"key": "distance to ghost", "w": -1}]}},
        ])
    except ProgramError as e:
        assert "distance to ghost" in str(e)
    else:
        raise AssertionError("expected ProgramError for an unresolvable ref")
