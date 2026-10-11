from __future__ import annotations

from maps.search.analyze_scene.geom_program import compact_results, run_program

_CANDS = [
    {"fid": "d0", "xy": [12.0, 10.0], "class": "car", "color": "white"},
    {"fid": "d1", "xy": [80.0, 80.0], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "z_solo", "name": "Manhattan House", "xy": [0.0, 0.0], "class": "building"},
]


def _run(ops, cands=None, anchors=None):
    return run_program([dict(c) for c in (cands or _CANDS)],
                       [dict(a) for a in (anchors or _ANCHORS)], {"ops": ops})


def _cmp_headers(out, ref):
    res = out["results"][ref]
    return res.refs if hasattr(res, "refs") else None


# --------------------------------------------------------------------------- #
# find rows carry name_match (the NAME matched, not the instance)              #
# --------------------------------------------------------------------------- #


def test_query_find_rows_carry_name_match():
    # A fuzzy query find scores every matched row; that trust rides each row
    # as name_match (a NAME matched, never WHICH instance) -- not match_score,
    # not overloaded onto the resolved-set token 'score'.
    out = _run([
        {"ref": "r0", "op": "find",
         "args": {"query": "manhattan house", "provenance": "prior"}},
    ])
    rows = out["results"]["r0"]
    assert rows, rows
    for r in rows:
        assert "name_match" in r, r
        assert "match_score" not in r, r


# --------------------------------------------------------------------------- #
# render: the name_match column never renders (the attached value stays)       #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# compare header carries name_match (not match_score)                          #
# --------------------------------------------------------------------------- #


def test_unscored_ref_carries_no_name_match():
    out = _run([
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["z_solo"]}},
    ])
    hdrs = _cmp_headers(out, "r2")
    assert hdrs, hdrs
    assert all("name_match" not in h and "match_score" not in h
               for h in hdrs), hdrs


def test_compact_compare_header_carries_name_match():
    # The rename survives the compact render the model actually reads.
    out = _run([
        {"ref": "r0", "op": "find",
         "args": {"query": "manhattan house", "provenance": "prior"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ])
    comp = compact_results(out["results"])["r2"]
    keyset = {k for h in (comp.get("refs") or []) for k in h}
    assert "name_match" in keyset, keyset
    assert "match_score" not in keyset, keyset
