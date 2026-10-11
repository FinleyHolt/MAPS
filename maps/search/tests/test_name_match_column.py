"""The name_match key: a query-find's fuzzy grounding trust survives as the
RAW row field and the compare-header grounding-trust key (never
``match_score``) -- the row COLUMN no longer renders
(7/8 not-observed, zero
reads-to-pick-a-row, anti-correlated on 2 committed rows).

Contract: the token ``score`` that a *resolved* set carries must not be the
same token a NAME match carries. The raw-field and compare-header key is
``name_match``; the string ``match_score`` appears NOWHERE on any served
surface (find rows, compare headers, or the tool text).
"""
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


def test_name_match_never_renders_as_a_grid_column():
    # Audit: 7/8 not-observed, zero
    # reads-to-pick-a-row in 8/8 packs -- the column is a render cut in
    # _GRID_COL_DROP, and the elide-constant special case died with it.
    # The attached value stays on every raw row (the ordering reads it).
    out = _run([
        {"ref": "r0", "op": "find",
         "args": {"query": "car", "provenance": "runtime"}},
    ], cands=[dict(c, search_text=c["class"]) for c in _CANDS])
    raw = out["results"]["r0"]
    assert all(r.get("name_match") == raw[0]["name_match"] for r in raw), raw
    env = compact_results(out["results"])["r0"]
    assert "name_match" not in env["rows"].splitlines()[0], env["rows"]
    # DISTINCT values render no column either: the cut is unconditional.
    out2 = _run([
        {"ref": "r0", "op": "find",
         "args": {"query": "manhattan house OR earl", "provenance": "prior"}},
    ], anchors=_ANCHORS + [{"fid": "z_earl", "name": "Earl Court",
                            "xy": [30.0, 0.0], "class": "building"}])
    raw2 = out2["results"]["r0"]
    if len({r.get("name_match") for r in raw2}) > 1:
        env2 = compact_results(out2["results"])["r0"]
        assert "name_match" not in env2["rows"].splitlines()[0], env2["rows"]


# --------------------------------------------------------------------------- #
# compare header carries name_match (not match_score)                          #
# --------------------------------------------------------------------------- #


def test_compare_header_carries_name_match():
    # A landmark grounded via find{query} carries its grounding-trust on the
    # compare header under the name_match key.
    out = _run([
        {"ref": "r0", "op": "find",
         "args": {"query": "manhattan house", "provenance": "prior"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ])
    hdrs = _cmp_headers(out, "r2")
    assert hdrs, hdrs
    assert any(h.get("ref_fid") == "z_solo" and "name_match" in h
               for h in hdrs), hdrs
    # the retired key is gone from every header
    assert all("match_score" not in h for h in hdrs), hdrs


def test_unscored_ref_carries_no_name_match():
    # A bare literal fid (nothing grounded it by query) carries no trust
    # column -- name_match keys on query-find provenance, same as the old
    # match_score did.
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
