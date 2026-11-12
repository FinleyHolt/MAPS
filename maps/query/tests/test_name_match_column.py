from __future__ import annotations

from maps.query.program import run_program

_CANDS = [
    {"fid": "d0", "xy": [12.0, 10.0], "class": "car", "color": "white"},
    {"fid": "d1", "xy": [80.0, 80.0], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "z_solo", "name": "Manhattan House", "xy": [0.0, 0.0], "class": "building"},
]


def _run(ops, cands=None, anchors=None):
    return run_program(
        [dict(c) for c in (cands or _CANDS)], [dict(a) for a in (anchors or _ANCHORS)], {"ops": ops}
    )


# --------------------------------------------------------------------------- #
# find rows carry name_match (the NAME matched, not the instance)              #
# --------------------------------------------------------------------------- #


def test_query_find_rows_carry_name_match():
    # A fuzzy query find scores every matched row; that trust rides each row
    # as name_match (a NAME matched, never WHICH instance) -- not match_score,
    # not overloaded onto the resolved-set token 'score'.
    out = _run(
        [
            {"ref": "r0", "op": "find", "args": {"query": "manhattan house"}},
        ]
    )
    rows = out["results"]["r0"]
    assert rows, rows
    for r in rows:
        assert "name_match" in r, r
        assert "match_score" not in r, r
