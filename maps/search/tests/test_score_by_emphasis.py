"""AGENT-WEIGHTED ordering — the ``score_by{items, emphasis}`` contract."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program, ProgramError
from maps.search.analyze_scene import geom_compute as G
from maps.search.parse_compile.reasoner_loop import (
    run_ops,
)

# Four cars at distinct positions; three white (color_white high), one gray. A
# compare against two point refs — the bare literal attaches the documented
# _xy family (d_xy/brg_xy/dc_xy); the labeled {ref, as:'n'} form is the
# named-columns path (d_n/brg_n/dc_n; a second BARE literal would be the
# loud ambiguity error).
_CANDS = [
    {"fid": "d0", "xy": [0.0, 0.0], "class": "car", "color": "white"},
    {"fid": "d1", "xy": [90.0, 5.0], "class": "car", "color": "white"},
    {"fid": "d2", "xy": [5.0, 90.0], "class": "car", "color": "gray"},
    {"fid": "d3", "xy": [50.0, 50.0], "class": "car", "color": "white"},
]
_BASE = [
    {"ref": "r0", "op": "find", "args": {}},
    {"ref": "r1", "op": "compare",
     "args": {"items": "r0",
              "refs": [[100.0, 0.0], {"ref": [0.0, 100.0], "as": "n"}]}},
]


def _score_by(emphasis=None):
    op = {"ref": "r2", "op": "score_by", "args": {"items": "r1"}}
    if emphasis is not None:
        op["args"]["emphasis"] = emphasis
    return {"ops": _BASE + [op]}


def _run_fids(prog):
    out = run_program([dict(c) for c in _CANDS], [], prog)
    term = list(out["results"])[-1]
    return [r["fid"] for r in out["results"][term]], out


# --- EXACT-KEYS: the sort fuses exactly what emphasis names ----------------


def test_orders_by_exactly_the_named_keys():
    emphasis = [{"key": "distance to xy", "w": -1}, {"key": "color_white", "w": 1}]
    fids, out = _run_fids(_score_by(emphasis))
    rows_r1 = out["results"]["r1"]
    expected = [r["fid"] for r in G.score_by(
        rows_r1, ["color_white", "distance to xy"], [1.0, -1.0])]
    assert fids == expected, (fids, expected)


def test_unnamed_present_column_contributes_nothing():
    # d_n is PRESENT on every row but NOT named: the one-key d_xy order
    # must equal G.score_by over d_xy alone -- and NAMING d_n changes it.
    one, out = _run_fids(_score_by([{"key": "distance to xy", "w": -1}]))
    rows_r1 = out["results"]["r1"]
    assert one == [r["fid"] for r in G.score_by(rows_r1, ["distance to xy"], [-1.0])]
    both, _ = _run_fids(_score_by([{"key": "distance to xy", "w": -1},
                                   {"key": "distance to n", "w": -1}]))
    assert both != one, (both, one)


# --- SIGN-IS-THE-AGENT'S ----------------------------------------------------


def test_emphasis_sign_controls_order():
    pos = _run_fids(_score_by([{"key": "bearing from xy", "w": 1}]))[0]
    neg = _run_fids(_score_by([{"key": "bearing from xy", "w": -1}]))[0]
    assert pos != neg, (pos, neg)
    # the agent owns the sign: the winner under +1 is the LOSER under -1.
    assert pos[0] != neg[0] and pos[0] == neg[-1], (pos, neg)


# --- NO-TERMINAL-REORDER: the program's own order is the surfaced order -----


def test_surfaced_order_is_the_programs_own_order():
    ns = {"cands": [dict(c) for c in _CANDS], "anchors": []}
    prog = _score_by([{"key": "bearing from xy", "w": 1}])   # agent's own weighted order
    out = run_ops(prog["ops"], namespace=ns)
    assert "error" not in out
    # the surfaced order is the AGENT's brg-weighted order verbatim -- the
    # harness applies no composite reorder to any terminal.
    agent_order = _run_fids(prog)[0]
    assert out["results"]["r2"]["fids"] == agent_order


def test_compare_keeps_input_order():
    # a compare attaches columns and never sorts, even though d_ columns are
    # on the rows.
    out = run_program([dict(c) for c in _CANDS], [], {"ops": _BASE})
    assert [r["fid"] for r in out["results"]["r1"]] == ["d0", "d1", "d2", "d3"]


# --- NEVER-TOUCHES-COMMIT: permutation only, no fid added/dropped/mutated ----


def test_reorder_is_a_commit_safe_permutation():
    ns = {"cands": [dict(c) for c in _CANDS], "anchors": []}
    prog = _score_by([{"key": "distance to xy", "w": -5}])
    out = run_ops(prog["ops"], namespace=ns)
    surfaced = out["results"]["r2"]["fids"]
    # PERMUTATION of the input candidate fids: none invented, none lost.
    assert sorted(surfaced) == sorted(c["fid"] for c in _CANDS)


# --- FAIL-LOUD: actionable errors, never a silent constant / no-op ----------


def test_emphasis_key_no_row_carries_fails_loud():
    try:
        _run_fids(_score_by([{"key": "distance to nope", "w": -1}]))
    except ProgramError as e:
        assert "nothing orders by 'distance to nope'" in str(e), str(e)
    else:
        raise AssertionError("expected ProgramError on an absent emphasis key")
