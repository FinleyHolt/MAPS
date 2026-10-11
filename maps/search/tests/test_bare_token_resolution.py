"""A bare token in an op arg resolves to the handle or fid it names."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import ProgramError, run_program

_CANDS = [
    {"fid": "d0", "xy": [1.0, 1.0], "class": "car"},
    {"fid": "d1", "xy": [9.0, 9.0], "class": "car"},
]
_ANCHORS = [
    {"fid": "b_a", "name": "Earl Street", "xy": [10.0, 10.0], "class": "building"},
]


def _run(ops, cands=None, anchors=None, seed=None):
    return run_program([dict(c) for c in (cands or _CANDS)],
                       [dict(a) for a in (anchors or _ANCHORS)],
                       {"ops": ops}, seed_results=seed)


def _fids(out, ref):
    return [r["fid"] for r in out["results"][ref]]


def _keep(ref, items, value="car"):
    # an items-consuming op (a class-eq filter that keeps everything of that
    # class) -- exercises the ONE bare-token items resolution path.
    return {"ref": ref, "op": "filter_items",
            "args": {"items": items, "key": "class", "op": "eq", "value": value}}


# --------------------------------------------------------------------------- #
# bare rN resolves as a prior-op handle                                        #
# --------------------------------------------------------------------------- #


def test_bare_rn_resolves_in_items():
    # items:'r0' (no sigil) resolves to the prior find's rows.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        _keep("r1", "r0"),
    ])
    assert set(_fids(out, "r1")) == {"d0", "d1"}


def test_dollar_rn_still_forwards():
    # $r0 is a forwarding slip for r0 -- SAME resolution, one meaning.
    out_bare = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        _keep("r1", "r0"),
    ])
    out_sig = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        _keep("r1", "$r0"),
    ])
    assert _fids(out_bare, "r1") == _fids(out_sig, "r1")


def test_bare_last_resolves_to_previous_result():
    # 'last' (bare reserved word) resolves to the immediately-preceding result.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        _keep("r1", "last"),
    ])
    assert set(_fids(out, "r1")) == {"d0", "d1"}
    # and the $-form is the same slip
    out2 = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        _keep("r1", "$last"),
    ])
    assert _fids(out, "r1") == _fids(out2, "r1")


def test_bare_cands_and_anchors_resolve():
    # 'cands' / 'anchors' (bare reserved surface words) resolve to the two
    # injected surface lists.
    out = _run([
        _keep("r0", "cands"),
        _keep("r1", "anchors", value="building"),
    ])
    assert set(_fids(out, "r0")) == {"d0", "d1"}
    assert set(_fids(out, "r1")) == {"b_a"}
    # $-form slips resolve identically
    out2 = _run([
        _keep("r0", "$cands"),
        _keep("r1", "$anchors", value="building"),
    ])
    assert _fids(out, "r0") == _fids(out2, "r0")
    assert _fids(out, "r1") == _fids(out2, "r1")


def test_bare_rn_as_compare_items_and_refs():
    # a bare rN in BOTH items and a refs element resolves.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "r1", "refs": ["r0"]}},
    ])
    rows = out["results"]["r2"]
    assert {r["fid"] for r in rows} == {"d0", "d1"}
    assert all("distance to b_a" in r for r in rows), rows


# --------------------------------------------------------------------------- #
# top-to-bottom stale-read guard STILL fires on the bare form                  #
# --------------------------------------------------------------------------- #


def test_forward_ref_order_free_in_program_def_wins():
    # Order-free: reading a name this program defines LATER (bare 'r2' or the
    # $-slip '$r2') resolves to the IN-PROGRAM def -- find r2 (runtime = d0, d1)
    # runs first and r0 filters THAT, never the same-named seed [d1].
    seed = {"r2": [{"fid": "d1", "xy": [9.0, 9.0], "class": "car"}]}
    for tok in ("r2", "$r2"):
        out = _run([
            _keep("r0", tok),
            {"ref": "r2", "op": "find",
             "args": {"provenance": "runtime"}},
        ], seed=seed)
        assert _fids(out, "r0") == ["d0", "d1"], (tok, _fids(out, "r0"))


def test_bare_prior_call_ref_resolves_from_seed_when_unshadowed():
    # A bare prior-call ref with NO in-program shadow resolves from the seed
    # (cross-call persistence, one refresh idiom) -- same as the $-form.
    seed = {"r2": [{"fid": "d1", "xy": [9.0, 9.0], "class": "car"}]}
    out = _run([
        _keep("r5", "r2"),
    ], seed=seed)
    assert _fids(out, "r5") == ["d1"]


def test_bare_unknown_ref_fails_loud():
    # A bare token that is neither a known ref/reserved word NOR a fid on the
    # surface fails loud (it is not silently treated as a missing fid).
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            _keep("r1", "r9"),
        ])
    except ProgramError as e:
        assert "r9" in str(e), str(e)
    else:
        raise AssertionError("an unknown bare ref must fail loud")
