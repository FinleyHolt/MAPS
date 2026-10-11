"""The same-named-tie index guard: a NAME is not a PLACE.

Indexing ``[0]`` into an UNORDERED set (a query-less browse, the raw
surface) that holds several instances all sharing the same name does NOT
resolve WHICH instance the cue means -- the position picks a tied same-named
row arbitrarily. The guard fires loud and names the law: resolve the
instance by the cue's OTHER refs (compare against them + score_by, or
filter_items to one), THEN index the ordered result.

An ORDERED result silences the guard: score_by output (cue_sorted) and a
query-matched find (best-first is real; its own tie note already stated any
within-tie arbitrariness) index freely. A genuinely-resolved head is not a
tie either, so ``[0]`` on it is always legal.
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import ProgramError, run_program

_CANDS = [
    {"fid": "d0", "xy": [12.0, 10.0], "class": "car"},
    {"fid": "d1", "xy": [80.0, 80.0], "class": "car"},
]
# THREE prior instances all named "Earl Street" -- a name that matches several
# instances is a whole hypothesis set, not one place.
_SAME_NAMED = [
    {"fid": "b_a", "name": "Earl Street", "xy": [10.0, 10.0], "class": "building"},
    {"fid": "b_b", "name": "Earl Street", "xy": [90.0, 10.0], "class": "building"},
    {"fid": "b_c", "name": "Earl Street", "xy": [50.0, 90.0], "class": "building"},
]
# Three DISTINCT-named priors -- no tie, so [0] is always a real resolution.
_DISTINCT = [
    {"fid": "b_a", "name": "Earl Street", "xy": [10.0, 10.0], "class": "building"},
    {"fid": "b_b", "name": "King Road", "xy": [90.0, 10.0], "class": "building"},
    {"fid": "b_c", "name": "Queen Ave", "xy": [50.0, 90.0], "class": "building"},
]


def _run(ops, anchors):
    return run_program([dict(c) for c in _CANDS],
                       [dict(a) for a in anchors], {"ops": ops})


def _assert_raises_law(ops, anchors):
    try:
        _run(ops, anchors)
    except ProgramError as e:
        msg = str(e).lower()
        # the law: a name is not a place; resolve which instance by its other
        # refs (score the set), then index the ordered result.
        assert "name" in msg and ("resolve which" in msg
                                  or "not a place" in msg), str(e)
        return str(e)
    raise AssertionError("expected the same-named-tie index guard to fire loud")


# --------------------------------------------------------------------------- #
# r0[0] over 3 same-named priors from an UNORDERED browse -> the law fires    #
# --------------------------------------------------------------------------- #


def test_bare_index_into_same_named_browse_raises_law_as_frame_reference():
    # r0[0] as a compare frame.reference (the single-geometry slot).
    _assert_raises_law([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "r0[0]", "side": "left"}}},
    ], _SAME_NAMED)


def test_bare_index_into_same_named_browse_raises_law_as_items():
    # r0[0] as a list-op items slot (the single-row wrap path).
    _assert_raises_law([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0[0]", "key": "class", "op": "eq", "value": "building"}},
    ], _SAME_NAMED)


def test_bare_index_into_same_named_browse_raises_law_as_viewer_endpoint():
    # r0[0]/r1[0] as the viewer frame's from/toward endpoints.
    _assert_raises_law([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r2", "op": "compare",
         "args": {"items": "cands", "frame": {"from": "r0[0]", "toward": "r1[0]"}}},
    ], _SAME_NAMED)


def test_dollar_form_index_into_same_named_browse_also_raises_law():
    # the $-slip form hits the SAME guard -- one meaning.
    _assert_raises_law([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0[0]"]}},
    ], _SAME_NAMED)


def test_query_matched_find_tied_head_is_legal_to_index():
    # A query-matched find is ORDERED (best-first is real): its result
    # already stated the within-tie arbitrariness in its own tie note, so
    # indexing it is the agent's informed pick, not a refused turn.
    out = _run([
        {"ref": "r0", "op": "find",
         "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "r0[0]", "side": "left"}}},
    ], _SAME_NAMED)
    assert "rV" in out["results"]


# --------------------------------------------------------------------------- #
# a resolved head -> [0] is legal (the guard fires ONLY on a genuine tie)      #
# --------------------------------------------------------------------------- #


def test_score_by_resolved_head_is_legal_to_index():
    # Resolve the same-named set against the cue's OTHER ref (a point the cue
    # pins) with compare + score_by. score_by gives the instances DISTINCT
    # scores, so the head is no longer a same-named TIE -- [0] on the scored
    # set is legal (the guard fires ONLY on a genuine tie).
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "compare", "args": {"items": "r0", "refs": [[10.0, 10.0]]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1", "emphasis": [{"key": "distance to xy", "w": -1}]}},
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "r2[0]", "side": "left"}}},
    ], _SAME_NAMED)
    # the program completed -- the scored-head [0] did not trip the guard.
    assert "rV" in out["results"]


def test_indexed_head_narrowed_to_one_is_legal_to_index():
    # The narrow-to-one variant: the scored head rN[0] fed as items collapses
    # to a single-row register, and [0] on that single row is the row itself
    # -- also legal.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "compare", "args": {"items": "r0", "refs": [[10.0, 10.0]]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1", "emphasis": [{"key": "distance to xy", "w": -1}]}},
        {"ref": "r3", "op": "filter_items",
         "args": {"items": "r2[0]", "key": "class", "op": "eq", "value": "building"}},
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "r3[0]", "side": "left"}}},
    ], _SAME_NAMED)
    assert "rV" in out["results"]


def test_distinct_named_set_index_is_legal():
    # A set whose rows carry DISTINCT names has no same-named tie, so [0]
    # resolves a real instance -- the guard never fires (even on an
    # unordered retrieval).
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "r0[0]", "side": "left"}}},
    ], _DISTINCT)
    # the program completed -- the distinct-named [0] did not trip the guard.
    assert "rV" in out["results"]


def test_filter_narrowed_single_row_index_is_legal():
    # filter_items narrows a same-named set to ONE row; [0] on that single
    # row is the row itself (not an ambiguous position over >=2 tied rows)
    # -- legal.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0", "key": "fid", "op": "eq", "value": "b_b"}},
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "r1[0]", "side": "left"}}},
    ], _SAME_NAMED)
    assert "rV" in out["results"]


def test_direct_surface_index_on_tied_head_still_raises():
    # Retrieval/surface order carries no cue signal: indexing anchors[...]
    # straight into a same-named tied head still raises the law.
    _assert_raises_law([
        {"ref": "rV", "op": "compare",
         "args": {"items": "cands", "frame": {"reference": "anchors[0]", "side": "left"}}},
    ], _SAME_NAMED)
