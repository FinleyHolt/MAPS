"""Anchor multiplicity: a compare ref that resolves to N same-named landmark
instances fans out into one per-instance column, reusing the target-candidate
column machinery. No anchor-specific branch — a multi-instance ref surfaces as
columns (never a single-fid collapse, never a duplicate-name census or
`ambiguous` header); anchors ride the same score_by discernment as targets.
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program, ProgramError

# Two target cars + one named landmark that has THREE same-named instances at
# distinct positions (a duplicate-named "Earl Street" building split across the
# block). The car sits next to instance road_a.
_CANDS = [
    {"fid": "d0", "xy": [12.0, 10.0], "class": "car", "color": "white"},
    {"fid": "d1", "xy": [80.0, 80.0], "class": "car", "color": "white"},
]
_ANCHORS = [
    {"fid": "b_a", "name": "Earl Street", "xy": [10.0, 10.0], "class": "building"},
    {"fid": "b_b", "name": "Earl Street", "xy": [90.0, 10.0], "class": "building"},
    {"fid": "b_c", "name": "Earl Street", "xy": [50.0, 90.0], "class": "building"},
    {"fid": "z_solo", "name": "Manhattan House", "xy": [0.0, 0.0], "class": "building"},
]


def _run(ops):
    return run_program([dict(c) for c in _CANDS],
                       [dict(a) for a in _ANCHORS], {"ops": ops})


def _cmp_headers(out, ref):
    res = out["results"][ref]
    return res.refs if hasattr(res, "refs") else None


def test_multi_instance_ref_fans_out_one_column_per_instance():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ])
    rows = out["results"]["r2"]
    # every candidate carries a proximity column to EACH of the 3 instances
    for row in rows:
        for fid in ("b_a", "b_b", "b_c"):
            assert f"distance to {fid}" in row, (list(row), fid)
    # no election / no cap: all three instances present as distinct columns
    keys = {k for row in rows for k in row if k.startswith("distance to b_")}
    assert keys == {"distance to b_a", "distance to b_b", "distance to b_c"}, keys


def test_model_resolves_the_instance_by_emphasis():
    # d0 sits by instance b_a. The model disambiguates exactly as it does
    # across target candidates: its emphasis names the instance column the
    # cue pins (score_by fuses exactly the named keys). Emphasis on d_b_a
    # floats d0 to rank-0. No bind, no election.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
        {"ref": "r3", "op": "score_by",
         "args": {"items": "$r2", "emphasis": [{"key": "distance to b_a", "w": -5}]}},
    ])
    term = list(out["results"])[-1]
    fids = [r["fid"] for r in out["results"][term]]
    assert fids[0] == "d0", fids


def test_explicit_single_index_into_same_named_browse_raises_the_law():
    # r0[0] into an UNORDERED browse holding THREE same-named 'Earl Street'
    # instances does not silently resolve to the first — a name is not a
    # place, so the position picks a tied same-named instance arbitrarily.
    # The guard fires loud and names the resolution path (score the set
    # against the cue's OTHER refs, then index). A query-matched find is
    # ORDERED and indexes freely (its own tie note stated the
    # arbitrariness); the fan-out onto per-instance columns (the bare-ref
    # path) keeps every instance.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
            {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0[0]"]}},
        ])
    except ProgramError as e:
        msg = str(e).lower()
        assert "name" in msg and ("resolve which" in msg
                                  or "not a place" in msg), str(e)
    else:
        raise AssertionError(
            "indexing an unresolved same-named set must raise the tie law")


def test_bare_fid_ref_single_column_unchanged():
    out = _run([
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["b_b"]}},
    ])
    rows = out["results"]["r2"]
    d_cols = {k for row in rows for k in row if k.startswith("distance to b_")}
    assert d_cols == {"distance to b_b"}, d_cols


def test_multi_instance_header_is_plain_columns():
    # A multi-instance ref surfaces as plain per-instance columns, not a warning:
    # no duplicate-name census, no `ambiguous` header.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ])
    res = out["results"]["r2"]
    hdrs = res.headers if hasattr(res, "headers") else []
    for h in hdrs:
        assert "ambiguous" not in h, h
    for row in out["results"]["r2"]:
        assert not any("ambiguous" in str(k).lower() for k in row)


def test_no_cap_fans_out_past_eight():
    # No instance-count cap: a ref resolving to many same-named instances (a long
    # fragmented road) fans out ONE column per instance, no fail-loud -- the
    # disambiguation signal is preserved at any count (bounded only by the token
    # net, which 9 x 2 cells is nowhere near).
    many = [{"fid": "m%d" % i, "name": "Long Road", "xy": [float(i * 20), 0.0],
             "class": "road"} for i in range(9)]
    out = run_program([dict(c) for c in _CANDS], many, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Long Road"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ]})
    rows = out["results"]["r2"]
    d_cols = {k for row in rows for k in row if k.startswith("distance to m")}
    assert len(d_cols) == 9, d_cols  # one column per instance, no cap


def test_token_overflow_guard_fires_loud():
    # The ONLY bound: a fan-out whose rendered size (instances x candidates) would
    # be a crazy amount of tokens fails loud and STEERS. 90 x 90 = 8100 cells > the
    # ~8000 net; genuine landmark fan-outs sit far below it.
    cands = [{"fid": "c%d" % i, "xy": [float(i), 1.0], "class": "car"}
             for i in range(90)]
    anchs = [{"fid": "a%d" % i, "name": "Grid Rd", "xy": [float(i), 0.0],
              "class": "road"} for i in range(90)]
    try:
        run_program(cands, anchs, {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Grid Rd"}},
            {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
        ]})
    except ProgramError as e:
        assert "narrow" in str(e).lower() and "cells" in str(e).lower(), str(e)
    else:
        raise AssertionError("expected the token-overflow guard to fire loud")


def test_self_compare_trips_guard_with_items_steer():
    # refs = the candidate set (comparing items against themselves) trips the net
    # once large, and the steer names the fix: pass them as items, not refs.
    cands = [{"fid": "c%d" % i, "xy": [float(i), 1.0], "class": "car"}
             for i in range(90)]
    try:
        run_program(cands, [], {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": ["$r0"]}},
        ]})
    except ProgramError as e:
        assert "items" in str(e).lower(), str(e)
    else:
        raise AssertionError("expected the self-compare to trip the token guard")


def test_single_instance_column_sort_via_muted_emphasis():
    # a w:0 entry is a legal no-pull term: the ordering reads the one signed
    # column, exactly (score_by fuses exactly the named keys at their weights).
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
        {"ref": "r3", "op": "score_by",
         "args": {"items": "$r2", "emphasis": [{"key": "distance to b_a", "w": -1},
                                               {"key": "distance to b_b", "w": 0},
                                               {"key": "distance to b_c", "w": 0}]}},
    ])
    term = list(out["results"])[-1]
    fids = [r["fid"] for r in out["results"][term]]
    assert fids[0] == "d0", fids  # nearest to instance b_a


def test_find_query_grounded_anchor_carries_name_match():
    # A landmark grounded via find{query} IS a query-find token, so its
    # grounding-trust score rides the compare header under name_match — the
    # NAME matched, not (necessarily) the instance. With find as THE one
    # retrieval op, every named-landmark grounding is fuzzy-SCORED — a weak
    # name hit is never invisible. (The retired key was match_score.)
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
    assert all("match_score" not in h for h in hdrs), hdrs

    # An UNSCORED ref (a bare literal fid — nothing grounded it by query)
    # carries no trust column: name_match keys on query-find provenance.
    out2 = _run([
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["z_solo"]}},
    ])
    hdrs2 = _cmp_headers(out2, "r2")
    assert hdrs2, hdrs2
    assert all("name_match" not in h and "match_score" not in h
               for h in hdrs2), hdrs2


def test_fanout_census_past_knee_is_render_only():
    # 7 same-named instances (past _ANCHOR_FANOUT_CENSUS_INSTANCES): the
    # DEFAULT grid swaps the per-instance block for a census + near_/by_/
    # spread_ cells per candidate. The RAW rows — the $rN every chained op
    # and the commit read consume — keep every per-instance column
    # byte-identical; nothing is dropped from the set.
    import copy

    from maps.search.analyze_scene.geom_program import compact_results

    anchs = [{"fid": "e%d" % i, "name": "Elm Street",
              "xy": [float(10 * i), 0.0], "class": "building"}
             for i in range(7)]
    out = run_program([dict(c) for c in _CANDS], anchs, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Elm Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ]})
    rows = out["results"]["r2"]
    d_cols = {k for row in rows for k in row
              if k.startswith("distance to ")}
    assert len(d_cols) == 7, d_cols  # whole-set fan-out, uncapped
    raw_before = copy.deepcopy([dict(r) for r in rows])

    rendered = compact_results(out["results"])
    head = rendered["r2"]
    census = head.get("anchor_census")
    assert census and census[0]["n_instances"] == 7, head.keys()
    assert set(census[0]["instance_fids"]) == {"e%d" % i for i in range(7)}
    # the grid renders the reduced cells, not the wide per-instance block
    grid = str(head.get("legend")) + str(head.get("rows"))
    assert "near_elm_street" in grid and "by_elm_street" in grid \
        and "spread_elm_street" in grid, grid
    assert not any(c in grid for c in d_cols), grid
    # the columns ECHO still names every real per-instance column (live on $rN)
    assert d_cols <= set(head.get("columns") or []), head.get("columns")
    # render never mutated the raw rows
    assert [dict(r) for r in rows] == raw_before

    # BELOW the knee (3 instances) the grid is unchanged — no census block.
    out2 = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Earl Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ])
    head2 = compact_results(out2["results"])["r2"]
    assert "anchor_census" not in head2, head2.keys()
    assert "distance to b_a" in (str(head2.get("legend")) + str(head2.get("rows"))), head2


def test_fanout_census_colliding_names_keep_distinct_cells():
    # Two DISTINCT ref names that sanitize to the same cell key ('Elm Street'
    # vs 'elm-street'), both past the knee in ONE compare: each group keeps
    # its own near_/by_/spread_ cells (counter-suffixed, mirroring the header
    # label collision rule) and its census entry names its cell_suffix — no
    # silent overwrite of one group's summary by the other.
    from maps.search.analyze_scene.geom_program import compact_results

    anchs = (
        [{"fid": "a%d" % i, "name": "Elm Street",
          "xy": [float(10 * i), 0.0], "class": "building"} for i in range(7)]
        + [{"fid": "b%d" % i, "name": "elm-street",
            "xy": [float(10 * i), 100.0], "class": "building"} for i in range(7)]
    )
    out = run_program([dict(c) for c in _CANDS], anchs, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "elm"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["$r0"]}},
    ]})
    head = compact_results(out["results"])["r2"]
    census = head.get("anchor_census") or []
    assert len(census) == 2, census
    suffixes = [c["cell_suffix"] for c in census]
    assert len(set(suffixes)) == 2, suffixes
    grid = str(head.get("legend")) + str(head.get("rows"))
    for sfx in suffixes:
        assert f"near_{sfx}" in grid and f"by_{sfx}" in grid, (sfx, grid)


def test_prior_search_text_folds_tags_and_walls_descriptions():
    # The descriptions wall, CI-pinned at the search-text builder: a prior
    # node with real OSM dict tags (the live MapObject shape) builds its
    # find-searchable free text from name + ALL tag KEYS and VALUES — and a
    # 'description'/'descriptions' tag NEVER feeds it (defense-in-depth for
    # "ground-truth labels stay out of the agent loop").
    from maps.search.analyze_scene.geom_program import _anchor_search_text

    class _Prior:
        name = "Earl Street"
        tags = {
            "highway": "residential",
            "description": "THE GT SENTENCE THAT MUST NEVER LEAK",
            "descriptions": "ALSO WALLED",
        }

    text = _anchor_search_text(_Prior())
    assert "earl street" in text, text
    assert "highway" in text, text      # tag KEYS fold in
    assert "residential" in text, text  # tag VALUES fold in
    assert "leak" not in text and "walled" not in text, text
