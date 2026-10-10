"""Argument-shape coercion (do-what-the-model-means, same family as the
fan-out / auto-attach): a refs-less compare that carries the frame aspect is
legal (it computes its own columns); as_points flattens a $rN nested in a
list literal and wraps a single resolved row into a one-item list. The
frame-compare surface itself is pinned in ``test_frame_aspect.py``.
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program, ProgramError

_CANDS = [  # spaced 10m along x so the frame's in-row flank columns attach
    {"fid": "d0", "xy": [30.0, 50.0], "class": "car"},
    {"fid": "a1", "xy": [20.0, 50.0], "class": "car"},
    {"fid": "b1", "xy": [40.0, 50.0], "class": "car"},
]
_ANCH = [
    {"fid": "road_1", "name": "Main Street", "xy": [50.0, 0.0], "class": "road",
     "centerline": [[0.0, 0.0], [100.0, 0.0]]},
]


def _run(ops):
    return run_program([dict(c) for c in _CANDS], [dict(a) for a in _ANCH], {"ops": ops})


def _has_prefix(rows, pre):
    return any(isinstance(r, dict) and any(str(k).startswith(pre) for k in r) for r in rows)


def test_compare_frame_only_no_refs():
    # refs:[] with the frame aspect computes its columns with no refs. A
    # road reference is the polyline CHART: along_/off_ (no side).
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0", "refs": [],
                  "frame": {"along": "road_1"}}},
    ])
    rows = out["results"]["r1"]
    assert _has_prefix(rows, "along road_1"), list(rows[0])
    assert all("distance to road_1" in r for r in rows), list(rows[0])


def test_compare_frame_only_refs_key_omitted():
    # refs KEY entirely absent (not just refs:[]) + the frame aspect -> legal:
    # the executor does not require the refs key when the frame computes.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0",
                  "frame": {"along": "road_1"}}},
    ])
    assert _has_prefix(out["results"]["r1"], "distance to road_1")


def test_frame_viewer_between_read():
    # frame:{from:A, toward:B} (bare fids): 'between' is a column read --
    # |vx_| small (lateral offset from the look ray) AND vy_ within the span.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0", "frame": {"from": "a1", "toward": "b1"}}},
    ])
    rows = out["results"]["r1"]
    assert any(("left of a1_b1" in r) or ("right of a1_b1" in r)
               for r in rows), list(rows[0])
    mid = next(r for r in rows if r["fid"] == "d0")  # midway between a1 and b1
    assert mid.get("left of a1_b1", mid.get("right of a1_b1", 9.9)) < 0.5, mid
    assert 0.0 <= mid["ahead of a1_b1"] <= 20.0, mid  # span 20



def test_compare_truly_empty_still_errors():
    # NEITHER refs NOR the frame aspect -> genuinely nothing to compute.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": []}},
        ])
    except ProgramError:
        pass
    else:
        raise AssertionError("a refs/frame-less compare must still error")


def test_walk_path_slice_windows_a_list():
    # $r0[:2] / $r0[1:] windows a prior list (agentic top-N windowing) instead of
    # being parsed as a bogus field '.: 2'.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "$r0[:2]", "key": "class", "op": "eq", "value": "car"}},
    ])
    assert len(out["results"]["r1"]) == 2


def test_walk_path_index_by_fid_is_clean_error_not_crash():
    # $rN[car_1] -- the model indexed by a FID, not a position. The slice regex
    # must not int() a fid string; a fid index is a clean ProgramError.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "$r0[b1]", "key": "class", "op": "eq", "value": "car"}},
        ])
    except ProgramError as e:
        assert "index" in str(e).lower()
    except Exception as e:  # noqa: BLE001
        raise AssertionError("fid-index must be ProgramError, not %s" % type(e).__name__)


def test_as_list_single_row_wrapped():
    # $r0[0] (index to one row) fed to a list-op wraps into a 1-item list, not
    # "cannot resolve a LIST from '$r0[0]'".
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "$r0[0]", "key": "class", "op": "eq", "value": "car"}},
    ])
    assert len(out["results"]["r1"]) == 1


def test_walk_path_index_zero_on_single_row():
    # rN[0] collapses to one dict; the model models it as a list and asks [0]
    # again -> the row itself (do-what-it-means), no "not a list of that size"
    # error.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "$r0",
                  "emphasis": [{"key": "along road_1", "w": -1}]}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r0", "refs": ["$r1[0][0]"]}},
    ])
    assert "r2" in out["results"]


def test_frame_degenerate_centerline_falls_back_to_point():
    # a node whose centerline is None (a degenerate point road) as the frame
    # reference must resolve as a POINT reference (its xy), NOT crash with
    # 'NoneType is not iterable'.
    cands = [{"fid": "d0", "xy": [5.0, 5.0], "class": "car"}]
    anch = [{"fid": "road_pt", "name": "Pt", "xy": [50.0, 50.0], "class": "road",
             "centerline": None}]
    out = run_program(cands, anch, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0",
                  "frame": {"reference": "road_pt", "side": "left"}}},
    ]})
    assert any(("left of road_pt" in r) or ("right of road_pt" in r)
               for r in out["results"]["r1"]), list(out["results"]["r1"][0])


def test_bare_rn_compare_ref_promoted():
    # compare refs:['r0'] (missing the $) where r0 is a LIVE ref -> promoted to $r0.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "prior", "query": "Main Street"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r2", "op": "compare", "args": {"items": "$r1", "refs": ["r0"]}},
    ])
    # The promoted ref RESOLVED: its gap columns attach. A road ref measures
    # its gap like any other shape -- it used to attach an 'along' chart
    # instead, which made 'distance to <road>' a different quantity than
    # 'distance to <building>' under one column name.
    assert _has_prefix(out["results"]["r2"], "distance to ")
    assert not _has_prefix(out["results"]["r2"], "along ")


def test_dangling_bare_rn_still_errors():
    # a bare rN that is NOT a live ref stays a loud error (no phantom promotion).
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": ["r9"]}},
        ])
    except ProgramError as e:
        assert "r9" in str(e)
    else:
        raise AssertionError("a dangling bare rN must still error")


def test_string_wrapped_xy_literal_compare_ref():
    # compare refs:['[50, 50]'] (an [x,y] point wrapped in a JSON string) -> parsed.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": ["[50, 50]"]}},
    ])
    assert _has_prefix(out["results"]["r1"], "distance to ")


def test_as_list_flattens_list_wrapped_ref_no_crash():
    # items:['$r0'] (the model list-wraps the ref): the nested $rN flattens to
    # its rows (as_list mirror of the as_points flatten).
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": ["$r0"], "key": "class", "op": "eq", "value": "car"}},
    ])
    assert {r["fid"] for r in out["results"]["r1"]} == {"d0", "a1", "b1"}


def test_as_points_nested_ref_in_list_literal():
    # compare items = ["$r0"] -- a $ref nested INSIDE the list literal -> the
    # point-list resolver flattens it to its rows.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": ["$r0"], "refs": ["road_1"]}},
    ])
    rows = out["results"]["r1"]
    assert {r["fid"] for r in rows} == {"d0", "a1", "b1"}
    # road_1 measures its GAP like any other shape -- an elongated ref no
    # longer swaps the gap for a lateral chart.
    assert _has_prefix(rows, "distance to road_1")


def test_as_points_single_row_wrapped():
    # compare items = "$r0[0]" -- a single resolved row where a point-LIST is
    # wanted -> wrapped into a one-item list.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0[0]", "refs": ["road_1"]}},
    ])
    assert len(out["results"]["r1"]) == 1


def test_as_list_unknown_fid_in_list_is_clean_error():
    # items:['ghost','d0'] -- a stale/typo'd fid mixed with a live one. The live
    # d0 resolves to a dict that carries 'class', so the downstream absent-key
    # guard would PASS and a later .get() on the raw 'ghost' string would crash
    # (AttributeError). as_list must reject the unknown fid LOUD instead, mirroring
    # the as_points loop -- a clean ProgramError naming the bad token.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": ["ghost", "d0"], "key": "class", "op": "eq", "value": "car"}},
        ])
    except ProgramError as e:
        assert "ghost" in str(e)
    except Exception as e:  # noqa: BLE001
        raise AssertionError("unknown fid in a list literal must be ProgramError, not %s"
                             % type(e).__name__)
    else:
        raise AssertionError("an unknown fid mixed into a list literal must error loud")


def test_walk_path_slice_too_many_colons_clean_error():
    # $r0[1:2:3:4] -- a >3-part window builds slice(1,2,3,4), which raises TypeError
    # (NOT ValueError). The except must catch it and give the tailored 'bad slice'
    # message, not leak a generic un-teachable failure.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "$r0[1:2:3:4]", "key": "class", "op": "eq", "value": "car"}},
        ])
    except ProgramError as e:
        assert "slice" in str(e).lower()
    except Exception as e:  # noqa: BLE001
        raise AssertionError("a >3-part slice must be ProgramError, not %s" % type(e).__name__)
    else:
        raise AssertionError("a >3-part slice window must error loud")


def test_walk_path_empty_window_from_nonempty_pool_is_loud():
    # $r0[5:] on a 3-row pool -> [] (start past the end; the [5:]/[:5] direction
    # slip). Python returns an empty list silently, which would drop the WHOLE
    # candidate pool with no signal. Must fail loud -- consistent with the
    # integer-index branch, which already errors on the analogous [i] miss.
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "$r0[5:]", "key": "class", "op": "eq", "value": "car"}},
        ])
    except ProgramError as e:
        assert "0 of 3" in str(e) or "empty window" in str(e)
    except Exception as e:  # noqa: BLE001
        raise AssertionError("an empty window on a non-empty pool must be ProgramError, not %s"
                             % type(e).__name__)
    else:
        raise AssertionError("a window that drops the whole pool must error loud")


def test_walk_path_start_offset_window_on_single_row_is_loud():
    # rN[0] collapses to one row; the model windows it with a START offset
    # ([1:]) believing it is multi-row -> [] . This drops the row silently;
    # must fail loud (the integer branch already errors on the analogous [1]).
    try:
        _run([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "$r0[0][1:]", "key": "class", "op": "eq", "value": "car"}},
        ])
    except ProgramError as e:
        assert "selected 0" in str(e) or "empty window" in str(e)
    except Exception as e:  # noqa: BLE001
        raise AssertionError("a start-offset window on a single row must be ProgramError, not %s"
                             % type(e).__name__)
    else:
        raise AssertionError("a start-offset window that empties a single-row result must error loud")


def test_walk_path_top_n_window_still_returns_rows():
    # non-regression: the blessed [:n] top-n window (and a full-coverage [n:]
    # that keeps rows) must still resolve, NOT trip the empty-window guard.
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "$r0[1:]", "key": "class", "op": "eq", "value": "car"}},
    ])
    assert len(out["results"]["r1"]) == 2  # dropped the first of 3, kept 2
