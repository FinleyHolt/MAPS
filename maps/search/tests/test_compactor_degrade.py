#!/usr/bin/env python
"""``_degrade_tool_result_content`` — the fids+census(+color) dump stubber."""
from __future__ import annotations

import json

from maps.search.analyze_scene.compactor import (
    _TRIM_RENDER_KEYS,
    _degrade_tool_result_content,
    _iter_op_heads,
    _rows_color_entries,
)
from maps.search.analyze_scene.geom_program import (
    _CueSortedRows,
    compact_results,
)


# A realistic color grid: a confident white (0.82 -> render "white" @ "0.8"), a
# below-floor red masked to "red?" @ "0.0", and a look-flagged "gray?" @ "0.4"
# (uncertain by color_flag, so its real conf survives). The color cell + '?'
# mask + color_conf all render INTO the rows grid string.
_COLOR_ROWS = [
    {"fid": "d0", "class": "car", "color": "white 0.80 / gray 0.10",
     "color_conf": 0.82, "xy": [0.0, 0.0]},
    {"fid": "d1", "class": "car", "color": "red 0.45 / black 0.30",
     "color_conf": 0.004, "xy": [10.0, 0.0]},   # below _COLOR_CONF_FLOOR -> '?'
    {"fid": "d3", "class": "car", "color": "gray", "color_conf": 0.4,
     "color_flag": "look", "xy": [30.0, 0.0]},  # look-flagged -> '?' @ real 0.4
]


def _analyze_head() -> dict:
    """A bare analyze head — the exact ``{len, fids, rows}`` shape
    ``geom_program.compact_results`` emits for a non-empty ORDERED list result (rows
    render only when a real ordering exists)."""
    return compact_results(
        {"r0": _CueSortedRows(dict(r) for r in _COLOR_ROWS)})["r0"]


def _nested_live_payload() -> dict:
    """The LIVE analyze tool-result payload shape: ``{"results": {"r0": head}}``."""
    return {"results": {"r0": _analyze_head()}}


# --------------------------------------------------------------------------- #
# _iter_op_heads: all three payload shapes resolve to the render-bearing head   #
# --------------------------------------------------------------------------- #


def test_iter_op_heads_matches_all_three_shapes():
    bare = _analyze_head()
    nested = _nested_live_payload()
    mapping = {"r0": _analyze_head(), "r1": _analyze_head()}
    assert [h["fids"] for h in _iter_op_heads(bare)] == [["d0", "d1", "d3"]]
    assert [h["fids"] for h in _iter_op_heads(nested)] == [["d0", "d1", "d3"]]
    assert len(list(_iter_op_heads(mapping))) == 2


# --------------------------------------------------------------------------- #
# Trim scope: drop ONLY the render keys, preserve fids + census (default)       #
# --------------------------------------------------------------------------- #


def test_drops_only_render_keys_preserves_fids():
    head = _analyze_head()
    assert "rows" in head                               # sanity: render key present
    # the live compact path serves the table legend on the first grid; a
    # superseded dump's legend is render weight (the live response
    # re-serves it), so the trim drops it with rows/units/grid.
    assert "legend" in head
    trimmed = json.loads(_degrade_tool_result_content(json.dumps(head)))
    for k in _TRIM_RENDER_KEYS:
        assert k not in trimmed
    assert trimmed["fids"] == ["d0", "d1", "d3"]         # no candidate leaves the pool
    assert trimmed["len"] == 3
    # A note is stamped so the reader knows the full set stays live in the $rN.
    assert "fids preserved" in trimmed["hint"]
    # keep_color defaults False -> no color table.
    assert "colors" not in trimmed


def test_preserves_census_and_columns():
    # A compare-style census head: render keys alongside the machine-readable
    # census (by_class / named / columns / refs / cut_stats). Only rows/units
    # go; the census survives.
    head = {
        "refs": [[100.0, 0.0]], "len": 120,
        "columns": ["color", "color_conf", "distance to r0"],
        "by_class": {"car": 100, "truck": 20}, "named": 5,
        "cut_stats": {"cut": 7, "admit_at": 3.2},
        "units": "d_/dc_/along_/off_/vx_/vy_ in meters",
        "rows": "| idx | fid |\n...",
        "hint": "existing hint",
    }
    trimmed = json.loads(_degrade_tool_result_content(json.dumps(head)))
    assert "rows" not in trimmed and "units" not in trimmed
    assert trimmed["by_class"] == {"car": 100, "truck": 20}
    assert trimmed["named"] == 5
    assert trimmed["columns"] == ["color", "color_conf", "distance to r0"]
    assert trimmed["refs"] == [[100.0, 0.0]]
    assert trimmed["cut_stats"] == {"cut": 7, "admit_at": 3.2}
    # Existing hint is EXTENDED, not clobbered.
    assert trimmed["hint"].startswith("existing hint")
    assert "fids preserved" in trimmed["hint"]


def test_noop_when_no_render_keys():
    # A census-only dump (no rows/units/grid) has nothing trimmable -> None,
    # so the caller leaves the message verbatim.
    census_only = {"len": 120, "by_class": {"car": 120}, "named": 0,
                   "hint": "wide retrieval summarized"}
    assert _degrade_tool_result_content(json.dumps(census_only)) is None


def test_noop_on_non_dict_and_non_json():
    assert _degrade_tool_result_content("not json at all") is None
    assert _degrade_tool_result_content(json.dumps([1, 2, 3])) is None


def test_mapping_form_trims_each_head():
    # Mapping shape: results can be a {op_ref: head} map; every inner head is
    # trimmed, fids preserved.
    mapping = {
        "r0": {"len": 2, "fids": ["a", "b"], "units": "U", "rows": "R"},
        "r1": {"len": 1, "fids": ["c"], "units": "U2", "rows": "R2"},
    }
    trimmed = json.loads(_degrade_tool_result_content(json.dumps(mapping)))
    for ref in ("r0", "r1"):
        assert "rows" not in trimmed[ref] and "units" not in trimmed[ref]
        assert trimmed[ref]["fids"] == mapping[ref]["fids"]


def test_nested_live_shape_is_trimmed():
    # The LIVE nested payload IS trimmed: _iter_op_heads resolves the head
    # through the "results" wrapper, not only bare/mapping heads.
    payload = _nested_live_payload()
    trimmed = _degrade_tool_result_content(json.dumps(payload))
    assert trimmed is not None                            # the nested shape trims
    body = json.loads(trimmed)
    head = body["results"]["r0"]
    for k in _TRIM_RENDER_KEYS:
        assert k not in head
    assert head["fids"] == ["d0", "d1", "d3"]             # pool intact
    assert "fids preserved" in head["hint"]


# --------------------------------------------------------------------------- #
# Color-reliability: keep_color=True survives the trim                         #
# --------------------------------------------------------------------------- #


def test_color_reliability_survives_with_keep_color():
    # Pre-degrade the color signal is present in the rendered grid.
    head = _analyze_head()
    assert "red?" in head["rows"]              # below-floor color masked '?'
    assert "gray?" in head["rows"]             # look-flagged color masked '?'

    trimmed = json.loads(
        _degrade_tool_result_content(json.dumps(head), keep_color=True))
    assert "rows" not in trimmed              # render keys still dropped ...
    colors = trimmed["colors"]              # ... but the color table survives
    # The per-fid color entries carry the color word and the '?' uncertainty
    # mask. The numeric color_conf float went with its column (use-ranked
    # diet: 1.4% mention, 3 landed) -- reliability rides the mask on the
    # word, and the ordering weights read the raw conf, never the render.
    assert "d0=white" in colors
    assert "d1=red?" in colors               # '?' mask intact on a below-floor read
    assert "d3=gray?" in colors              # '?' mask intact on a look-flagged read
    surviving = json.dumps(trimmed)
    assert "?" in surviving                   # the uncertainty mask reached the model
    assert "@" not in colors                  # no float rides the table any more


def test_keep_color_true_on_nested_live_shape():
    # keep_color flows through the nested-shape resolver too (the tier-1 path).
    payload = _nested_live_payload()
    trimmed = json.loads(
        _degrade_tool_result_content(json.dumps(payload), keep_color=True))
    head = trimmed["results"]["r0"]
    assert "rows" not in head
    assert "d3=gray?" in head["colors"]
    assert "@" not in head["colors"]


def test_keep_color_false_drops_rows_without_colors():
    head = _analyze_head()
    trimmed = json.loads(
        _degrade_tool_result_content(json.dumps(head), keep_color=False))
    assert "rows" not in trimmed and "units" not in trimmed
    assert "colors" not in trimmed           # no color table added
    assert trimmed["fids"] == ["d0", "d1", "d3"]   # but the pool is intact


# --------------------------------------------------------------------------- #
# _rows_color_entries: parse fid=color[@conf] out of a real grid              #
# --------------------------------------------------------------------------- #


def test_rows_color_entries_parses_grid():
    # The parser still reads a conf column when one is present (older stored
    # transcripts).
    entries = _rows_color_entries(_analyze_head()["rows"])
    assert "d0=white" in entries
    assert "d1=red?" in entries
    assert "d3=gray?" in entries              # '?' mask preserved verbatim


def test_fids_never_removed_by_any_form():
    # Every shape/keep_color combination preserves fids exactly.
    bare = _degrade_tool_result_content(json.dumps(_analyze_head()))
    nested = _degrade_tool_result_content(json.dumps(_nested_live_payload()),
                                          keep_color=True)
    assert json.loads(bare)["fids"] == ["d0", "d1", "d3"]
    assert json.loads(nested)["results"]["r0"]["fids"] == ["d0", "d1", "d3"]


# --------------------------------------------------------------------------- #
# Grid parsing edge cases                                                       #
# --------------------------------------------------------------------------- #


def test_separator_row_not_leaked_into_color_table():
    """The markdown '| --- | --- |' separator row is not a data row."""
    entries = _rows_color_entries(_analyze_head()["rows"])
    assert "---=---@---" not in entries
