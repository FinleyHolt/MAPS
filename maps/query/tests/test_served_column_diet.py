"""The served-surface diet: what the renderer drops, and what it must not."""
from __future__ import annotations

import pytest

from maps.query.result_tables import (
    _GRID_COL_DROP, _TAG_RENDER_DROP, _grid_cell, _grid_columns, _model_facing_row,
    _render_grid_rows)
from maps.query.op_io import ProgramError
from maps.query.program import run_program

# One prior row shaped like the live surface: kept tags, dropped tags, a
# dropped column and several kept ones.
ROW = {
    "fid": "prior_52",
    "name": "Willmore Road",
    "tags": {"addr_city": "Birmingham",
             "addr_postcode": "B20 3JL", "addr_street": "Willmore Road",
             "building": "residential", "tiger_cfcc": "A41",
             "surface": "paved"},
    "xy": [316.25, 473.0],
    "footprint_area_m2": 139.91,
    "axis_length_m": 18.15,
    "color": "gray",
    "color_conf": 0.87,
    "name_match": 0.9,
    "score": 0.4123,
    "end0_xy": [300.0, 470.0],
    "end1_xy": [332.0, 476.0],
}


def test_the_live_header_keeps_and_drops_by_name():
    """The header has no idx and none of the drop set (name_match and score by name),
    keeps the shape, colour and identity columns, and its separator and body match
    its width."""
    proj = [_model_facing_row(ROW)]
    header, sep, body = _render_grid_rows(proj, _grid_columns(proj)).splitlines()[:3]
    assert header.startswith("| fid |"), header
    assert " idx " not in header
    assert header.count("|") == sep.count("|") == body.count("|")
    for col in _GRID_COL_DROP:
        assert col not in header, col
    assert "name_match" not in header and " score " not in header, header
    for col in ("footprint_area_m2", "end0_xy", "end1_xy", "color",
                "name", "tags", "xy"):
        assert col in header, col


def test_header_renders_keys_verbatim_with_units_separate():
    cols = ["distance to new_park_street", "facing new_park_street",
            "footprint_area_m2"]
    header = _render_grid_rows([], cols).splitlines()[0]
    for c in cols:
        assert c in header, header
    assert "(m)" not in header, header
    assert "(deg)" not in header, header


def test_dropped_tag_keys_do_not_render_and_kept_ones_do():
    cell = _grid_cell(ROW["tags"])
    for key in _TAG_RENDER_DROP:
        assert key not in cell, key
    # addr_street is the load-bearing one: 63.7% of cues name a street.
    for key in ("addr_street", "building", "surface"):
        assert key in cell, key


def test_a_row_whose_tags_are_all_dropped_renders_the_absent_marker():
    """Not an empty cell, which would column-break nothing but reads as a tag set that
    exists and is blank."""
    assert _grid_cell({"addr_city": "Birmingham", "tiger_cfcc": "A41"}) == "-"


def test_the_drop_is_a_render_cut_not_a_data_cut():
    """The node keeps every tag, so ``find`` still reaches a dropped key."""
    from maps.query.surface import _anchor_search_text

    class _Node:
        name = "Willmore Road"
        tags = dict(ROW["tags"])

    text = _anchor_search_text(_Node())
    assert "b20 3jl" in text                    # a DROPPED key's value
    assert "addr postcode" in text              # and its key
    assert "willmore road" in text


def test_score_by_grid_orders_without_a_score_column():
    from maps.query.result_tables import compact_results
    from maps.query.program import run_program

    def _prior(fid, x, y):
        return {"fid": fid, "name": None, "xy": [float(x), float(y)],
                "centerline": None, "tags": {"building": "yes"},
                "color": "", "search_text": "building yes"}

    world = [_prior("prior_a", 0, 10), _prior("prior_b", 30, 0),
             _prior("prior_c", 0, 20)]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": [[0.0, 0.0]]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "center distance to xy", "w": -1}]}},
    ]})
    rows = out["results"]["r2"]
    assert len(rows) >= 2                       # vacuity floor
    assert rows[0]["fid"] == "prior_a"          # rank-0 = nearest
    assert all("score" in r for r in rows), rows[0].keys()
    header = compact_results(out["results"])["r2"]["rows"].splitlines()[0]
    assert "center distance to xy" in header, header
    assert " score " not in header, header


def test_a_footprint_carries_its_area_and_no_unserved_shape_column():
    from maps.query.spatial_math import (
        contour_shape_features,
        mask_shape_features,
    )

    cells = mask_shape_features([(0, 0), (0, 1), (1, 0), (1, 1)], 2.0)
    assert cells == {"footprint_area_m2": 16.0}, cells
    box = [[[0.0, 0.0], [8.0, 0.0], [8.0, 3.0], [0.0, 3.0], [0.0, 0.0]]]
    assert contour_shape_features(box) == {"footprint_area_m2": 24.0}
    assert contour_shape_features([box[0][:-1]]) is None


@pytest.mark.parametrize("op", [
    {"op": "score_by", "args": {"emphasis": [{"key": "nope", "w": 1}]}},
    {"op": "filter_items", "args": {"key": "nope", "op": "truthy"}}])
def test_an_absent_key_refusal_names_only_served_columns(op):
    with pytest.raises(ProgramError) as e:
        run_program([], [dict(ROW)], {"ops": [
            {"ref": "r0", "op": "find", "args": {"query": "Willmore Road"}},
            {"ref": "r1", "op": op["op"], "args": {"items": "r0", **op["args"]}}]})
    msg = str(e.value)
    assert "footprint_area_m2" in msg, msg
    assert not any(k in msg for k in _GRID_COL_DROP), msg


def test_bake_bookkeeping_tags_never_reach_a_row():
    from maps.query.surface import _anchor_search_text, _row_tags

    class _Node:
        name = ""
        tags = {"building": "yes", "worldkit_layer": "buildings", "osm_id": "way/1",
                "worldkit_height_m": "6", "source_tag": "bing", "Source": "survey"}

    assert _row_tags(_Node()) == {"building": "yes"}
    assert _anchor_search_text(_Node()) == "building yes"
