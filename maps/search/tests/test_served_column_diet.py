"""The served-surface diet: what the renderer drops, and what it must not."""
from __future__ import annotations

import pytest

from maps.search.analyze_scene.geom_program import (
    _GRID_COL_DROP,
    _TAG_RENDER_DROP,
    ProgramError,
    _grid_cell,
    _render_grid_rows,
    run_program,
)

# One prior row shaped like the live surface: kept tags, dropped tags, a
# dropped column and several kept ones.
ROW = {
    "fid": "prior_52",
    "name": "Willmore Road",
    "tags": {"osm_id": "('way', 161692721)", "addr_city": "Birmingham",
             "addr_postcode": "B20 3JL", "addr_street": "Willmore Road",
             "building": "residential", "source_tag": "bing",
             "surface": "paved"},
    "xy": [316.25, 473.0],
    "footprint_area_m2": 139.91,
    "aspect_ratio": 1.02,
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
    header, sep, body = _render_grid_rows([ROW]).splitlines()[:3]
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
    assert _grid_cell({"osm_id": "1", "source_tag": "bing"}) == "-"


def test_the_drop_is_a_render_cut_not_a_data_cut():
    """The node keeps every tag, so ``find`` still reaches a dropped key."""
    from maps.search.analyze_scene.geom_program import _anchor_search_text

    class _Node:
        name = "Willmore Road"
        tags = dict(ROW["tags"])

    text = _anchor_search_text(_Node())
    assert "b20 3jl" in text                    # a DROPPED key's value
    assert "addr postcode" in text              # and its key
    assert "willmore road" in text


def test_score_by_grid_orders_without_a_score_column():
    from maps.search.analyze_scene.geom_program import (
        compact_results,
        run_program,
    )

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


def test_no_axis_heading_is_produced_at_all():
    from maps.search.analyze_scene.geom_compute import (
        contour_shape_features,
        mask_shape_features,
    )

    cells = mask_shape_features([(0, 0), (0, 1), (1, 0), (1, 1)], 2.0)
    assert set(cells) == {"footprint_area_m2", "aspect_ratio"}, cells
    box = [[[0.0, 0.0], [8.0, 0.0], [8.0, 3.0], [0.0, 3.0], [0.0, 0.0]]]
    contour = contour_shape_features(box)
    assert set(contour) == {"footprint_area_m2", "aspect_ratio"}, contour


def test_shape_scalars_are_gone_from_the_mission_surface():
    """aspect_ratio is a lossy min-rect proxy for shape, answered by the crop channel."""
    header = _render_grid_rows([ROW]).splitlines()[0]
    assert "aspect_ratio" not in header, header


def test_the_row_path_obeys_the_same_diet_as_the_grid():
    """A render cut is not a diet while a second render path exists."""
    from maps.search.analyze_scene.geom_program import (
        _RENDER_INPUT_ROW_FIELDS,
    )
    from maps.search.parse_compile.python_surface import _ROW_HIDDEN

    for col in _GRID_COL_DROP:
        assert col in _ROW_HIDDEN, col
    for col in _RENDER_INPUT_ROW_FIELDS:
        assert col in _ROW_HIDDEN, col
    # Vacuity floor: an emptied drop set would satisfy the loops above.
    assert len(_GRID_COL_DROP) >= 5, _GRID_COL_DROP
    assert {"name_match", "score", "aspect_ratio"} <= _ROW_HIDDEN


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
