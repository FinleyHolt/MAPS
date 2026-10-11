"""The served-surface diet: what the renderer drops, and what it must not.

These pins exist because the diet passed all 527 search tests without one of
them noticing. Every fixture in the suite hands ``_render_grid_rows`` its own
synthetic header, so nothing asserted the shape the LIVE renderer emits -- the
"derived tests fail open" hazard, in its exact form. The contract below is
stated by name so its disappearance cannot go unremarked.

The cut rule is the COMPREHENSION rule: a column goes or stays by what the
model DOES with its values, not how often it names it. A mention counter
inverts the two most consequential calls (footprint_area_m2 and end1_xy are
quoted by VALUE, never by name).
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import (
    _GRID_COL_DROP,
    _TAG_RENDER_DROP,
    _grid_cell,
    _render_grid_rows,
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


def test_the_header_carries_no_idx_column():
    """``idx`` restated the row's own position. ``rN[i]`` addresses position
    directly and needs no printed index -- and was itself emitted 0 times in
    2,700 episodes, as was any mention of ``idx`` in a say sentence."""
    header = _render_grid_rows([ROW]).splitlines()[0]
    assert header.startswith("| fid |"), header
    assert " idx " not in header


def test_the_separator_still_matches_the_header_width():
    """Dropping a column and not its rule renders a broken markdown table."""
    header, sep, body = _render_grid_rows([ROW]).splitlines()[:3]
    assert header.count("|") == sep.count("|") == body.count("|")


def test_a_dropped_column_does_not_render():
    header = _render_grid_rows([ROW]).splitlines()[0]
    for col in _GRID_COL_DROP:
        assert col not in header, col


def test_the_kept_shape_columns_still_render():
    """Each keep has a NAMED reason (the drop-set comment carries them):
    footprint_area_m2 5/8 load-bearing sleeper (the only size-adjective sort
    and only degenerate-mask detector), end0_xy/end1_xy 8/8 consumed as a
    PAIR (direction, extent, ends), and color the 8/8 primary target-side
    signal. The audit REMOVED two former keeps: name_match (7/8
    not-observed) now sits in the drop set, and the min-area rectangle's
    axis heading is gone from the tree entirely."""
    header = _render_grid_rows([ROW]).splitlines()[0]
    for col in ("footprint_area_m2", "end0_xy", "end1_xy", "color",
                "name", "tags", "xy"):
        assert col in header, col


def test_dropped_tag_keys_do_not_render_and_kept_ones_do():
    cell = _grid_cell(ROW["tags"])
    for key in _TAG_RENDER_DROP:
        assert key not in cell, key
    # addr_street is the load-bearing one: 63.7% of cues name a street.
    for key in ("addr_street", "building", "surface"):
        assert key in cell, key


def test_a_row_whose_tags_are_all_dropped_renders_the_absent_marker():
    """Not an empty cell, which would column-break nothing but reads as a
    tag set that exists and is blank."""
    assert _grid_cell({"osm_id": "1", "source_tag": "bing"}) == "-"


def test_the_drop_is_a_render_cut_not_a_data_cut():
    """The node keeps every tag, so ``find`` still reaches a dropped key.

    ``_anchor_search_text`` is the fuzzy-match haystack and it walks the
    node's OWN tags, not the rendered cell. Asserting against the builder --
    rather than against a string this test wrote -- is what makes it fail if
    someone ever starts building search_text from the rendered surface.
    """
    from maps.search.analyze_scene.geom_program import _anchor_search_text

    class _Node:
        name = "Willmore Road"
        tags = dict(ROW["tags"])

    text = _anchor_search_text(_Node())
    assert "b20 3jl" in text                    # a DROPPED key's value
    assert "addr postcode" in text              # and its key
    assert "willmore road" in text


def test_score_by_grid_orders_without_a_score_column():
    """The ordering echo IS rank order + the raw discriminator columns: the
    emphasis key the model named renders first in the budget walk, and the
    composite ``score`` never renders (audit: inert 1/1/4/2, never decided a
    commit as a value; on colour blocks a rendered duplicate of an in-row
    value). The raw rows keep ``score`` -- render cut only."""
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
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
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


def test_the_comprehension_cuts_are_pinned_by_name():
    """test_a_dropped_column_does_not_render iterates the LIVE drop set, so
    it fails open: a name removed from ``_GRID_COL_DROP`` just shrinks the
    loop (measured: reverting a cut left every test green). Pin the audit's
    comprehension cuts by NAME -- name_match and score keep their own suites
    and ride here as the one authoritative drop list."""
    header = _render_grid_rows([ROW]).splitlines()[0]
    assert "name_match" not in header, header
    assert " score " not in header, header


def test_no_axis_heading_is_produced_at_all():
    """Orientation is a RELATION to the feature a thing faces, so the
    min-area rectangle's heading is deleted at the producer rather than cut
    at the renderer. A render cut was not enough: the same column reached
    the model down the Row path (1,378 served occurrences over 43 of 93
    episodes, calcity_conj_93_72fe9987), and an undirected [0, 180) heading
    reads identically for a north-facing and a south-facing house on one
    street -- the exact question that family asks. Pinned at the two
    producers and at the served-key pattern, because a drop-set entry can
    only hide a key that something still writes."""
    from maps.search.analyze_scene.geom_compute import (
        contour_shape_features,
        mask_shape_features,
    )
    from maps.search.analyze_scene.geom_program import KEY_PATTERN

    cells = mask_shape_features([(0, 0), (0, 1), (1, 0), (1, 1)], 2.0)
    assert set(cells) == {"footprint_area_m2", "aspect_ratio"}, cells
    box = [[[0.0, 0.0], [8.0, 0.0], [8.0, 3.0], [0.0, 3.0], [0.0, 0.0]]]
    contour = contour_shape_features(box)
    assert set(contour) == {"footprint_area_m2", "aspect_ratio"}, contour
    assert "orientation" not in KEY_PATTERN, KEY_PATTERN


def test_shape_scalars_are_gone_from_the_mission_surface():
    """Owner decision: aspect_ratio is a lossy min-rect proxy for shape -- a
    looking question, answered by the crop channel. The render drop hides the
    column; this drives the ONE remaining served mention (the filter_items
    scalar-fields hint) and pins that it no longer advertises aspect_ratio
    and steers to looking instead. The field itself stays on the node."""
    import pytest

    from maps.search.analyze_scene.geom_program import (
        ProgramError,
        run_program,
    )

    header = _render_grid_rows([ROW]).splitlines()[0]
    assert "aspect_ratio" not in header, header
    world = [{"fid": "prior_a", "name": None, "xy": [1.0, 2.0],
              "centerline": None, "tags": {"building": "yes"},
              "color": "", "search_text": "building yes"}]
    with pytest.raises(ProgramError) as err:
        run_program([], world, {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "r0", "key": "xy", "op": "gt", "value": 5}},
        ]})
    served = str(err.value)
    assert "SCALAR key" in served, served
    assert "aspect_ratio" not in served, served
    assert "looking question" in served, served


def test_the_row_path_obeys_the_same_diet_as_the_grid():
    """A render cut is not a diet while a second render path exists.

    The grid dropped ``_GRID_COL_DROP``; ``Row.keys()`` filtered only the
    geometry blobs, and ``Row.__repr__`` is built from ``keys()``. So
    ``print(row)`` -- which the Row docstring itself teaches as the way to
    read a row -- handed back every column the audit cut. Measured on
    calcity_conj_93_72fe9987 before this pin: aspect_ratio 1,315 served
    occurrences over 49 of 93 episodes, color_conf 1,158 over 43, score 431
    over 45, axis_length_m 192 over 36, name_match 160 over 32.

    ``_RENDER_INPUT_ROW_FIELDS`` rides the same rule: color_shade and
    color_flag are folded into the compact color cell and are not columns.
    """
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
