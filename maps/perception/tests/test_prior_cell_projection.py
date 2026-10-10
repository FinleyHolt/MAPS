"""Every cell projects at once, and lands exactly where one-at-a-time did.

``_rasterize_cells_to_mask`` runs once per prior in view, and a Cal City stop
has ~100 of those. Walking cells in Python meant ~63,000 four-row allocations
and four-row matmuls per stop -- measured at 1.13 s of a 4.3 s stop, the
largest single cost left in one, and it ran on EVERY stop of every episode.

The vectorized form has to be byte-identical, not close: these quads are
filled into the mask that the prior's colour is read over, so a one-pixel
difference is a different colour verdict on a candidate the model ranks.
Checked offline against all 6,188 Cal City footprints (472,470 cells in the
first 500 alone): zero differing. These tests keep it that way.
"""

from __future__ import annotations

import cv2
import numpy as np

from maps.perception.viewpoint.base import TileBounds
from maps.perception.viewpoint.overlay import (
    _project_cell_quads,
    _project_cell_quads_one_by_one,
    _rasterize_cells_to_mask,
)

_IMG = 256
_BOUNDS = TileBounds(x_min=0.0, y_min=0.0, cell_size_m=2.0, grid_shape=(512, 512))


def _homography():
    src = np.array([(100.0, 100.0), (420.0, 100.0), (420.0, 340.0), (100.0, 340.0)],
                   dtype=np.float32)
    dst = np.array([(0, 0), (_IMG - 1, 0), (_IMG - 1, _IMG - 1), (0, _IMG - 1)],
                   dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def _footprints():
    """A few shapes with the spread the bake actually carries: a compact
    building, a long thin way, and one large enough that the per-cell walk
    was the whole cost."""
    return [
        [[r, c] for r in range(60, 66) for c in range(70, 78)],
        [[80, c] for c in range(50, 140)],
        [[r, c] for r in range(90, 130) for c in range(60, 110)],
    ]


def test_the_two_projections_agree_cell_for_cell():
    h_t = _homography()
    checked = 0
    for cells in _footprints():
        fast = _project_cell_quads(cells, _BOUNDS, h_t, 2.0, 1.0)
        slow = _project_cell_quads_one_by_one(cells, _BOUNDS, h_t, 2.0, 1.0)
        assert len(fast) == len(slow), (len(fast), len(slow))
        # vacuity floor: a projection that kept nothing agrees trivially
        assert len(fast) >= 40, len(fast)
        for a, b in zip(fast, slow):
            assert np.array_equal(a, b), (a, b)
            assert a.dtype == b.dtype == np.int32
        checked += len(fast)
    assert checked >= 2000, checked


def test_the_mask_is_the_same_mask():
    """The quads are the means; the mask is what the colour is read over."""
    h_t = _homography()
    for cells in _footprints():
        got = _rasterize_cells_to_mask(cells, _BOUNDS, h_t, _IMG, _IMG)
        quads = _project_cell_quads_one_by_one(cells, _BOUNDS, h_t, 2.0, 1.0)
        from maps.perception.viewpoint.overlay import _fill_region

        want = _fill_region(quads, img_h=_IMG, img_w=_IMG)
        assert got is not None and want is not None
        assert got.bbox == want.bbox, (got.bbox, want.bbox)
        assert got.frame_shape == want.frame_shape
        assert np.array_equal(got.crop, want.crop)
        assert got.crop.any(), "an empty crop compares equal for free"


def test_malformed_cells_fall_back_and_keep_the_good_ones():
    """A ragged list is a corrupt bake, not a shape the fast path handles.

    The per-cell walk drops the bad entries and keeps the rest; the fast path
    says so by returning ``None`` rather than guessing.
    """
    h_t = _homography()
    ragged = [[60, 70], [61], [62, 72], "nope", [63, 73]]
    assert _project_cell_quads(ragged, _BOUNDS, h_t, 2.0, 1.0) is None
    kept = _project_cell_quads_one_by_one(ragged, _BOUNDS, h_t, 2.0, 1.0)
    assert len(kept) == 3, kept
    # and the caller still produces a mask from the survivors
    assert _rasterize_cells_to_mask(ragged, _BOUNDS, h_t, _IMG, _IMG) is not None


def test_a_prior_with_no_cells_is_skipped_and_the_rest_of_the_stop_survives():
    """An empty footprint projects to no mask (``None``), the same as one wholly
    outside the view, so the builder skips that prior and keeps the others.
    A frame-sized array in its place raised on ``rows_cols()`` and the caller's
    broad except dropped the whole stop's prior overlay."""
    from maps.perception.viewpoint.overlay import build_prior_polys_ortho

    assert _rasterize_cells_to_mask([], _BOUNDS, _homography(), _IMG, _IMG) is None
    features = {"building": [
        {"fid": "empty", "centroid_row": 62, "centroid_col": 74, "cells": []},
        {"fid": "kept", "centroid_row": 62, "centroid_col": 74, "cells": _footprints()[0]},
    ]}
    corners = [(100.0, 340.0), (420.0, 340.0), (420.0, 100.0), (100.0, 100.0)]
    polys, table = build_prior_polys_ortho(
        rgb_shape=(_IMG, _IMG), features=features, bounds=_BOUNDS,
        view_corners_xy=corners, fids_to_render=["empty", "kept"],
    )
    assert [p["fid"] for p in polys] == ["kept"]
    assert [r["fid"] for r in table] == ["kept"]
