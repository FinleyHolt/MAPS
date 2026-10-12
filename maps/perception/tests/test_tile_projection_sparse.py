"""The tile projector emits each occupied cell once, row-major, clipped to the tile."""

from __future__ import annotations

import numpy as np

from maps.perception.geometry.geolocate import (
    TileFrame,
    project_mask_to_tile_cells_ortho,
)
from maps.perception.tests.regions import region_of


# Non-square grid so a row/col transposition cannot pass, and a view square much
# smaller than one cell-grid span so many pixels collapse into one cell.
_TILE = TileFrame(x_min=0.0, y_min=0.0, cell_size_m=2.0, grid_shape=(64, 96))


def _corners(x0, y0, side):
    return ((x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side))


def test_cells_are_unique_and_row_major():
    rng = np.random.default_rng(7)
    pixel_mask = rng.random((48, 48)) < 0.35
    tp = project_mask_to_tile_cells_ortho(
        region_of(pixel_mask), view_corners_xy=_corners(10.0, 12.0, 30.0), tile=_TILE
    )
    assert tp is not None
    flat = tp.rows.astype(np.int64) * _TILE.grid_shape[1] + tp.cols
    assert np.all(np.diff(flat) > 0)  # strictly ascending: unique and row-major
    # Many pixels collapse into each cell.
    assert 0 < tp.rows.size < int(pixel_mask.sum())
    assert tp.rows.dtype == np.int32 and tp.cols.dtype == np.int32


def test_cells_partly_outside_the_tile_are_clipped():
    pixel_mask = np.ones((16, 16), dtype=bool)
    # Straddles the tile's south-west corner: half the view square is negative.
    tp = project_mask_to_tile_cells_ortho(
        region_of(pixel_mask),
        view_corners_xy=_corners(-20.0, -20.0, 40.0),
        tile=_TILE
    )
    assert tp is not None
    H, W = _TILE.grid_shape
    assert int(tp.rows.min()) >= 0 and int(tp.rows.max()) < H
    assert int(tp.cols.min()) >= 0 and int(tp.cols.max()) < W
    assert tp.rows.size > 0


def test_fully_outside_the_tile_drops():
    pixel_mask = np.ones((16, 16), dtype=bool)
    tp = project_mask_to_tile_cells_ortho(
        region_of(pixel_mask),
        view_corners_xy=_corners(-500.0, -500.0, 40.0),
        tile=_TILE
    )
    assert tp is None


def test_min_pixels_counts_the_regions_own_pixels():
    """The under-threshold drop counts set pixels, and a region that examined
    the wrong ones would keep or drop detections the geometry does not."""
    corners = _corners(10.0, 12.0, 30.0)
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, 30:34] = True  # exactly 16 px, the threshold
    for drop_one, expect in ((False, True), (True, False)):
        m = mask.copy()
        if drop_one:
            m[30, 30] = False
        region = region_of(m)
        tp = project_mask_to_tile_cells_ortho(region, view_corners_xy=corners, tile=_TILE)
        assert (tp is not None) is expect, region.area
