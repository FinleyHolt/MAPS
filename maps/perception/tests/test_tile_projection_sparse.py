"""The tile projectors emit sparse cells identical to a dense raster.

Both projectors used to allocate a ``(tile_H, tile_W)`` bool scratch grid per
detection and scan it back with ``np.where``. On a full-city tile that is ~10 M
cells (10 MB) per detection, and the result crossed a pickled HTTP boundary into
the world-state ingest server. The sparse path dedupes on the flat cell index
instead. This pins that the swap changed the representation and nothing else:
cells, order, centroid, bbox source, and count all match the dense computation.
"""

from __future__ import annotations

import numpy as np

from maps.perception.geometry.geolocate import (
    TileFrame,
    project_mask_to_tile_cells_ortho,
)
from maps.util.mask_region import MaskRegion


def _dense_reference(rows, cols, grid_shape):
    """What the deleted dense path computed: raster, then ``np.where``."""
    H, W = grid_shape
    grid = np.zeros((H, W), dtype=bool)
    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if not np.any(in_bounds):
        return None
    grid[rows[in_bounds], cols[in_bounds]] = True
    return np.where(grid)


def _cell_indices(tp_pixel_mask, view_corners_xy, tile):
    """The per-pixel (row, col) indices the projector derives before dedupe."""
    rr, cc = np.where(tp_pixel_mask)
    (flx, fly), (frx, fry), _br, (blx, bly) = (
        view_corners_xy[0],
        view_corners_xy[1],
        view_corners_xy[2],
        view_corners_xy[3],
    )
    h, w = tp_pixel_mask.shape
    u = cc.astype(np.float64) / max(w - 1, 1)
    v = rr.astype(np.float64) / max(h - 1, 1)
    xs = flx + u * (frx - flx) + v * (blx - flx)
    ys = fly + u * (fry - fly) + v * (bly - fly)
    col_idx = np.rint((xs - tile.x_min) / tile.cell_size_m - 0.5).astype(np.int64)
    row_idx = np.rint((ys - tile.y_min) / tile.cell_size_m - 0.5).astype(np.int64)
    return row_idx, col_idx


# Non-square grid so a row/col transposition cannot pass, and a view square much
# smaller than one cell-grid span so many pixels collapse into one cell (the
# dedupe case the dense raster used to handle implicitly).
_TILE = TileFrame(x_min=0.0, y_min=0.0, cell_size_m=2.0, grid_shape=(64, 96))


def _corners(x0, y0, side):
    return ((x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side))


def test_sparse_projection_matches_dense_raster():
    rng = np.random.default_rng(7)
    pixel_mask = rng.random((48, 48)) < 0.35
    corners = _corners(10.0, 12.0, 30.0)

    tp = project_mask_to_tile_cells_ortho(
        MaskRegion.from_full(pixel_mask), view_corners_xy=corners, tile=_TILE
    )
    assert tp is not None

    row_idx, col_idx = _cell_indices(pixel_mask, corners, _TILE)
    ref_rows, ref_cols = _dense_reference(row_idx, col_idx, _TILE.grid_shape)

    # Same cells, same row-major order — not merely the same set.
    assert np.array_equal(tp.rows, ref_rows)
    assert np.array_equal(tp.cols, ref_cols)

    # Vacuity floor: the projection has to actually collapse pixels into a
    # smaller set of real cells, or a projector returning junk would pass the
    # equality checks above by matching an equally junk reference.
    assert 0 < tp.rows.size < int(pixel_mask.sum())
    assert tp.rows.dtype == np.int32 and tp.cols.dtype == np.int32


def test_cells_partly_outside_the_tile_are_clipped():
    pixel_mask = np.ones((16, 16), dtype=bool)
    # Straddles the tile's south-west corner: half the view square is negative.
    tp = project_mask_to_tile_cells_ortho(
        MaskRegion.from_full(pixel_mask),
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
        MaskRegion.from_full(pixel_mask),
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
        region = MaskRegion.from_full(m)
        tp = project_mask_to_tile_cells_ortho(region, view_corners_xy=corners, tile=_TILE)
        assert (tp is not None) is expect, region.area
