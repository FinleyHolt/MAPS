"""Raster window geometry: world-to-pixel mapping and out-of-bounds padding.

``_safe_crop`` cuts the belief-grid channels (semantic / height / alpha) out
of the block raster over the ortho crop's world window, so its padding
decides what the reasoner sees at a block edge. The view square itself is
pinned by ``test_ortho_crop_geometry``.
"""

from __future__ import annotations

import numpy as np

from maps.perception.observation import BlockObservation, _safe_crop


def _block(H: int = 20, W: int = 20) -> BlockObservation:
    semantic = np.arange(H * W, dtype=np.int32).reshape(H, W)
    height = semantic.astype(np.float32) * 0.1
    return BlockObservation(
        map_name="test_block",
        semantic=semantic,
        height=height,
        x_min=0.0,
        y_min=0.0,
        px_per_m=1.0,
    )


def test_world_to_px_consistent_with_crop():
    block = _block()
    # World pose at (10, 10), raster is 20x20 with y_min=0 -> row ~ (20-1) - 10 = 9.
    r, c = block.world_to_px(10.0, 10.0)
    assert 0 <= r < 20
    assert 0 <= c < 20


def test_interior_crop_is_the_raster_itself():
    # Bounds are INCLUSIVE on both ends: 8..12 is five rows, not four.
    block = _block()
    H, W = block.shape
    out = _safe_crop(block.semantic, 8, 12, 8, 12, H, W, fill=0)
    assert out.shape == (5, 5)
    assert (out > 0).all()
    assert np.array_equal(out, block.semantic[8:13, 8:13])


def test_out_of_bounds_pads_with_fill():
    block = _block()
    H, W = block.shape
    out = _safe_crop(block.semantic, -3, 3, -3, 3, H, W, fill=0)
    assert out.shape == (7, 7)
    # The overhang is filled; the overlap carries the raster's own values.
    assert (out[:3, :3] == 0).all()
    assert np.array_equal(out[3:, 3:], block.semantic[0:4, 0:4])


def test_fill_value_is_honoured():
    block = _block()
    H, W = block.shape
    out = _safe_crop(block.height, -2, 2, -2, 2, H, W, fill=-1.0)
    assert out.shape == (5, 5)
    assert (out[:2, :2] == -1.0).all()
