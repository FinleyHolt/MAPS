"""Filling a prior's box draws exactly the pixels a whole-frame fill would,
including for polygons that hang off the frame."""

from __future__ import annotations

import cv2
import numpy as np

from maps.perception.ortho.prior_overlay import _fill_region
from maps.perception.tests.regions import full

_H, _W = 120, 160


def _reference(polys) -> np.ndarray:
    """What a frame-sized fill produced."""
    mask = np.zeros((_H, _W), np.uint8)
    cv2.fillPoly(mask, polys, 1)
    return mask.astype(bool)


def _quad(x0, y0, x1, y1):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32
    )


def _tri(pts):
    return np.array(pts, dtype=np.int32)


def test_region_fill_matches_a_whole_frame_fill():
    cases = {
        "interior quad": [_quad(20, 30, 60, 70)],
        "two disjoint quads": [_quad(5, 5, 20, 20), _quad(100, 80, 150, 110)],
        "triangle": [_tri([[30, 10], [90, 40], [40, 100]])],
        "touching the frame edge": [_quad(0, 0, 25, 25)],
        "hanging off the left": [_quad(-40, 20, 30, 60)],
        "hanging off the bottom-right": [_quad(120, 90, 300, 400)],
        "straddling the whole frame": [_quad(-50, -50, 400, 400)],
    }
    clipped_seen = False
    for name, polys in cases.items():
        ref = _reference(polys)
        region = _fill_region(polys, img_h=_H, img_w=_W)
        assert region is not None, name
        assert np.array_equal(full(region), ref), name
        assert region.area == int(ref.sum()), name
        # The box is inside the frame even when the polygon is not.
        r0, c0, r1, c1 = region.bbox
        assert 0 <= r0 <= r1 < _H and 0 <= c0 <= c1 < _W, name
        if polys[0].min() < 0 or polys[0].max() >= max(_H, _W):
            clipped_seen = True

    # Vacuity floor: the clipping cases must have run, since that is where a
    # box derived from raw polygon extents would have gone out of bounds.
    assert clipped_seen


def test_a_small_prior_does_not_get_a_frame_sized_array():
    """A feature covering a fraction of the frame carries a crop of that scale."""
    region = _fill_region([_quad(20, 30, 40, 50)], img_h=_H, img_w=_W)
    assert region is not None
    assert region.crop.size < (_H * _W) / 10
    assert region.frame_shape == (_H, _W)


def test_nothing_inside_the_frame_is_none():
    """None means the prior touches no pixel, so it is skipped."""
    assert _fill_region([], img_h=_H, img_w=_W) is None
    assert _fill_region([_quad(-400, -400, -300, -300)], img_h=_H, img_w=_W) is None
    assert _fill_region([_quad(500, 500, 600, 600)], img_h=_H, img_w=_W) is None
    # A degenerate polygon is not nothing: cv2 fills its single pixel.
    degenerate = [_tri([[10, 10], [10, 10], [10, 10]])]
    ref = _reference(degenerate)
    assert int(ref.sum()) == 1
    got = _fill_region(degenerate, img_h=_H, img_w=_W)
    assert got is not None and np.array_equal(full(got), ref)
