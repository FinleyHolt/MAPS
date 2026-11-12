"""``MaskRegion`` refuses the state the bare (crop, bbox) pair allowed.

The point of the type is not brevity. Threading a box alongside a mask as two
arguments lets a caller pass a box belonging to a different mask, which does
not raise: it silently narrows a scan so the detection keeps a plausible
``world_xy`` a few metres off with every count still matching. These tests pin
that the mismatch is a construction-time error instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from maps.perception.mask_region import MaskRegion
from maps.perception.tests.regions import full, region_of


def _mask(shape=(40, 50), r0=10, c0=12, h=6, w=8):
    m = np.zeros(shape, dtype=bool)
    m[r0 : r0 + h, c0 : c0 + w] = True
    return m


def test_a_box_that_does_not_match_its_crop_is_rejected():
    crop = np.ones((6, 8), dtype=bool)
    MaskRegion(crop, (10, 12, 15, 19), (40, 50))  # the matching box is fine

    for bad in [(10, 12, 15, 20), (10, 12, 16, 19), (0, 0, 5, 5), (10, 12, 10, 12)]:
        with pytest.raises(ValueError, match="crop shape"):
            MaskRegion(crop, bad, (40, 50))


def test_a_box_outside_the_frame_is_rejected():
    crop = np.ones((6, 8), dtype=bool)
    for bad_frame in [(12, 50), (40, 15), (5, 5)]:
        with pytest.raises(ValueError, match="outside frame"):
            MaskRegion(crop, (10, 12, 15, 19), bad_frame)
    with pytest.raises(ValueError, match="outside frame"):
        MaskRegion(crop, (-1, 12, 4, 19), (40, 50))


def test_round_trip_through_a_frame_is_lossless():
    m = _mask()
    region = region_of(m)
    assert region is not None
    assert np.array_equal(full(region), m)
    assert region.bbox == (10, 12, 15, 19)
    assert region.area == int(m.sum())
    # The crop really is smaller — otherwise this round-trips a whole frame.
    assert region.crop.size < m.size / 4


def test_rows_cols_matches_np_where_over_the_frame():
    """Aggregates are taken over this sequence, so order matters, not just
    membership."""
    rng = np.random.default_rng(5)
    m = np.zeros((60, 80), dtype=bool)
    m[20:35, 30:50] = rng.random((15, 20)) < 0.6
    m[20, 30] = m[34, 49] = True

    region = region_of(m)
    rr_ref, cc_ref = np.where(m)
    rr, cc = region.rows_cols()
    assert np.array_equal(rr, rr_ref)
    assert np.array_equal(cc, cc_ref)


def test_window_writes_reach_the_frame_and_nothing_else():
    region = region_of(_mask())
    cov = np.zeros((40, 50), dtype=np.int16)
    region.window(cov)[region.crop] += 1

    assert cov.sum() == region.area
    assert np.array_equal(cov > 0, full(region))


def test_empty_mask_is_not_a_region():
    assert region_of(np.zeros((10, 10), dtype=bool)) is None
    region = region_of(_mask())
    assert region.with_crop(np.zeros_like(region.crop)).tightened() is None


def test_tightened_keeps_the_pixels_and_shrinks_the_box():
    m = _mask()
    parent = region_of(m)
    # A component living in the parent's box: only part of it is set.
    comp_crop = np.zeros_like(parent.crop)
    comp_crop[0:2, 0:3] = parent.crop[0:2, 0:3]
    comp = parent.with_crop(comp_crop)
    assert comp.bbox == parent.bbox  # untightened, by design

    tight = comp.tightened()
    assert tight is not None
    assert np.array_equal(full(tight), full(comp))
    assert tight.area == comp.area
    assert tight.crop.size < comp.crop.size  # the box really shrank
