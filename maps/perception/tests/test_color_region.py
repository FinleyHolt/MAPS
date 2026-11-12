"""Both colour reads (the adapter's ``_dominant_color`` and M6's
``m6_attributes``) read a detection's ``MaskRegion``, never the frame."""

from __future__ import annotations

import numpy as np

from maps.perception.sam.detect_client import _dominant_color, _interior_mask
from maps.perception.color import m6_attributes
from maps.perception.tests.regions import region_of

_SHAPE = (200, 240)


def _rgb(rng):
    return rng.integers(0, 256, size=(*_SHAPE, 3), dtype=np.uint8)


def _blob(r0, c0, h, w, rng, fill=1.0):
    m = np.zeros(_SHAPE, dtype=bool)
    m[r0 : r0 + h, c0 : c0 + w] = rng.random((h, w)) < fill
    m[r0, c0] = m[r0 + h - 1, c0 + w - 1] = True
    return region_of(m)


def test_colour_is_read_through_the_region_and_never_the_frame():
    """Placements against both frame edges and in the middle: an off-by-one in
    the box would read neighbouring pixels and shift the median."""
    rng = np.random.default_rng(3)
    rgb = _rgb(rng)
    eroded_seen = fallback_seen = False

    for r0, c0, h, w in [
        (0, 0, 26, 34),  # top-left frame edge
        (80, 100, 40, 46),  # interior
        (170, 205, 30, 35),  # bottom-right frame edge
        (50, 50, 3, 4),  # under _COLOR_INTERIOR_MIN_PX
    ]:
        region = _blob(r0, c0, h, w, rng)

        dom = _dominant_color(rgb, region)
        assert dom is not None, (r0, c0)
        assert isinstance(dom[0], str) and len(dom[1]) == 3

        inner = _interior_mask(region)
        # The selection is crop-local and never frame-sized.
        assert inner.shape == region.crop.shape
        if int(inner.sum()) < region.area:
            eroded_seen = True
            # M6 needs a body to erode to; on the real detections it does.
            mix = m6_attributes(rgb, region)
            assert mix is not None, (r0, c0)
            assert set(mix) == {"color", "color_conf", "color_shade", "color_flag"}
        else:
            # Too small for the interior selection: _dominant_color falls back
            # to the whole mask, M6 declines. Both are pre-existing behaviour
            # and are asserted so a change to either is visible.
            fallback_seen = True
            assert m6_attributes(rgb, region) is None, (r0, c0)

    # Vacuity floor: both the erosion and the too-small fallback must have run.
    assert eroded_seen and fallback_seen


def test_colour_values_are_pinned():
    """A flat-colour object reads as that colour. Pins the sampling, not just
    the plumbing: reading the wrong pixels would still return a valid dict."""
    rgb = np.zeros((*_SHAPE, 3), dtype=np.uint8)
    rgb[..., :] = (30, 30, 30)  # dark background
    rgb[80:120, 100:140, :] = (235, 235, 235)  # a white object
    m = np.zeros(_SHAPE, dtype=bool)
    m[80:120, 100:140] = True
    region = region_of(m)

    name, med, conf = _dominant_color(rgb, region)
    assert name == "white", (name, med)
    assert med == [235, 235, 235]
    assert 0.0 <= conf <= 1.0
    assert m6_attributes(rgb, region)["color"].startswith("white")


def test_degenerate_inputs_return_none():
    rng = np.random.default_rng(5)
    rgb = _rgb(rng)
    region = _blob(80, 100, 20, 20, rng)

    # rgb whose frame does not match the region's frame.
    assert m6_attributes(np.zeros((8, 8, 3), np.uint8), region) is None
    # A region carrying no set pixels.
    empty = region.with_crop(np.zeros_like(region.crop))
    assert m6_attributes(rgb, empty) is None


def test_a_bare_array_raises_instead_of_silently_returning_none():
    """``m6_attributes`` catches every failure so colour cannot kill a
    flight; a bare array instead of a region is a programming error and must
    fail at the call site, not read as "no colour anywhere"."""
    import pytest

    rng = np.random.default_rng(7)
    rgb = _rgb(rng)
    bare = np.zeros(_SHAPE, dtype=bool)
    bare[80:100, 100:120] = True

    with pytest.raises(TypeError, match="MaskRegion"):
        m6_attributes(rgb, bare)

    # None is still the degenerate-input path, not a type error.
    assert m6_attributes(rgb, None) is None
