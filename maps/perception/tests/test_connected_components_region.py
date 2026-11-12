"""Region-native component labelling matches a whole-frame reference.

``_connected_components`` labels a ``MaskRegion``'s crop rather than a
frame-sized canvas. The reference here is an independent whole-frame
``ndimage.label``, not a second call to the same function, so this pins the
crop-native result against whole-frame labelling.

Order is the load-bearing part, not just membership: ``_decode_one`` takes
``max(comps, key=area)`` as the detection's primary and ``max`` returns the
FIRST maximum, so two equal-area components are separated by list position
alone.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from maps.perception.sam import detect_client as dc
from maps.perception.sam.detect_client import (
    _connected_components,
    split_into_instances,
)
from maps.perception.tests.regions import full, region_of

_SHAPE = (160, 200)


def _blocks(*rects):
    m = np.zeros(_SHAPE, dtype=bool)
    for r0, c0, h, w in rects:
        m[r0 : r0 + h, c0 : c0 + w] = True
    return m


def _reference_components(mask):
    """What a whole-frame label pass returns, in its own order."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return [np.asarray(mask, dtype=bool)]
    return [labels == i for i in range(1, n + 1)]


def test_components_and_order_match_a_whole_frame_reference():
    cases = {
        "single": _blocks((40, 50, 20, 25)),
        "two disjoint": _blocks((20, 30, 12, 14), (90, 130, 18, 20)),
        "three, raster order not size order": _blocks(
            (10, 10, 6, 6), (60, 90, 30, 34), (120, 20, 10, 12)
        ),
        "touching the frame edges": _blocks((0, 0, 14, 16), (146, 184, 14, 16)),
        "diagonal touch is ONE component under 8-connectivity": _blocks(
            (50, 50, 10, 10), (60, 60, 10, 10)
        ),
    }
    split_seen = False
    for name, mask in cases.items():
        region = region_of(mask)
        got = _connected_components(region)
        ref = _reference_components(mask)
        assert len(got) == len(ref), name
        for i, (g, r) in enumerate(zip(got, ref)):
            assert np.array_equal(full(g), r), f"{name}: component {i} differs"
        if len(ref) > 1:
            split_seen = True
            # A multi-component result re-boxes each component to its own
            # extent, so no component drags the parent's span behind it.
            assert all(c.crop.size < region.crop.size for c in got), name

    # Vacuity floor: a multi-component case has to have run, or every
    # comparison was the single-component early return.
    assert split_seen


def test_equal_area_components_keep_their_tie_break():
    """``max(key=area)`` picks by list position when areas tie, so a
    reordering would silently move which component becomes the primary."""
    mask = _blocks((20, 20, 10, 10), (100, 150, 10, 10))
    got = _connected_components(region_of(mask))
    ref = _reference_components(mask)

    assert len(got) == 2
    assert got[0].area == got[1].area  # the tie is real
    assert np.array_equal(
        full(max(got, key=lambda c: c.area)),
        max(ref, key=lambda c: int(c.sum())),
    )


def test_single_component_hands_back_the_same_region_object():
    """``_split_and_pack`` relies on the unsplit case being the very same
    object, so the identity is contract, not incidental."""
    region = region_of(_blocks((40, 50, 20, 25)))
    comps = _connected_components(region)
    assert len(comps) == 1 and comps[0] is region


def test_split_into_instances_drops_speckle_and_redundant_components(monkeypatch):
    """The split's own rules on top of the labelling: a component under the
    noise floor goes, and one already owned by another detection's primary
    goes. Both read through the component's own window."""
    mask = _blocks((30, 40, 24, 26), (100, 150, 3, 3), (120, 20, 12, 14))
    region = region_of(mask)
    cov = np.zeros(_SHAPE, dtype=np.int16)
    cov[120:132, 20:34] = 1  # the third component is already covered

    comps = _connected_components(region)
    assert len(comps) == 3

    monkeypatch.setattr(dc, "MIN_AREA_PX", 1)
    loose = split_into_instances(comps)
    assert len(loose) == 3  # nothing dropped when nothing gates

    monkeypatch.setattr(dc, "MIN_AREA_PX", 16)
    speckle_gone = split_into_instances(comps)
    assert len(speckle_gone) == 2  # the 3x3 is under the floor

    both_gone = split_into_instances(comps, coverage_count=cov)
    assert len(both_gone) == 1  # and the covered one is redundant
    assert both_gone[0].area == max(c.area for c in loose)  # the primary survives

    for c in both_gone:
        assert c.frame_shape == _SHAPE
        assert 0 < c.area <= int(mask.sum())
