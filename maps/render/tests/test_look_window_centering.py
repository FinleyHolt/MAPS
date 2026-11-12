"""The crop window shifts inside the raster but keeps its subject in frame."""
from __future__ import annotations

from maps.render.look_render import (
    LOOK_CAR_MARGIN_M,
    look_window_bounds_m,
    resolve_window_center,
)

# A 0.1 m/px ortho; the bounds are pixel counts.
_MIN_M, _ = look_window_bounds_m(0.1)

_BOUNDS = (113.4375, 1600.0, 400.0375, 1802.3)
_HALF = _MIN_M / 2.0
_KEEP = _HALF - LOOK_CAR_MARGIN_M / 2.0


def test_the_edge_shift_trims_the_border_up_to_the_cap():
    mid = (250.0, 1700.0)
    assert resolve_window_center(mid, mid, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M) == mid
    assert resolve_window_center(
        (1.0, 2.0), (3.0, 4.0), None, _HALF, LOOK_CAR_MARGIN_M) == (3.0, 4.0)
    # within the cap of the right edge: shifted fully in-bounds
    near = (396.0, 1700.0)
    cx, cy = resolve_window_center(near, near, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert (cx, cy) == (_BOUNDS[2] - _HALF, near[1])
    # closer than the cap allows: the shift stops at the cap
    near2 = (398.0, 1700.0)
    cx2, _ = resolve_window_center(near2, near2, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert cx2 == near2[0] - _KEEP


def test_the_cap_is_per_axis_not_a_radius():
    # An intended centre at the cap on both axes is legal per axis and over
    # by radius; a radius cap would pull it back and drop a hull corner.
    sub = (250.0, 1700.0)
    want = (sub[0] + _KEEP, sub[1] + _KEEP)
    cx, cy = resolve_window_center(sub, want, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert (round(cx, 6), round(cy, 6)) == (round(want[0], 6), round(want[1], 6))
    # one axis past the cap clamps on that axis alone
    over = (sub[0] + _KEEP + 3.0, sub[1] + _KEEP)
    cx2, cy2 = resolve_window_center(sub, over, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert round(cx2, 6) == round(sub[0] + _KEEP, 6)
    assert round(cy2, 6) == round(sub[1] + _KEEP, 6)
