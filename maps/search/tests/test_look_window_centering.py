"""The crop window centers on ITS requested fid (7875 C5-gate B-3 residual).

``resolve_window_center`` derives one window center per requested subject:
the in-bounds edge-shift trims the black out-of-bounds border, but the total
shift from the subject's own position is capped at half_ctx - car_margin/2 —
so two DIFFERENT fids can never collapse onto one byte-identical displaced
window that contains neither (measured 7875: finalists prior_257 and
prior_652 both served window [385.04..400.04, 1600..1615], prior_640 a
third displaced window; 3/5 finalists unverifiable).
"""
from __future__ import annotations

from maps.search.render.look_render import (
    LOOK_CAR_MARGIN_M,
    look_window_bounds_m,
    resolve_window_center,
)

# cambridge_block_4 is a 0.1 m/px ortho; the bounds are pixel counts.
_MIN_M, _ = look_window_bounds_m(0.1)

# cambridge_block_4's real raster extent (the 7875 fixture).
_BOUNDS = (113.4375, 1600.0, 400.0375, 1802.3)
_HALF = _MIN_M / 2.0
_KEEP = _HALF - LOOK_CAR_MARGIN_M / 2.0

# The recorded request's five finalists (fid xy off the served rows).
_FIDS = {
    "prior_403": (373.12, 1605.1),
    "prior_696": (385.12, 1617.1),
    "prior_257": (423.12, 1575.1),
    "prior_640": (351.12, 1565.1),
    "prior_652": (449.12, 1583.1),
}


def _window(xy):
    cx, cy = resolve_window_center(xy, xy, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    return (round(cx - _HALF, 2), round(cy - _HALF, 2),
            round(cx + _HALF, 2), round(cy + _HALF, 2))


def test_7875_every_finalist_gets_a_distinct_window():
    windows = {fid: _window(xy) for fid, xy in _FIDS.items()}
    assert len(set(windows.values())) == len(windows), windows


def test_shift_capped_at_keep_in_frame_margin():
    # PER AXIS, which IS the containment condition for an axis-aligned
    # window: the subject keeps a half-car of margin inside each edge. A
    # Euclidean radius cap is the same rule shrunk by up to sqrt(2) on a
    # diagonal shift, and that shrink clipped requested rows out of windows
    # wide enough to hold them.
    shifted = 0
    for xy in _FIDS.values():
        cx, cy = resolve_window_center(
            xy, xy, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
        assert abs(cx - xy[0]) <= _KEEP + 1e-9, (xy, cx)
        assert abs(cy - xy[1]) <= _KEEP + 1e-9, (xy, cy)
        shifted += int((cx, cy) != tuple(xy))
    # vacuity floor: the cap is only tested where a shift was attempted.
    assert shifted >= 4, shifted


def test_the_cap_is_per_axis_not_a_radius():
    # Per axis and Euclidean disagree only on a DIAGONAL shift, so the
    # assertion above (which a radius cap also satisfies) cannot pin the
    # rule. An intended centre at KEEP on BOTH axes is legal per axis and
    # over by radius (KEEP*sqrt(2)): a radius cap pulls it back to
    # KEEP/sqrt(2) per axis and the far corner of a hull-sized window falls
    # out; the per-axis rule leaves it exactly where it was asked for.
    sub = (250.0, 1700.0)                       # comfortably inside _BOUNDS
    want = (sub[0] + _KEEP, sub[1] + _KEEP)
    cx, cy = resolve_window_center(
        sub, want, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert (round(cx, 6), round(cy, 6)) == (round(want[0], 6),
                                            round(want[1], 6)), (cx, cy)
    # ... and one axis past the cap still clamps, on that axis ALONE.
    over = (sub[0] + _KEEP + 3.0, sub[1] + _KEEP)
    cx2, cy2 = resolve_window_center(
        sub, over, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert round(cx2, 6) == round(sub[0] + _KEEP, 6), cx2
    assert round(cy2, 6) == round(sub[1] + _KEEP, 6), cy2


def test_7875_edge_collapse_is_wider_under_the_per_axis_cap():
    # The per-axis cap is LOOSER, so the region where the edge-clamp
    # collapses two subjects onto one window grows from a quarter-disc of
    # radius KEEP to its circumscribing square (x4/pi in area). Recorded,
    # not fixed: containment -- the half of 7875 that made the finalists
    # unverifiable -- still holds for both, and the delivered pixels still
    # differ because each crop marks its own subject.
    #
    # (Two subjects that BOTH sit inside the edge-clamp band already shared
    # a window under either rule -- ``_window((396.0, 1700.0)) ==
    # _window((396.5, 1700.0))`` on both. What is new is the DIAGONAL pair
    # below, which the radius cap separated.)
    a, b = (396.0, 1604.0), (396.5, 1604.5)
    wa, wb = _window(a), _window(b)
    assert wa == wb, (wa, wb)                 # the widened case, named
    for xy, w in ((a, wa), (b, wb)):
        assert w[0] <= xy[0] <= w[2] and w[1] <= xy[1] <= w[3], (xy, w)
    # the collapsed window is the one the module docstring names as the
    # 7875 defect -- pinned so a later widening of the cap is loud
    assert wa == (385.04, 1600.0, 400.04, 1615.0), wa


def test_in_bounds_subject_keeps_the_full_edge_trim():
    # A subject comfortably inside the raster is untouched; one within half
    # a context of the edge shifts fully in-bounds (subject stays in frame,
    # window all valid pixels) — the pre-existing behavior.
    mid = (250.0, 1700.0)
    assert resolve_window_center(
        mid, mid, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M) == mid
    near = (396.0, 1700.0)  # 4.04 m inside the right edge: trim fits the cap
    cx, cy = resolve_window_center(
        near, near, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert cx == _BOUNDS[2] - _HALF  # clamped fully in-bounds
    assert abs(cx - near[0]) <= _KEEP
    assert cy == near[1]
    # closer than the cap allows: the shift stops at KEEP (subject keeps a
    # half-car margin in frame; the sub-meter off-raster sliver is the
    # captioned no-data border)
    near2 = (398.0, 1700.0)
    cx2, _ = resolve_window_center(
        near2, near2, _BOUNDS, _HALF, LOOK_CAR_MARGIN_M)
    assert cx2 == near2[0] - _KEEP


def test_no_bounds_means_no_shift():
    assert resolve_window_center(
        (1.0, 2.0), (3.0, 4.0), None, _HALF, LOOK_CAR_MARGIN_M) == (3.0, 4.0)
