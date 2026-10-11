"""The look crop fits whatever ids the request carries.

The agent picks its workflow by what it passes — many ids -> one wide badged
frame for spatial relations, one id -> a tight high-magnification crop — and
the geometry follows the distinction the crop schema always made
(measured: ordering is flat to 25 badges while identity wants
magnification). Pinned
here END-TO-END on a synthetic bake: the hull fit, the serving-budget clamp
with the subject-keeps-its-window guarantee, the derived upscale, and the
badge ceiling with its never-silent drop note.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from maps.search.render.look_render import (
    LOOK_CAR_MARGIN_M,
    LOOK_MAX_BADGES,
    LOOK_TARGET_PX,
    look_window_bounds_m,
    render_candidate_crops,
)
from maps.search.tests.ortho_fixture import write_rasters

# The bounds are pixel counts; the fixture ortho below is 0.1 m/px.
_MIN_M, _BUDGET_M = look_window_bounds_m(0.1)

MAP = "unit_test_look_block"
GROUND_Z = 10.0


def _req(center_xy, marks_xy=(), target_xy=None, fid="c0"):
    """One resolved crop spec, marks in the caller's order, target LAST —
    the same shape (and target position) ``_build_crop_requests`` emits."""
    marks = [{"fid": f"m{i}", "xy": list(xy), "boundary": None,
              "label": f"car m{i}", "is_target": False}
             for i, xy in enumerate(marks_xy)]
    if target_xy is not None:
        marks.append({"fid": "tgt", "xy": list(target_xy), "boundary": None,
                      "label": "TARGET car", "is_target": True})
    return {"center_fid": fid, "center_xy": list(center_xy),
            "center_label": f"car {fid}", "center_boundary": None,
            "annotate": bool(marks), "marks": marks}


def _render_one(ortho_dir, req):
    recs = render_candidate_crops([req], map_name=MAP, rgbd_dir=ortho_dir)
    assert len(recs) == 1, recs
    return recs[0]


def _img_side(rec) -> int:
    from PIL import Image

    # width = the delivered raster side; the overlay adds its margin band
    # (the tiny "N up" strip) to the HEIGHT only.
    return Image.open(io.BytesIO(rec["jpeg"])).size[0]


def test_many_ids_fit_one_frame(ortho_dir):
    # The tu_a defect case: co-requested ids must be IN the frame, badged.
    rec = _render_one(ortho_dir, _req(
        (100.0, 100.0),
        marks_xy=[(70.0, 100.0), (130.0, 100.0), (100.0, 75.0), (100.0, 125.0)]))
    assert rec["context_m"] == 60.0 + LOOK_CAR_MARGIN_M, rec["context_m"]
    assert rec["marks_out_of_view"] == [], rec
    assert rec["marks_unbadged_in_frame"] == [], rec
    assert rec["marks_dropped"] == 0, rec
    assert len(rec["legend"]) == 5, rec["legend"]  # center + 4 context marks
    assert rec["upscale"] == 1, rec  # 660 px native ships native


def test_a_window_that_covers_the_hull_frames_every_requested_point(ortho_dir):
    """The invariant: unclamped, EVERY requested point is in frame.

    Two independent ways the old centring broke it, each pinned by its own
    case below — a symmetric lattice exercises only the second, so a test
    built on one alone passes with the arithmetic mean still in place.
    """
    # (a) the ARITHMETIC MEAN vs the hull centre. Points bunched on one
    # side pull the mean off the hull centre; the window is SIZED off the
    # hull, so only the hull centre is guaranteed to contain it. Mean here
    # is x=108.33 against a hull centre of 112 in a 30 m window: the far
    # point falls out at 123 < 124.
    row = [(100.0, 100.0), (101.0, 100.0), (124.0, 100.0)]
    # VACUITY FLOOR, computed from the POINTS and not from the record: the
    # two candidate centres must differ by enough to move a row out of a
    # 30 m window, or this case cannot tell them apart.
    _mean_x = sum(p[0] for p in row) / len(row)
    _hull_c_x = (max(p[0] for p in row) + min(p[0] for p in row)) / 2.0
    assert abs(_mean_x - _hull_c_x) > 3.0, (_mean_x, _hull_c_x)
    rec = _render_one(ortho_dir, _req(row[0], marks_xy=row[1:]))
    assert rec["context_m"] == 24.0 + LOOK_CAR_MARGIN_M, rec["context_m"]
    assert rec["marks_out_of_view"] == [], rec
    assert len(rec["legend"]) == len(row), rec["legend"]

    # (b) the EUCLIDEAN keep-in-frame cap. The subject sits at a hull
    # CORNER, so the shift to the hull centre is diagonal and a radius cap
    # shrinks it by up to sqrt(2) — rows then fall out of a frame wide
    # enough to hold them (measured: 16 of 24 drawn, 8 named out-of-view,
    # window fully inside the raster).
    grid = [(85.0 + 6.0 * (i % 5), 85.0 + 6.0 * (i // 5)) for i in range(24)]
    # VACUITY FLOOR, again from the POINTS: the subject-to-hull-centre shift
    # must be DIAGONAL and longer than the cap, or a radius cap and a
    # per-axis cap agree and the case proves nothing.
    _ctx = 24.0 + LOOK_CAR_MARGIN_M
    _dx = (max(p[0] for p in grid) + min(p[0] for p in grid)) / 2.0 - grid[0][0]
    _dy = (max(p[1] for p in grid) + min(p[1] for p in grid)) / 2.0 - grid[0][1]
    _keep = _ctx / 2.0 - LOOK_CAR_MARGIN_M / 2.0
    assert min(_dx, _dy) > 0, (_dx, _dy)
    assert max(_dx, _dy) <= _keep, (_dx, _dy, _keep)        # legal per axis
    assert (_dx ** 2 + _dy ** 2) ** 0.5 > _keep, (_dx, _dy)  # illegal by radius
    rec = _render_one(ortho_dir, _req(grid[0], marks_xy=grid[1:]))
    assert rec["context_m"] == _ctx, rec["context_m"]
    assert rec["marks_out_of_view"] == [], rec
    assert rec["marks_unbadged_in_frame"] == [], rec
    assert rec["marks_dropped"] == 0, rec
    assert len(rec["legend"]) == len(grid), rec["legend"]
    x0, y0, x1, y1 = rec["world_window"]
    for x, y in grid:
        assert x0 <= x <= x1 and y0 <= y <= y1, ((x, y), rec["world_window"])
    assert rec["subject_in_window"] is not False, rec


def test_single_id_keeps_the_tight_crop(ortho_dir):
    # The identity workflow survives untouched: MIN window, x3, 450 px.
    rec = _render_one(ortho_dir, _req((100.0, 100.0)))
    assert rec["context_m"] == _MIN_M, rec["context_m"]
    assert rec["upscale"] == 3, rec
    assert _img_side(rec) == LOOK_TARGET_PX, rec
    assert rec["legend"] == [{"digit": 1, "fid": "c0", "label": "car c0",
                              "role": "center"}], rec["legend"]


def test_mid_span_upscale_is_derived_not_moded(ortho_dir):
    # 24 m hull -> 30 m window -> 300 px native -> x2 -> 600 px delivered:
    # the factor follows the span, not an annotate mode.
    rec = _render_one(ortho_dir, _req(
        (100.0, 100.0), marks_xy=[(88.0, 100.0), (112.0, 100.0)]))
    assert rec["context_m"] == 30.0, rec["context_m"]
    assert rec["upscale"] == 2, rec
    assert _img_side(rec) == 600, rec


def test_budget_clamps_and_the_far_anchor_falls_off(ortho_dir):
    # Beyond the serving budget the far anchor is dropped WITH its name in
    # the record; the subject keeps its own window (recenter cap).
    rec = _render_one(ortho_dir, _req(
        (30.0, 100.0), marks_xy=[(180.0, 100.0)]))
    assert rec["context_m"] == _BUDGET_M, rec["context_m"]
    assert rec["marks_out_of_view"] == ["m0"], rec
    assert rec["subject_in_window"] is True, rec
    assert rec["upscale"] == 1, rec
    assert _img_side(rec) == round(_BUDGET_M * 10), rec


def test_badge_cap_binds_at_25_never_silently(ortho_dir):
    # 24 requested neighbours + a LAST-positioned target: 23 neighbours draw,
    # the 24th is dropped and counted, the late target still gets its
    # reserved slot — 25 badges total, digits 1..25.
    grid = [(100.0 + 8.0 * (i % 5 - 2), 100.0 + 8.0 * (i // 5 - 2))
            for i in range(25) if i != 12]  # 24 spots around the center cell
    rec = _render_one(ortho_dir, _req(
        (100.0, 100.0), marks_xy=grid, target_xy=(98.0, 97.0)))
    legend = rec["legend"]
    assert len(legend) == LOOK_MAX_BADGES, legend
    assert rec["marks_dropped"] == 1, rec
    roles = [e["role"] for e in legend]
    assert roles[0] == "center" and "target" in roles, roles
    # digits are a permutation of 1..25 (the target appends last, digit 2)
    assert sorted(e["digit"] for e in legend) == list(
        range(1, LOOK_MAX_BADGES + 1))


@pytest.fixture
def coarse_ortho_dir(tmp_path, monkeypatch):
    """The same 200 m block on a 0.3 m/px grid — Cal City's bake spacing."""
    from maps.perception.ortho import camera_model
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    n = 667
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    rgb[..., 0] = (np.arange(n, dtype=np.uint32)[:, None] % 256).astype(np.uint8)
    rgb[..., 1] = (np.arange(n, dtype=np.uint32)[None, :] % 256).astype(np.uint8)
    rgb[..., 2] = 96
    write_rasters(tmp_path, MAP, rgb, np.full((n, n), GROUND_Z + 5.0, dtype=np.float32), m_per_px=0.3)
    monkeypatch.setitem(GROUND_LEVEL, MAP, GROUND_Z)
    camera_model._ORTHO_RASTER_CACHE.clear()
    return tmp_path


def test_a_coarser_grid_buys_a_wider_window_for_the_same_pixels(
        coarse_ortho_dir):
    # THE contract of the pixel-denominated bounds, and the only test whose
    # fixture is not 0.1 m/px: the window is a PIXEL budget, so a 3x coarser
    # grid holds 3x the ground at the same delivered size. Writing the
    # bounds in metres instead capped this map at a third of its budget.
    min_m, budget_m = look_window_bounds_m(0.3)
    assert (min_m, budget_m) == (45.0, 300.0), (min_m, budget_m)
    assert budget_m > _BUDGET_M and min_m > _MIN_M  # vacuity: it MOVED

    # Ceiling: a far anchor that overflows the window still clamps to it,
    # and the delivered side matches what the fine grid delivers at 100 m.
    wide = _render_one(coarse_ortho_dir, _req(
        (90.0, 100.0), marks_xy=[(540.0, 100.0)]))
    assert wide["context_m"] == budget_m, wide["context_m"]
    assert wide["upscale"] == 1, wide
    assert _img_side(wide) == round(_BUDGET_M * 10), wide  # same pixels
    assert wide["subject_in_window"] is True, wide

    # Floor: a lone maskless FID takes the derived floor, not 15 m, and the
    # <=3x upscale still lands it on the delivered target side.
    tight = _render_one(coarse_ortho_dir, _req((100.0, 100.0)))
    assert tight["context_m"] == min_m, tight["context_m"]
    assert _img_side(tight) == LOOK_TARGET_PX, tight


def test_a_shape_too_big_for_the_frame_is_not_drawn_on_it(ortho_dir):
    """The renderer half of the outline contract (the loop hands every
    footprint over; ``test_mission_loop`` pins that half).

    The outline is the mask decimated to <=96 vertices: exact at the scale
    of the whole shape, and white chords across the imagery inside a window
    holding a fraction of it. Every vertex in frame, or no ink.
    """
    # A target whose footprint runs 80 m past the window on every side.
    req = _req((100.0, 100.0), target_xy=(104.0, 100.0))
    req["marks"][-1]["boundary"] = [[20.0, 20.0], [180.0, 20.0],
                                    [180.0, 180.0], [20.0, 180.0]]
    huge = _render_one(ortho_dir, req)
    assert huge["outlined"] == [], huge["outlined"]
    # ...and it keeps its badge: the model is still told WHICH thing.
    assert [e["role"] for e in huge["legend"]] == ["center", "target"], huge

    # VACUITY FLOOR: the same request with a footprint that fits IS drawn,
    # or the assertion above passes on a renderer that outlines nothing.
    req["marks"][-1]["boundary"] = [[102.0, 98.0], [106.0, 98.0],
                                    [106.0, 102.0], [102.0, 102.0]]
    fits = _render_one(ortho_dir, req)
    assert [o["fid"] for o in fits["outlined"]] == ["tgt"], fits["outlined"]
