"""The crop window's OBSERVATION ceiling (``obs_span_m``) — what it clamps."""

from __future__ import annotations

import numpy as np

from maps.render.look_render import (
    look_window_bounds_m,
    render_candidate_crops,
)

# 1024 visual tokens = a 1024x1024 px image; the serve sets no pixel cap.
LOOK_PIXEL_BUDGET_PX = 1024 * 1024

# The bounds are pixel counts; the fixture ortho below is 0.1 m/px.
_MIN_M, _BUDGET_M = look_window_bounds_m(0.1)

MAP = "unit_test_obs_span_block"
GROUND_Z = 10.0


ORTHO_N = 2800


def _lattice(n: int, span_m: float, cx: float = 260.0, cy: float = 260.0):
    """``n`` marks on a square lattice whose bbox side is ``span_m``."""
    side = int(n ** 0.5) + 1
    step = span_m / max(1, side - 1)
    return [{"fid": f"detected_{i}",
             "xy": [cx + step * (i % side - (side - 1) / 2.0),
                    cy + step * (i // side - (side - 1) / 2.0)]}
            for i in range(n)]


def _render(ortho_dir, set_span_m: float, obs_span_m: float | None, n: int = 24):
    pts = _lattice(n, set_span_m)
    req = {
        "center_fid": pts[0]["fid"],
        "center_xy": pts[0]["xy"],
        "center_label": "car",
        "center_boundary": None,
        "marks": [{"fid": p["fid"], "xy": p["xy"], "label": "car",
                   "is_target": False} for p in pts],
    }
    if obs_span_m is not None:
        req["obs_span_m"] = obs_span_m
    (rec,) = render_candidate_crops([req], map_name=MAP, rgbd_dir=ortho_dir)
    return rec


def _delivered_px(rec) -> tuple[int, int]:
    import cv2
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    return int(img.shape[1]), int(img.shape[0])


def test_the_observation_ceiling_clamps_the_window_and_drops_badges(ortho_dir):
    # An 80 m set wants an 86 m window (hull + a car of margin). A stop
    # that only observed 40 m across cannot serve it: the window is the
    # 40 m the stop saw, and the marks outside it go undrawn.
    free = _render(ortho_dir, 80.0, None)
    capped = _render(ortho_dir, 80.0, 40.0)

    # VACUITY FLOOR: this asserts nothing unless the unclamped window was
    # really wider than the ceiling and really held more badges. A set
    # that happened to fit inside 40 m would pass every line below while
    # exercising no clamp at all.
    assert free["context_m"] > 40.0, free["context_m"]
    assert _BUDGET_M > free["context_m"], free["context_m"]  # not the budget
    assert 0 < len(capped["legend"]) < len(free["legend"]), (
        len(capped["legend"]), len(free["legend"]))

    assert capped["context_m"] == 40.0, capped["context_m"]
    # the served text names every row the ceiling pushed out of frame
    assert len(capped["marks_out_of_view"]) == 24 - len(capped["legend"]), capped


def test_a_ceiling_under_the_minimum_window_cannot_shrink_it_further(ortho_dir):
    # A stop that observed less than the minimum window still gets the
    # minimum: below MIN the subject has no surround to be placed in, and
    # MIN > LOOK_CAR_MARGIN_M is what keeps the recenter cap positive.
    rec = _render(ortho_dir, 80.0, 5.0)
    assert rec["context_m"] == _MIN_M, rec["context_m"]


def test_a_nonpositive_span_is_not_a_ceiling(ortho_dir):
    # 2*(alt - ground) goes NEGATIVE on a block whose ground datum sits
    # above the altitude cap (the AMSL field blocks: ground 695-880 m
    # against altitude_cap_m 200). A negative side is not a 0 m window and
    # not a 15 m one — it is the absence of a measurement, so the request
    # carries no observation ceiling and the budget alone clamps.
    free = _render(ortho_dir, 80.0, None)
    negative = _render(ortho_dir, 80.0, -187.6)
    assert negative["context_m"] == free["context_m"], (
        negative["context_m"], free["context_m"])
    assert len(negative["legend"]) == len(free["legend"])


def test_the_budget_window_stays_inside_the_pixel_budget(ortho_dir):
    # Two more metres would ship 1020 x 1040 = 1,060,800 and blow it. Nothing else in
    # the tree checks the constant against the budget it cites, and no serve path clamps
    # an oversized crop for us — sglang sets no pixel cap.
    rec = _render(ortho_dir, 400.0, 300.0)
    # VACUITY FLOOR: the budget, not the hull and not the observation
    # ceiling, is what sized this window.
    assert rec["context_m"] == _BUDGET_M, rec["context_m"]
    w, h = _delivered_px(rec)
    assert w * h <= LOOK_PIXEL_BUDGET_PX, (w, h, w * h)
    over_side = int(round((_BUDGET_M + 2.0) * 10))
    assert over_side * (over_side + 20) > LOOK_PIXEL_BUDGET_PX, over_side
