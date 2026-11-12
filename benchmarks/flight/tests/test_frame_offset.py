"""The similarity sweep keeps the coarse best when no finer yaw beats it."""

from __future__ import annotations

from benchmarks.flight import frame_offset
from benchmarks.flight.tests import world


def test_a_mosaic_that_needs_no_turn_is_given_none():
    img = world.texture(600, 400)
    du, dv, yaw, response = frame_offset.similarity(img, img, yaw_reach_deg=8.0)
    assert yaw == 0.0
    assert (du, dv) == (0.0, 0.0)
    assert response >= frame_offset.offset(img, img)[2] - 1e-9
