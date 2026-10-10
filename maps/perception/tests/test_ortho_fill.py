"""Ortho hole fill: enclosed no-data is filled, frame-connected no-data stays black."""

from __future__ import annotations

import numpy as np

from maps.perception.ortho.ortho_fill import fill_ortho_holes


def test_fills_enclosed_holes_and_keeps_the_frame_border_black():
    h, w = 40, 40
    rgb = np.full((h, w, 3), 120, np.uint8)
    height = np.full((h, w), 7.0, np.float32)
    alpha = np.full((h, w), 255, np.uint8)
    alpha[:, :3] = 0                     # no-data reaching the frame edge
    alpha[18:22, 18:22] = 0              # a hole enclosed by coverage
    rgb[alpha == 0] = 0
    height[alpha == 0] = 0.0
    before = (rgb.copy(), height.copy(), alpha.copy())

    out_rgb, out_h, out_a = fill_ortho_holes(rgb, height, alpha)

    hole = np.zeros((h, w), bool)
    hole[18:22, 18:22] = True
    assert (out_a[hole] == 255).all() and (out_h[hole] == 7.0).all()
    assert (out_rgb[hole] > 0).all()
    assert (out_a[:, :3] == 0).all() and (out_rgb[:, :3] == 0).all()
    covered = before[2] > 0
    assert np.array_equal(out_rgb[covered], before[0][covered])
    for a, b in zip((rgb, height, alpha), before):
        assert np.array_equal(a, b)      # inputs are not mutated
