"""The similarity sweep keeps the coarse best when no finer yaw beats it."""

from __future__ import annotations

import numpy as np

from benchmarks.flight import frame_offset


def _mosaic(n: int = 600) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(3)
    img = np.full((n, n, 3), 150, np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, n - 30, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + 12), int(y + 9)),
                      tuple(int(c) for c in rng.integers(20, 255, 3)), -1)
    return img


def test_a_mosaic_that_needs_no_turn_is_given_none():
    img = _mosaic()
    du, dv, yaw, response = frame_offset.similarity(img, img, yaw_reach_deg=8.0)
    assert yaw == 0.0
    assert (du, dv) == (0.0, 0.0)
    assert response >= frame_offset.offset(img, img)[2] - 1e-9
