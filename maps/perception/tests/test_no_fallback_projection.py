"""Ortho mask projection: real depth or no detection.

``project_mask_to_world_ortho`` places a mask through the view-square affine
and keeps only pixels with finite, positive depth; there is no flat-ground
fallback.
"""

from __future__ import annotations

import numpy as np

from maps.perception.geometry.geolocate import project_mask_to_world_ortho
from maps.perception.tests.regions import region_of

H, W = 48, 64
# North-up square [FL, FR, BR, BL] over x in [-32, 32], y in [-24, 24].
CORNERS = ((-32.0, 24.0), (32.0, 24.0), (32.0, -24.0), (-32.0, -24.0))


def _project(mask: np.ndarray, depth: np.ndarray | None):
    return project_mask_to_world_ortho(region_of(mask), depth=depth, view_corners_xy=CORNERS)


def _centered(half: int = 6) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[H // 2 - half : H // 2 + half + 1, W // 2 - half : W // 2 + half + 1] = True
    return mask


def test_constant_depth_yields_real_world_xy():
    projected = _project(_centered(), np.full((H, W), 50.0, np.float32))
    assert projected is not None
    assert abs(projected[0]) < 1.0 and abs(projected[1]) < 1.0


def test_missing_or_invalid_depth_drops_detection():
    assert _project(_centered(), None) is None
    assert _project(_centered(), np.full((H, W), np.nan, np.float32)) is None
    assert _project(_centered(), np.zeros((H, W), np.float32)) is None


def test_partial_valid_depth_uses_valid_pixels_only():
    depth = np.full((H, W), np.nan, np.float32)
    depth[:, : W // 2] = 50.0
    projected = _project(np.ones((H, W), dtype=bool), depth)
    assert projected is not None
    assert projected[0] < 0.0  # the valid left half lies west


def test_below_threshold_drops_detection():
    mask = np.zeros((H, W), dtype=bool)
    mask[H // 2, W // 2] = True
    assert _project(mask, np.full((H, W), 50.0, np.float32)) is None
