"""Test helpers: the oracle view of a region (the frame-sized boolean array) and a nadir square view."""

from __future__ import annotations

import numpy as np

from maps.perception.observation import Observation
from maps.util.mask_region import MaskRegion


def full(region: MaskRegion) -> np.ndarray:
    out = np.zeros(region.frame_shape, dtype=bool)
    region.window(out)[...] = region.crop
    return out


def square_view(rgb: np.ndarray, depth: np.ndarray | None = None) -> Observation:
    """A nadir view from 100 m whose frame covers the world square (0, 0)-(100, 100)."""
    return Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=100.0,
        yaw_deg=0.0,
        world_window=(0.0, 0.0, 100.0, 100.0),
        rgb=rgb,
        depth=depth,
        ortho_view_corners_xy=((0.0, 100.0), (100.0, 100.0), (100.0, 0.0), (0.0, 0.0)),
    )
