"""Test helpers: a region from a frame-sized mask, the oracle view of a region (the frame-sized boolean
array) and a nadir square view."""

from __future__ import annotations

import numpy as np

from maps.perception.observation import Observation
from maps.util.mask_region import MaskRegion


def region_of(mask: np.ndarray) -> MaskRegion | None:
    """A frame-sized mask cropped to its own extent; None when nothing is set."""
    m = np.asarray(mask, dtype=bool)
    return MaskRegion(m, (0, 0, m.shape[0] - 1, m.shape[1] - 1), m.shape).tightened()


def full(region: MaskRegion) -> np.ndarray:
    out = np.zeros(region.frame_shape, dtype=bool)
    region.window(out)[...] = region.crop
    return out


def square_view(rgb: np.ndarray, depth: np.ndarray | None = None) -> Observation:
    """A nadir view from 100 m whose frame covers the world square (0, 0)-(100, 100)."""
    return Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=100.0,
        world_window=(0.0, 0.0, 100.0, 100.0),
        rgb=rgb,
        depth=depth,
        ortho_view_corners_xy=((0.0, 100.0), (100.0, 100.0), (100.0, 0.0), (0.0, 0.0)),
    )
