"""The tests' oracle view of a region: the frame-sized boolean array."""

from __future__ import annotations

import numpy as np

from maps.util.mask_region import MaskRegion


def full(region: MaskRegion) -> np.ndarray:
    out = np.zeros(region.frame_shape, dtype=bool)
    region.window(out)[...] = region.crop
    return out
