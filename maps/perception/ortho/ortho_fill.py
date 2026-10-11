"""Interior hole fill for a baked ortho (``alpha == 0`` is no-data).

A black component enclosed by coverage (a photogrammetry gap under canopy, a
scan seam) is filled; one touching the frame edge (water, scan edges, the
block's exterior) stays black, so no fill smears toward a frame boundary.
"""

from __future__ import annotations

import numpy as np

# TELEA inpaint neighborhood radius (px). Holes are dominated by 1-4 px
# components; 3 px reaches across them without long-range smearing.
INPAINT_RADIUS_PX = 3


def fill_in_coverage_holes(
    rgb: np.ndarray,
    hole_mask: np.ndarray,
    no_coverage_mask: np.ndarray,
) -> np.ndarray:
    """TELEA-inpaint ``hole_mask``; only those pixels change. The inpaint mask
    also covers the no-coverage fringe beside a hole (restored afterwards), so
    TELEA never sources colour from black no-data.
    """
    import cv2

    if not hole_mask.any():
        return rgb.copy()
    fringe_r = INPAINT_RADIUS_PX + 2
    fringe_kernel = np.ones((2 * fringe_r + 1, 2 * fringe_r + 1), np.uint8)
    near_hole = cv2.dilate(hole_mask.astype(np.uint8), fringe_kernel).astype(bool)
    inpaint_mask = (hole_mask | (no_coverage_mask & near_hole)).astype(np.uint8)

    filled = cv2.inpaint(
        np.ascontiguousarray(rgb),
        inpaint_mask,
        INPAINT_RADIUS_PX,
        cv2.INPAINT_TELEA,
    )
    keep = ~(inpaint_mask.astype(bool)) | hole_mask
    out = rgb.copy()
    out[keep] = filled[keep]
    return out


def _nearest_fill(layer: np.ndarray, covered: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill ``hole`` pixels of ``layer`` from the nearest ``covered`` pixel
    (for height, where interpolation would invent values)."""
    from scipy import ndimage

    if not hole.any():
        return layer
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    out = layer.copy()
    out[hole] = layer[iy, ix][hole]
    return out


def interior_holes(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``alpha == 0`` into (interior holes, frame-connected no-data) by
    4-connected components: one touching the frame edge is frame-connected."""
    from scipy import ndimage

    black = alpha == 0
    lbl, _ = ndimage.label(black)
    edge = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
    edge.discard(0)
    frame = np.isin(lbl, list(edge))
    hole = black & ~frame
    return hole, frame


def fill_ortho_holes(
    rgb: np.ndarray,
    height: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rgb, height, alpha)`` with every interior hole filled: ``rgb`` by
    TELEA, ``height`` from the nearest covered pixel, ``alpha`` 255.
    Frame-connected no-data stays black. Inputs are not mutated.
    """
    black = alpha == 0
    hole, no_coverage = interior_holes(alpha)
    covered = ~black

    filled_rgb = fill_in_coverage_holes(rgb, hole, no_coverage)
    filled_height = _nearest_fill(height, covered, hole)
    filled_alpha = alpha.copy()
    filled_alpha[hole] = 255
    return filled_rgb, filled_height, filled_alpha
