"""Validity-masked interior hole fill for baked 2D orthomosaics.

A top-down ortho baked from a point cloud (``camera_model.bake_ortho_from_ply``)
or a PDAL rasterize leaves empty 0.1 m cells as ``alpha=0`` (rendered black).
Two kinds of no-data coexist:

- **interior speckle holes** — cells enclosed by real coverage (photogrammetry
  gaps under tree canopy, thin scan seams). These are safe to fill: TELEA
  inpaint sources only from the surrounding real texture.
- **boundary no-coverage** — water / glass / scan edges / the block's exterior
  frame. These are TRUE no-data; filling them projects invented texture toward
  the frame edge (long TELEA smears). They MUST stay black.

``fill_ortho_holes`` fills the interior holes and leaves boundary no-data black,
so the baked ortho has zero *interior* black while never inventing ground at a
frame boundary. The interior/boundary split is the erosion + connected-component
reconstruction in :func:`split_black_mask` (a black component is boundary iff it
contains an all-black ``NO_COVERAGE_ERODE_PX`` neighborhood).

The low-level recipe (``split_black_mask`` + ``fill_in_coverage_holes``) is the
single fill engine shared by the ortho bake and any raster consumer.
"""

from __future__ import annotations

import numpy as np

# A black connected component is classified as BOUNDARY no-coverage (true
# no-data: water/glass/scan edge/frame border) iff it survives erosion by this
# square kernel — i.e. it contains at least one all-black KxK neighborhood.
# Everything else black is an interior speckle hole. 15 px (1.5 m at 0.1 m/px).
NO_COVERAGE_ERODE_PX = 15

# TELEA inpaint neighborhood radius (px). Holes are dominated by 1-4 px
# components; 3 px reaches across them without long-range smearing.
INPAINT_RADIUS_PX = 3






def fill_in_coverage_holes(
    rgb: np.ndarray,
    hole_mask: np.ndarray,
    no_coverage_mask: np.ndarray,
) -> np.ndarray:
    """TELEA-inpaint the interior holes; leave boundary no-coverage black.

    The inpaint mask is the holes plus a thin fringe of no-coverage pixels
    bordering a hole, so TELEA never *sources* color from a black no-data
    neighbor (which would drag fills toward black at coverage boundaries). The
    fringe — and the whole no-coverage region — is restored to black afterwards,
    so no-coverage output is black and only ``hole_mask`` pixels differ from the
    input. Non-black input pixels are byte-identical.
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
    """Fill ``hole`` pixels of ``layer`` from the nearest ``covered`` pixel.

    Used for non-photometric channels (height, semantic class-ids) where TELEA's
    smooth interpolation is wrong (categorical labels) or gives a nonsense metric
    (a height of 0 under a filled canopy pixel). Nearest-neighbour keeps values
    exact and label-safe.
    """
    from scipy import ndimage

    if not hole.any():
        return layer
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    out = layer.copy()
    out[hole] = layer[iy, ix][hole]
    return out


def interior_holes(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``alpha==0`` no-data into (interior holes, frame-connected no-data).

    Interior = any black connected component NOT touching the frame edge (any
    size — a large enclosed canopy gap counts). Frame-connected = the exterior
    border plus any no-data that reaches an edge. This is the strict
    frame-connectivity policy: fill everything enclosed, so no fill can ever
    project to a frame boundary (no edge smear), and the filled ortho has zero
    *interior* black. Uses the same 4-connectivity as the interior-black audit.
    """
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
    semantic: np.ndarray | None = None,
):
    """Frame-connectivity hole fill for a baked ortho, keyed off ``alpha``.

    ``alpha == 0`` is the exact no-data mask. Every hole ENCLOSED by coverage
    (not connected to the frame edge — any size) is filled: ``rgb`` via TELEA
    inpaint, ``height`` and (optional) ``semantic`` via nearest-covered-pixel
    fill, ``alpha`` → 255. No-data that reaches the frame edge (the exterior
    border, water/scan edges opening to it) stays black with ``alpha`` 0, so no
    fill ever projects to a frame boundary. Result: zero interior black, no
    edge smear.

    Returns ``(rgb, height, alpha)`` or ``(rgb, height, alpha, semantic)`` when
    a semantic layer is supplied. Inputs are not mutated.
    """
    black = alpha == 0
    hole, no_coverage = interior_holes(alpha)
    covered = ~black

    filled_rgb = fill_in_coverage_holes(rgb, hole, no_coverage)
    filled_height = _nearest_fill(height, covered, hole)
    filled_alpha = alpha.copy()
    filled_alpha[hole] = 255

    if semantic is None:
        return filled_rgb, filled_height, filled_alpha
    filled_semantic = _nearest_fill(semantic, covered, hole)
    return filled_rgb, filled_height, filled_alpha, filled_semantic
