"""The pass every ortho perception site runs after ``detect``: per detection,
project the mask to tile cells, back-fill its ``world_contour`` and stamp the
M6 colour."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maps.perception.color import m6_attributes
from maps.perception.detection import DetectedObject
from maps.perception.geometry.geolocate import (
    TileFrame,
    project_mask_to_tile_cells_ortho,
    world_contour_from_region,
)


def project_and_stamp_ortho_detections(
    detections: Sequence[DetectedObject],
    *,
    view_corners_xy: tuple,
    rgb: np.ndarray,
    tile: TileFrame,
) -> dict[int, np.ndarray]:
    """Return ``{detection_index: cells}`` for projectable masks, each mask's
    ``(N, 2) int32`` (row, col) tile cells.

    Mutates each projectable detection (``SamDetectClient`` output: a mask and
    the client's median colour) in place: ``metadata["world_contour"]`` is
    traced from the mask and the M6 colour read over ``rgb``, the crop the
    masks were segmented on, replaces the client's colour.
    """
    tracked: dict[int, np.ndarray] = {}
    for i, det in enumerate(detections):
        region = det.mask
        cells = project_mask_to_tile_cells_ortho(region, view_corners_xy=view_corners_xy, tile=tile)
        if cells is None:
            continue
        wc = world_contour_from_region(region, view_corners_xy)
        if wc:
            det.metadata["world_contour"] = wc
        attrs = m6_attributes(np.asarray(rgb), region)
        if attrs:
            det.attributes.update(attrs)
            # stripped to best_view_px in _detection_to_mapobject
            det.attributes["color_px"] = str(region.area)
        tracked[i] = cells
    return tracked
