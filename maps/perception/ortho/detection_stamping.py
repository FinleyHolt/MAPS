"""The pass every ortho perception site runs after ``detect``: per detection,
project the mask to tile cells, back-fill its ``world_contour`` and stamp the
M6 colour."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maps.perception.base import DetectedObject
from maps.perception.geometry.geolocate import (
    TileFrame,
    TileProjectedMask,
    project_mask_to_tile_cells_ortho,
    world_contour_from_region,
)
from maps.perception.viz.color import m6_attributes


def project_and_stamp_ortho_detections(
    detections: Sequence[DetectedObject],
    *,
    view_corners_xy: tuple,
    rgb: np.ndarray | None,
    tile: TileFrame,
) -> dict[int, TileProjectedMask]:
    """Return ``{detection_index: TileProjectedMask}`` for projectable masks.

    Mutates the detections in place: ``metadata["world_contour"]`` and the
    ``color*`` attributes are stamped per the rules below. ``rgb`` is the
    crop the masks were segmented on; ``None`` disables color stamping.
    """
    tracked: dict[int, TileProjectedMask] = {}
    for i, det in enumerate(detections):
        if det.mask is None:
            continue
        region = det.mask
        tp = project_mask_to_tile_cells_ortho(
            region,
            view_corners_xy=view_corners_xy,
            tile=tile,
        )
        if tp is not None and not det.metadata.get("world_contour"):
            wc = world_contour_from_region(region, view_corners_xy)
            if wc:
                det.metadata["world_contour"] = wc
        # M6 replaces the adapter's single-word colour (the one carrying
        # color_rgb) and never a colour set without color_rgb.
        if (
            tp is not None
            and rgb is not None
            and (
                "color" not in det.attributes
                or "color_rgb" in det.attributes
            )
        ):
            attrs = m6_attributes(np.asarray(rgb), region)
            if attrs:
                det.attributes.update(attrs)
                # stripped to best_view_px in _detection_to_mapobject
                det.attributes["color_px"] = str(region.area)
        if tp is not None:
            tracked[i] = tp
    return tracked
