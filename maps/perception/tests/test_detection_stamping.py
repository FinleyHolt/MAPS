"""Contract for the shared ortho paint pass (``detection_stamping``).

Pins what both call sites (a flight and the retrospective re-perception)
rely on: ``world_contour`` and the M6 colour stamped, the client's median
colour superseded, and the ``color_conf`` format.
"""

from __future__ import annotations

import numpy as np

from maps.perception.tests.regions import region_of

from maps.perception.detection import DetectedObject
from maps.perception.geometry.geolocate import TileFrame
from maps.perception.ortho.detection_stamping import (
    project_and_stamp_ortho_detections,
)

# 64x64 crop over a 100 m axis-aligned view square on a 2 m tile grid.
_CORNERS = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
_TILE = TileFrame(x_min=0.0, y_min=0.0, cell_size_m=2.0, grid_shape=(64, 64))


def _det(source_tier: str, attributes: dict | None = None) -> DetectedObject:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    return DetectedObject(
        world_xy=(0.0, 0.0),
        label="car",
        confidence=0.9,
        source_tier=source_tier,
        attributes=dict(attributes or {}),
        mask=region_of(mask),
    )


def _rgb() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def test_real_tier_gets_contour_and_color():
    det = _det("sam")
    project_and_stamp_ortho_detections(
        [det], view_corners_xy=_CORNERS, rgb=_rgb(), tile=_TILE
    )
    assert det.metadata.get("world_contour"), "real tier must get a contour"
    assert "color" in det.attributes
    # color_conf is stamped as a 3-decimal string.
    float(det.attributes["color_conf"])
    assert len(det.attributes["color_conf"].split(".")[1]) == 3
    assert det.attributes["color_px"] == str(20 * 20)


def test_adapter_median_color_is_superseded():
    # color WITH color_rgb is the adapter's single-word median — same
    # fidelity class, so the mixture pass replaces it.
    det = _det(
        "sam", attributes={"color": "mud", "color_rgb": "1,2,3"}
    )
    project_and_stamp_ortho_detections(
        [det], view_corners_xy=_CORNERS, rgb=_rgb(), tile=_TILE
    )
    assert det.attributes["color"] != "mud"
    assert "color_conf" in det.attributes
