"""Contract for the shared ortho paint pass (``detection_stamping``).

Pins the gates both call sites (the sweep's Phase A and the
retrospective re-perception) rely on: projection keys for projectable
masks only, ``world_contour``/color stamped, the
``color``/``color_rgb`` supersession rule, and the ``color_conf``
format.
"""

from __future__ import annotations

import numpy as np

from maps.util.mask_region import MaskRegion

from maps.perception.base import DetectedObject
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
        mask=MaskRegion.from_full(mask),
    )


def _rgb() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def test_projects_only_maskful_detections():
    dets = [
        _det("sam"),
        DetectedObject(world_xy=(0.0, 0.0), label="car", confidence=0.5, source_tier="sam"),
    ]
    tracked = project_and_stamp_ortho_detections(
        dets, view_corners_xy=_CORNERS, rgb=_rgb(), tile=_TILE
    )
    assert set(tracked) == {0}
    assert tracked[0].rows.size > 0


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


def test_higher_fidelity_color_never_overwritten():
    # A color WITHOUT color_rgb is a higher-fidelity fill (GT table /
    # vision) — the mixture pass must not touch it.
    det = _det("sam", attributes={"color": "cyan"})
    project_and_stamp_ortho_detections(
        [det], view_corners_xy=_CORNERS, rgb=_rgb(), tile=_TILE
    )
    assert det.attributes["color"] == "cyan"
    assert "color_conf" not in det.attributes


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


def test_no_rgb_disables_color_but_not_contour():
    det = _det("sam")
    project_and_stamp_ortho_detections(
        [det], view_corners_xy=_CORNERS, rgb=None, tile=_TILE
    )
    assert det.metadata.get("world_contour")
    assert "color" not in det.attributes
