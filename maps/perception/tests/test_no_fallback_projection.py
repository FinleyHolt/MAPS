"""Depth-aware mask projection contract.

Covers ``project_mask_to_world`` and ``project_mask_to_tile_cells``:
real depth or no detection. No flat-ground / bearing-ray fallback.
"""

from __future__ import annotations

import math

import numpy as np

from maps.util.mask_region import MaskRegion

from maps.perception.ortho.camera_model import CameraIntrinsics
from maps.perception.geometry.geolocate import (
    TileFrame,
    project_mask_to_tile_cells,
    project_mask_to_world,
)
from maps.perception.geometry.sensor_spec import SensorSpec


def _sensor(
    *,
    mount_pitch_deg: float = 90.0,
    max_range_m: float = 200.0,
    image_width: int = 64,
    image_height: int = 48,
) -> SensorSpec:
    return SensorSpec(
        name="test_proj",
        intrinsics=CameraIntrinsics(
            hfov_deg=90.0,
            vfov_deg=60.0,
            image_width=image_width,
            image_height=image_height,
        ),
        mount_pitch_deg=mount_pitch_deg,
        max_range_m=max_range_m,
    )


def _mask_centered(H: int, W: int, half_extent: int = 6) -> np.ndarray:
    """Square mask centered on the image, ``(2*half_extent+1)^2`` pixels."""
    mask = np.zeros((H, W), dtype=bool)
    cy, cx = H // 2, W // 2
    mask[
        cy - half_extent : cy + half_extent + 1, cx - half_extent : cx + half_extent + 1
    ] = True
    return mask


def test_constant_depth_yields_real_world_xy():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    altitude = 50.0
    depth = np.full((H, W), altitude, dtype=np.float32)
    mask = _mask_centered(H, W, half_extent=6)
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert projected is not None
    world_xy = projected.world_xy
    assert math.isfinite(world_xy[0]) and math.isfinite(world_xy[1])
    # The projected mask carries enough valid pixels to clear the
    # min_valid_depth_pixels threshold (13x13 mask = 169 pixels).
    assert projected.n_valid_pixels >= 169


def test_all_nan_depth_drops_detection():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    depth = np.full((H, W), float("nan"), dtype=np.float32)
    mask = _mask_centered(H, W, half_extent=6)
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert projected is None


def test_missing_depth_drops_detection():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    mask = _mask_centered(H, W, half_extent=6)
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=None,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert projected is None


def test_partial_valid_depth_uses_valid_pixels_only():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    altitude = 50.0
    # Half-valid depth: left half finite, right half NaN.
    depth = np.full((H, W), float("nan"), dtype=np.float32)
    depth[:, : W // 2] = altitude
    mask = np.ones((H, W), dtype=bool)
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert projected is not None
    world_xy = projected.world_xy
    # World-frame for nadir + yaw=0: image-col grows east, so the left
    # half of the image lies west of the pose (world_x <= 0).
    assert world_xy[0] < 0.0  # centroid sits west of the pose


def test_below_threshold_drops_detection():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    altitude = 50.0
    # Single-pixel mask: clearly below default threshold of 16.
    depth = np.full((H, W), altitude, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    mask[H // 2, W // 2] = True
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
        min_valid_depth_pixels=16,
    )
    assert projected is None


def test_oblique_sensor_with_valid_depth_returns_finite_world_xy():
    """The nadir special case is gone; oblique pitches work the same way."""
    sensor = _sensor(mount_pitch_deg=45.0)
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    # Constant 30 m forward-distance — well within max_range.
    depth = np.full((H, W), 30.0, dtype=np.float32)
    mask = _mask_centered(H, W, half_extent=6)
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert projected is not None
    world_xy = projected.world_xy
    assert math.isfinite(world_xy[0]) and math.isfinite(world_xy[1])
    assert projected.n_valid_pixels > 0


def test_beyond_max_range_drops_detection():
    """Depth > sensor.max_range_m is masked out in project_pixels_to_world,
    so a mask whose every pixel is beyond range has zero valid samples
    and drops."""
    sensor = _sensor(max_range_m=100.0)
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    depth = np.full((H, W), 500.0, dtype=np.float32)  # all beyond range
    mask = _mask_centered(H, W, half_extent=6)
    projected = project_mask_to_world(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert projected is None


def test_tile_cells_rasterises_to_real_cells():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    altitude = 50.0
    depth = np.full((H, W), altitude, dtype=np.float32)
    mask = _mask_centered(H, W, half_extent=8)
    tile = TileFrame(x_min=-200.0, y_min=-200.0, cell_size_m=1.0, grid_shape=(400, 400))
    tp = project_mask_to_tile_cells(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
        tile=tile,
    )
    assert tp is not None
    assert tp.cell_count > 0


def test_tile_cells_drops_when_depth_missing():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    mask = _mask_centered(H, W, half_extent=8)
    tile = TileFrame(x_min=-200.0, y_min=-200.0, cell_size_m=1.0, grid_shape=(400, 400))
    tp = project_mask_to_tile_cells(
        MaskRegion.from_full(mask),
        depth=None,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
        tile=tile,
    )
    assert tp is None


def test_tile_cells_out_of_bounds_returns_none():
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    altitude = 50.0
    depth = np.full((H, W), altitude, dtype=np.float32)
    mask = _mask_centered(H, W, half_extent=8)
    # Tile far away from the projection footprint.
    tile = TileFrame(x_min=5000.0, y_min=5000.0, cell_size_m=1.0, grid_shape=(50, 50))
    tp = project_mask_to_tile_cells(
        MaskRegion.from_full(mask),
        depth=depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
        tile=tile,
    )
    assert tp is None
