"""Tests for ``perception.geolocate.project_pixels_to_world``.

Covers the depth-aware per-pixel back-projection added to support
oblique-mount facade extraction. Validates the perspective unprojection
math against analytic ground truth at three poses (nadir flat, forward
horizontal, forward yawed) and the NaN handling on invalid depth.
"""

from __future__ import annotations

import math

import numpy as np

from maps.perception.geometry.geolocate import project_pixels_to_world
from maps.perception.geometry.sensor_spec import CameraIntrinsics, SensorSpec


def _sensor(*, mount_pitch_deg: float, max_range_m: float = 200.0) -> SensorSpec:
    return SensorSpec(
        name="test_depth",
        intrinsics=CameraIntrinsics(hfov_deg=90.0, vfov_deg=60.0, image_width=64, image_height=48),
        mount_pitch_deg=mount_pitch_deg,
        max_range_m=max_range_m,
    )


def test_nadir_constant_depth_lands_on_ground_plane():
    sensor = _sensor(mount_pitch_deg=90.0)
    altitude = 50.0
    H = sensor.intrinsics.image_height
    W = sensor.intrinsics.image_width
    depth = np.full((H, W), altitude, dtype=np.float32)

    xyz = project_pixels_to_world(
        depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
    )
    # Every pixel rays z_cam = altitude, so world_z = altitude + body_z = 0.
    assert xyz.shape == (H, W, 3)
    assert np.allclose(xyz[..., 2], 0.0, atol=1e-3)
    # Center pixel maps to (0, 0).
    cy_idx, cx_idx = H // 2, W // 2
    assert math.isclose(float(xyz[cy_idx, cx_idx, 0]), 0.0, abs_tol=1.0)
    assert math.isclose(float(xyz[cy_idx, cx_idx, 1]), 0.0, abs_tol=1.0)


def test_forward_facing_center_pixel_projects_north_at_depth():
    sensor = _sensor(mount_pitch_deg=0.0)
    H = sensor.intrinsics.image_height
    W = sensor.intrinsics.image_width
    depth = np.full((H, W), 10.0, dtype=np.float32)
    xyz = project_pixels_to_world(
        depth,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
    )
    cy_idx, cx_idx = H // 2, W // 2
    # Center pixel of a horizontal camera at yaw=0 (facing north): 10 m north.
    assert math.isclose(float(xyz[cy_idx, cx_idx, 0]), 0.0, abs_tol=0.5)
    assert math.isclose(float(xyz[cy_idx, cx_idx, 1]), 10.0, abs_tol=0.5)
    assert math.isclose(float(xyz[cy_idx, cx_idx, 2]), 50.0, abs_tol=0.5)


def test_forward_facing_yaw90_center_pixel_projects_east():
    sensor = _sensor(mount_pitch_deg=0.0)
    H = sensor.intrinsics.image_height
    W = sensor.intrinsics.image_width
    depth = np.full((H, W), 10.0, dtype=np.float32)
    xyz = project_pixels_to_world(
        depth,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=90.0,
        sensor=sensor,
    )
    cy_idx, cx_idx = H // 2, W // 2
    # Facing east now.
    assert math.isclose(float(xyz[cy_idx, cx_idx, 0]), 10.0, abs_tol=0.5)
    assert math.isclose(float(xyz[cy_idx, cx_idx, 1]), 0.0, abs_tol=0.5)


def test_oblique_sloped_depth_recovers_z_band():
    """45 deg down-forward mount with depth = altitude / sin(45) lands z=0."""
    sensor = _sensor(mount_pitch_deg=45.0)
    altitude = 10.0
    s = math.sin(math.radians(45.0))
    depth_val = altitude / s
    H = sensor.intrinsics.image_height
    W = sensor.intrinsics.image_width
    depth = np.full((H, W), depth_val, dtype=np.float32)
    xyz = project_pixels_to_world(
        depth,
        pose_xy=(0.0, 0.0),
        altitude_m=altitude,
        yaw_deg=0.0,
        sensor=sensor,
    )
    cy_idx, cx_idx = H // 2, W // 2
    # Center ray goes 45 deg down-forward; lands on z=0.
    assert math.isclose(float(xyz[cy_idx, cx_idx, 2]), 0.0, abs_tol=0.5)
    # Lands north (forward at yaw=0) of pose by altitude/tan(45) = altitude.
    assert math.isclose(float(xyz[cy_idx, cx_idx, 1]), altitude, abs_tol=0.5)


def test_invalid_depth_becomes_nan():
    sensor = _sensor(mount_pitch_deg=90.0, max_range_m=100.0)
    H = sensor.intrinsics.image_height
    W = sensor.intrinsics.image_width
    depth = np.full((H, W), 50.0, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = -1.0
    depth[0, 2] = np.nan
    depth[0, 3] = 200.0  # beyond max_range
    xyz = project_pixels_to_world(
        depth,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        sensor=sensor,
    )
    assert np.all(np.isnan(xyz[0, 0]))
    assert np.all(np.isnan(xyz[0, 1]))
    assert np.all(np.isnan(xyz[0, 2]))
    assert np.all(np.isnan(xyz[0, 3]))
    # Valid pixels stay finite.
    assert np.all(np.isfinite(xyz[1:, :]))
