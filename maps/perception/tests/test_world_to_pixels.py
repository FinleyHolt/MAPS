"""``project_world_to_pixels`` is the exact inverse of ``project_pixels_to_world``.

"Exact" is the whole claim: the calibration puts its residual in pixels, so
any error here is indistinguishable from a real boresight misalignment and
would be absorbed silently into the fitted angles. These tests round-trip
through both directions and demand sub-pixel agreement, at nadir and off it,
level and tilted, at every heading.
"""

from __future__ import annotations

import numpy as np
import pytest

from maps.perception.geometry.geolocate import (
    project_pixels_to_world,
    project_world_to_pixels,
)
from maps.perception.geometry.sensor_spec import SensorSpec, resolve_sensor
from maps.perception.ortho.camera_model import CameraIntrinsics


def _sensor(*, max_range_m: float = float("inf")) -> SensorSpec:
    """A wide test camera: 90x60 deg puts real off-nadir angle in the corners,
    where a rotation error and a focal error diverge."""
    return SensorSpec(
        name="test_w2p",
        intrinsics=CameraIntrinsics(
            hfov_deg=90.0, vfov_deg=60.0, image_width=64, image_height=48
        ),
        mount_pitch_deg=90.0,
        max_range_m=max_range_m,
    )


def _round_trip(sensor: SensorSpec, depth_value: float, **pose_kw):
    """``(max |pixel error|, the float32 floor it must beat)``.

    ``project_pixels_to_world`` builds its own pixel grid at cell centers, so
    the expected answer is exactly that grid -- a test of the inverse alone,
    with no tolerance stacked in from a hand-built expectation.

    It also returns its world points as **float32**, a deliberate memory
    choice for a full-frame runtime buffer. That quantises position to
    ``R * eps32`` at a world coordinate of magnitude ``R``, and the inverse
    turns position back into pixels at ``fx / depth``. So no round trip can
    beat that product however exact the algebra is, and a fixed tolerance
    would either be slack near the origin or fail at a real tile coordinate
    (Cal City runs to ~5 km, where the floor is 8e-3 px). Deriving the bound
    keeps the test tight everywhere: a transposed matrix or a flipped
    heading sign misses by whole pixels, four orders of magnitude clear of
    this.
    """
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    depth = np.full((H, W), depth_value, dtype=np.float64)
    world = project_pixels_to_world(depth, sensor=sensor, **pose_kw)
    back = project_world_to_pixels(world.astype(np.float64), sensor=sensor, **pose_kw)

    us = np.arange(W, dtype=np.float64) + 0.5
    vs = np.arange(H, dtype=np.float64) + 0.5
    u_grid, v_grid = np.meshgrid(us, vs)
    expected = np.stack([u_grid, v_grid], axis=-1)

    assert np.all(np.isfinite(back)), "a valid forward projection came back NaN"

    quantum_m = float(np.nanmax(np.abs(world))) * float(np.finfo(np.float32).eps)
    floor_px = 4.0 * quantum_m * sensor.intrinsics.fx / depth_value
    return float(np.abs(back - expected).max()), floor_px


def test_pinhole_relation_holds_independently():
    """Pinned against closed form, not against the forward pass.

    Every other test here is a round trip, which agrees with itself even if
    both directions share a wrong convention. This one states the answer
    independently: for a nadir camera at height ``h``, a ground point ``d``
    meters from the nadir point images exactly ``f*d/h`` pixels from the
    principal point. Square pixels so one focal length covers both axes.
    """
    sensor = SensorSpec(
        name="test_square",
        intrinsics=CameraIntrinsics(
            hfov_deg=90.0, vfov_deg=90.0, image_width=64, image_height=64
        ),
        mount_pitch_deg=90.0,
        max_range_m=float("inf"),
    )
    f = sensor.intrinsics.fx
    assert f == pytest.approx(sensor.intrinsics.fy), "test sensor is not square"
    h, d = 100.0, 10.0
    pose = dict(pose_xy=(0.0, 0.0), altitude_m=h, yaw_deg=0.0, sensor=sensor)
    center = np.array([sensor.intrinsics.cx, sensor.intrinsics.cy])

    below = project_world_to_pixels(np.array([0.0, 0.0, 0.0]), **pose)
    assert below == pytest.approx(center, abs=1e-9)

    north = project_world_to_pixels(np.array([0.0, d, 0.0]), **pose)
    east = project_world_to_pixels(np.array([d, 0.0, 0.0]), **pose)
    expected_px = f * d / h
    assert np.linalg.norm(north - center) == pytest.approx(expected_px, rel=1e-9)
    assert np.linalg.norm(east - center) == pytest.approx(expected_px, rel=1e-9)

    # North and east must land on perpendicular image axes. A collapsed
    # rotation would put both on the same one and still pass the magnitudes.
    assert np.dot(north - center, east - center) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("yaw_deg", [0.0, 37.0, 90.0, 180.0, 270.0, 359.5, -45.0])
def test_round_trip_every_heading(yaw_deg):
    """The heading rotation inverts to itself. If that involution were wrong it
    would show as a heading-dependent error, which is exactly the shape of a
    boresight yaw -- so a bug here would be fitted rather than caught."""
    err, floor = _round_trip(
        _sensor(), 250.0, pose_xy=(1234.5, -678.9), altitude_m=250.0, yaw_deg=yaw_deg
    )
    assert err < floor, f"yaw={yaw_deg}: {err} px, float32 floor {floor} px"


@pytest.mark.parametrize(
    ("roll_deg", "pitch_deg"), [(0.0, 0.0), (12.0, 0.0), (0.0, -7.0), (-25.0, 4.0)]
)
def test_round_trip_airframe_tilt(roll_deg, pitch_deg):
    """Banked and pitched frames must invert too. The calibration passes are
    flown wings-level, but the vision correction runs on every frame."""
    err, floor = _round_trip(
        _sensor(),
        400.0,
        pose_xy=(10.0, 20.0),
        altitude_m=400.0,
        yaw_deg=113.0,
        airframe_roll_deg=roll_deg,
        airframe_pitch_deg=pitch_deg,
    )
    assert err < floor, f"roll={roll_deg} pitch={pitch_deg}: {err} px, floor {floor}"


def test_round_trip_oryx_at_band_altitudes():
    """The real C-12 sensor at both TMP altitudes, 1,500 and 3,000 ft AGL, at a
    Cal City tile coordinate rather than a convenient origin."""
    oryx = resolve_sensor("oryx_orx32_25mm_nadir")
    for agl_m in (457.2, 914.4):
        err, floor = _round_trip(
            oryx, agl_m, pose_xy=(4987.0, 2201.0), altitude_m=agl_m, yaw_deg=91.0
        )
        assert err < floor, f"agl={agl_m}: {err} px, float32 floor {floor} px"
        # The bound has to stay small enough to be worth asserting: 0.05 px at
        # 3,000 ft is 8 mm on the ground, against a >=1.6 m registration floor.
        assert floor < 0.05, f"agl={agl_m}: float32 floor {floor} px is too slack"


def test_off_nadir_corner_is_actually_off_nadir():
    """Vacuity floor. A round trip through an all-nadir geometry would pass
    even if the rotation were dropped entirely, so assert the test frame
    really does span meaningful off-nadir angle before trusting the round
    trips above."""
    sensor = _sensor()
    H, W = sensor.intrinsics.image_height, sensor.intrinsics.image_width
    depth = np.full((H, W), 100.0, dtype=np.float64)
    world = project_pixels_to_world(
        depth, sensor=sensor, pose_xy=(0.0, 0.0), altitude_m=100.0, yaw_deg=0.0
    )
    ground_offset = np.hypot(world[..., 0], world[..., 1])
    assert ground_offset.max() > 50.0, (
        f"corner is only {ground_offset.max():.1f} m from nadir at 100 m AGL; "
        "the frame does not exercise off-nadir geometry"
    )


def test_behind_the_camera_is_nan():
    """A world point above the camera cannot be imaged by a down-looking sensor.
    The inverse must refuse it rather than return the mirrored pixel a bare
    divide would produce -- an OSM footprint behind the aircraft would
    otherwise land a phantom polygon in the registration mask."""
    sensor = _sensor()
    behind = np.array([[0.0, 0.0, 500.0]])  # 400 m ABOVE a camera at 100 m
    uv = project_world_to_pixels(
        behind, sensor=sensor, pose_xy=(0.0, 0.0), altitude_m=100.0, yaw_deg=0.0
    )
    assert np.all(np.isnan(uv))


def test_beyond_max_range_is_nan():
    """The forward drops depth past ``max_range_m``; the inverse must agree, or
    a round trip disagrees about which points exist."""
    sensor = _sensor(max_range_m=150.0)
    far = np.array([[0.0, 0.0, -100.0]])  # 200 m below a camera at 100 m
    uv = project_world_to_pixels(
        far, sensor=sensor, pose_xy=(0.0, 0.0), altitude_m=100.0, yaw_deg=0.0
    )
    assert np.all(np.isnan(uv))


def test_accepts_any_leading_shape():
    """Callers hand it a flat list of control points, a polygon ring, or a
    grid. Only the last axis is fixed."""
    sensor = _sensor()
    pose = dict(pose_xy=(0.0, 0.0), altitude_m=100.0, yaw_deg=0.0, sensor=sensor)
    single = project_world_to_pixels(np.array([0.0, 10.0, 0.0]), **pose)
    flat = project_world_to_pixels(np.zeros((7, 3)), **pose)
    grid = project_world_to_pixels(np.zeros((3, 4, 3)), **pose)
    assert single.shape == (2,)
    assert flat.shape == (7, 2)
    assert grid.shape == (3, 4, 2)


def test_rejects_wrong_last_dimension():
    sensor = _sensor()
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        project_world_to_pixels(
            np.zeros((5, 2)),
            pose_xy=(0.0, 0.0),
            altitude_m=100.0,
            yaw_deg=0.0,
            sensor=sensor,
        )
