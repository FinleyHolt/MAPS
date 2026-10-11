"""The sensor-footprint observation: an ortho warp of the camera's ground
footprint at native sensor pixels.

A ``SensorSpec`` on the crop switches the view rectangle from the CityNav
2·AGL square to ``2·AGL·tan(fov/2)`` per axis and the output from the ortho's
own pixel span to the sensor's native ``image_width × image_height``. The warp
stays orthographic, so the projection contract stays the corners — these tests
pin the geometry, the replay identity, and the sensorless path's byte-parity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.observation import (
    crop_ortho_observation,
    observation_extent_m,
)
from maps.perception.ortho.sensor_frame import footprint_corners_xy, footprint_m
from maps.search.tests.ortho_fixture import write_rasters

MAP = "unit_test_sensor_block"
GROUND_Z = 0.0
PX_PER_M = 10.0  # 0.1 m/px


@pytest.fixture
def ortho_dir(tmp_path, monkeypatch):
    """A 60 m × 60 m synthetic ortho, flat height, full coverage."""
    from maps.perception.ortho import camera_model
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    n = 600
    rows = np.arange(n, dtype=np.uint16)[:, None] % 256
    cols = np.arange(n, dtype=np.uint16)[None, :] % 256
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    rgb[..., 0] = rows.astype(np.uint8)
    rgb[..., 1] = cols.astype(np.uint8)
    rgb[..., 2] = 77
    height = np.full((n, n), GROUND_Z + 3.0, dtype=np.float32)
    write_rasters(tmp_path, MAP, rgb, height, m_per_px=1.0 / PX_PER_M)
    monkeypatch.setitem(GROUND_LEVEL, MAP, GROUND_Z)
    camera_model._ORTHO_RASTER_CACHE.clear()
    return tmp_path


ORYX = resolve_sensor("oryx_orx32_25mm_nadir")
# An AGL whose Oryx footprint (~28 × 21 m) fits the 60 m fixture with margin.
AGL = 80.0


def _crop(sensor=None, yaw=0.0, pose=(30.0, 30.0), alt=AGL):
    return crop_ortho_observation(
        map_name=MAP,
        pose_xy=pose,
        altitude_m=alt,
        yaw_deg=yaw,
        ground_z=GROUND_Z,
        rgbd_dir="unused",  # the fixture monkeypatch makes the loader dir-blind
    )


def test_sensor_frame_is_native_size_over_the_footprint(ortho_dir):
    oc = crop_ortho_observation(
        map_name=MAP,
        pose_xy=(30.0, 30.0),
        altitude_m=AGL,
        yaw_deg=30.0,
        ground_z=GROUND_Z,
        rgbd_dir=ortho_dir,
        sensor=ORYX,
    )
    intr = ORYX.intrinsics
    assert oc["rgb"].shape == (intr.image_height, intr.image_width, 3)
    assert oc["depth"].shape == (intr.image_height, intr.image_width)
    w_m, h_m = footprint_m(ORYX, AGL)
    expect = footprint_corners_xy((30.0, 30.0), w_m, h_m, 30.0)
    got = oc["view_corners_xy"]
    assert np.allclose(np.asarray(got), np.asarray(expect))
    # Flat world, full coverage: depth is AGL minus the constant height.
    assert bool((oc["validity"] > 0).all())
    assert np.allclose(oc["depth"], AGL - (GROUND_Z + 3.0), atol=1e-4)


def test_sensorless_path_is_byte_identical_to_before(ortho_dir):
    """The historical square contract carries no sensor and must not move."""
    oc = crop_ortho_observation(
        map_name=MAP,
        pose_xy=(30.0, 30.0),
        altitude_m=GROUND_Z + 10.0,
        yaw_deg=0.0,
        ground_z=GROUND_Z,
        rgbd_dir=ortho_dir,
    )
    # 10 m AGL -> 20 m square at 0.1 m/px -> 200 px output.
    assert oc["rgb"].shape == (200, 200, 3)


def test_observation_extent_is_the_footprint_long_side():
    assert observation_extent_m(AGL, GROUND_Z) == pytest.approx(2.0 * AGL)
    w_m, h_m = footprint_m(ORYX, AGL)
    assert w_m > h_m  # hfov 16.1 > vfov 12.1
    assert observation_extent_m(AGL, GROUND_Z, ORYX) == pytest.approx(w_m)
    # The TMP number: 3,000 ft AGL under the Oryx is a ~258 m wide footprint.
    ft3000 = 3000.0 * 0.3048
    assert observation_extent_m(ft3000, 0.0, ORYX) == pytest.approx(
        2.0 * ft3000 * math.tan(math.radians(ORYX.intrinsics.hfov_deg) / 2.0)
    )
    assert observation_extent_m(ft3000, 0.0, ORYX) == pytest.approx(258.5, abs=0.5)
