"""Geometry pins for the nadir camera-frame render.

The two things that can silently go wrong are the footprint arithmetic (an
altitude that maps to the wrong ground rectangle) and the world -> ortho-pixel
mapping (a frame centred somewhere other than the aim point, or flipped in y).
Both are checked here against a synthetic ortho with a marker at a known world
coordinate -- no dataset, no fixtures.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest

from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.ortho.sensor_frame import (
    footprint_corners_xy,
    footprint_m,
    render_frame,
)

ORYX = resolve_sensor("oryx_orx32_25mm_nadir")


class Bounds(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def test_footprint_is_two_agl_tan_half_fov():
    agl = 457.2  # 1500 ft
    w, h = footprint_m(ORYX, agl)
    assert w == pytest.approx(2 * agl * math.tan(math.radians(ORYX.intrinsics.hfov_deg) / 2), rel=1e-12)
    assert h == pytest.approx(2 * agl * math.tan(math.radians(ORYX.intrinsics.vfov_deg) / 2), rel=1e-12)
    # The values the Edwards site is specified against (the 1/1.8" IMX252
    # behind the 25 mm lens), so a sensor-card edit that moves them shows up here.
    assert (round(w, 1), round(h, 1)) == (129.2, 96.9)
    assert w / ORYX.intrinsics.image_width == pytest.approx(0.0631, abs=5e-5)


def test_corners_are_north_up_at_yaw_zero():
    fl, fr, br, bl = footprint_corners_xy((100.0, 200.0), 40.0, 20.0, 0.0)
    assert fl == pytest.approx((80.0, 210.0))  # image top-left  = NW
    assert fr == pytest.approx((120.0, 210.0))  # image top-right = NE
    assert br == pytest.approx((120.0, 190.0))
    assert bl == pytest.approx((80.0, 190.0))


def test_corners_rotate_clockwise_with_heading():
    """Heading 90 (east) swings 'front' to +x, so the image top edge faces east."""
    fl, fr, _, _ = footprint_corners_xy((0.0, 0.0), 40.0, 20.0, 90.0)
    assert fl == pytest.approx((10.0, 20.0))
    assert fr == pytest.approx((10.0, -20.0))


def _synthetic_ortho(px_per_m: float, bounds: Bounds, marker_xy: tuple[float, float]):
    """Black ortho with one white pixel at ``marker_xy`` (north-up, row 0 = y_max)."""
    w = round((bounds.x_max - bounds.x_min) * px_per_m)
    h = round((bounds.y_max - bounds.y_min) * px_per_m)
    rgb = np.zeros((h, w, 3), np.uint8)
    col = round((marker_xy[0] - bounds.x_min) * px_per_m)
    row = (h - 1) - round((marker_xy[1] - bounds.y_min) * px_per_m)
    rgb[row - 1 : row + 2, col - 1 : col + 2] = 255
    return rgb


@pytest.mark.parametrize("mount_yaw_deg, across, down", [(0.0, 0.75, 0.25), (180.0, 0.25, 0.75)])
def test_marker_lands_where_the_projection_predicts(mount_yaw_deg, across, down):
    """A camera bolted in facing aft mirrors the frame about its centre, and
    the renderer has to mirror with it or the twin disagrees with the
    projector that reads the real frames (the C-12's Oryx is installed so)."""
    from dataclasses import replace

    bounds = Bounds(0.0, 0.0, 1000.0, 1000.0)
    px_per_m, agl = 2.0, 457.2
    aim = (500.0, 500.0)
    sensor = replace(ORYX, mount_yaw_deg=mount_yaw_deg)
    # Offset the marker a quarter-frame east and an eighth-frame north of the aim
    # point, so a centred-but-flipped render would fail rather than pass by luck.
    w_m, h_m = footprint_m(sensor, agl)
    marker = (aim[0] + w_m / 4.0, aim[1] + h_m / 4.0)
    rgb = _synthetic_ortho(px_per_m, bounds, marker)

    frame = render_frame(
        rgb, bounds, px_per_m, sensor=sensor, pose_xy=aim, agl_m=agl, yaw_deg=0.0
    )
    assert frame.shape == (1536, 2048, 3)

    rows, cols = np.nonzero(frame.max(axis=2) > 128)
    assert rows.size, "marker did not survive the warp"
    # Facing forward a quarter-frame east is 3/4 across and a quarter-frame
    # north is 1/4 down; facing aft it is the mirror of both. The marker sits
    # on the ortho's own grid, so the render is read against where the grid
    # put it: up to a quarter metre off the nominal, several frame pixels here.
    way = 1.0 if mount_yaw_deg == 0.0 else -1.0
    on_grid = (bounds.x_min + round((marker[0] - bounds.x_min) * px_per_m) / px_per_m,
               bounds.y_min + round((marker[1] - bounds.y_min) * px_per_m) / px_per_m)
    assert cols.mean() == pytest.approx(2048 * across + way * (on_grid[0] - marker[0]) / (w_m / 2048), abs=2.0)
    assert rows.mean() == pytest.approx(1536 * down - way * (on_grid[1] - marker[1]) / (h_m / 1536), abs=2.0)


def test_out_of_block_footprint_fails_loud():
    bounds = Bounds(0.0, 0.0, 1000.0, 1000.0)
    rgb = np.zeros((2000, 2000, 3), np.uint8)
    with pytest.raises(ValueError, match="outside the block"):
        render_frame(
            rgb, bounds, 2.0, sensor=ORYX, pose_xy=(20.0, 500.0), agl_m=457.2, yaw_deg=0.0
        )


def test_ground_corners_come_off_the_projector_and_match_the_level_rectangle():
    """An aligned nadir sensor lands its corner pixel centres within a pixel
    of the level footprint rectangle's corners, and nowhere else."""
    from dataclasses import replace

    from maps.perception.geometry.sensor_spec import resolve_sensor
    from maps.perception.ortho.sensor_frame import footprint_corners_xy, footprint_m, ground_corners_xy

    # Aligned: the installed aft mount is checked at the end, and would
    # otherwise only relabel which corner is which.
    sensor = replace(resolve_sensor("oryx_orx32_25mm_nadir"), mount_yaw_deg=0.0)
    agl, pose, yaw = 762.0, (1000.0, 2000.0), 37.0
    w_m, h_m = footprint_m(sensor, agl)
    level = footprint_corners_xy(pose, w_m, h_m, yaw)
    ours = ground_corners_xy(sensor, pose, agl, yaw)
    gsd = max(w_m / sensor.intrinsics.image_width, h_m / sensor.intrinsics.image_height)
    for (x, y), (lx, ly) in zip(ours, level):
        assert math.hypot(x - lx, y - ly) < gsd, (x - lx, y - ly, gsd)
    # And in the right direction: the corner PIXEL CENTRES sit half a pixel
    # inside the rectangle's edges, so the footprint is exactly one pixel
    # narrower and shorter. An inverted sign would still pass the bound above.
    def side(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    assert side(ours[0], ours[1]) == pytest.approx(side(level[0], level[1]) - w_m / sensor.intrinsics.image_width, abs=1e-3)
    assert side(ours[1], ours[2]) == pytest.approx(side(level[1], level[2]) - h_m / sensor.intrinsics.image_height, abs=1e-3)

    # A mount pitch is a boresight the twin must render: the whole footprint
    # slides along track by the projector's own tan.
    pitched = replace(sensor, mount_pitch_deg=sensor.mount_pitch_deg + 1.0)
    moved = ground_corners_xy(pitched, pose, agl, yaw)
    shift = [((x2 - x1), (y2 - y1)) for (x1, y1), (x2, y2) in zip(ours, moved)]
    along = [dx * math.sin(math.radians(yaw)) + dy * math.cos(math.radians(yaw)) for dx, dy in shift]
    assert all(abs(abs(a) - agl * math.tan(math.radians(1.0))) < 0.5 for a in along), along
    assert len({round(a, 1) > 0 for a in along}) == 1, "the four corners moved in different directions"

    # Bolted in facing aft the camera sees the SAME ground, and names the
    # corners two apart: the footprint is the mount's business only through
    # which pixel looks at which corner of it.
    aft = ground_corners_xy(replace(sensor, mount_yaw_deg=180.0), pose, agl, yaw)
    for i, (x, y) in enumerate(aft):
        ox, oy = ours[(i + 2) % 4]
        assert math.hypot(x - ox, y - oy) < 1e-6, (i, x - ox, y - oy)


def test_the_window_under_the_footprint_renders_the_whole_ortho_s_frame():
    """The frame is what the whole raster would give, byte for byte.

    Only the window under the footprint is warped, because ``remap`` refuses
    a source of 32,767 px or more and a block-wide ortho passes that at
    0.3 m. The window is free only if it changes no pixel, so the render is
    compared against the same map run over the entire raster. Noise imagery,
    so a tap that moves by one pixel shows.
    """
    import cv2

    from maps.perception.ortho.sensor_frame import ground_corners_xy, source_map

    bounds, px_per_m = Bounds(0.0, 0.0, 1000.0, 1000.0), 2.0
    rgb = np.random.default_rng(7).integers(0, 256, (2000, 2000, 3), dtype=np.uint8)
    for pose, agl, yaw, roll, pitch in (
        ((500.0, 500.0), 457.2, 0.0, 0.0, 0.0),
        ((300.0, 700.0), 610.0, 37.0, 12.0, -4.0),
        ((620.0, 410.0), 300.0, 250.0, -20.0, 6.0),
    ):
        frame = render_frame(
            rgb, bounds, px_per_m, sensor=ORYX, pose_xy=pose, agl_m=agl, yaw_deg=yaw,
            airframe_roll_deg=roll, airframe_pitch_deg=pitch,
        )
        corners = ground_corners_xy(ORYX, pose, agl, yaw, airframe_roll_deg=roll,
                                    airframe_pitch_deg=pitch)
        src = np.array(
            [
                [(x - bounds.x_min) * px_per_m, (rgb.shape[0] - 1) - (y - bounds.y_min) * px_per_m]
                for (x, y) in corners
            ],
            dtype=np.float32,
        )
        map_x, map_y = source_map(src, ORYX.intrinsics.image_width, ORYX.intrinsics.image_height)
        whole = cv2.remap(
            rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        assert np.array_equal(frame, whole)
        # The window was a window: a footprint that spanned the raster would
        # make the comparison above vacuous.
        assert np.ptp(map_x) < 0.4 * rgb.shape[1] and np.ptp(map_y) < 0.4 * rgb.shape[0]
