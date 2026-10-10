"""Pin the nadir ortho view-square geometry (``view_square_corners_xy``).

Replaces the deleted upstream ``verify_citynav_native_geometry`` due-north gate.
The corner order ``[FL, FR, BR, BL]`` and yaw convention (0 = North, CW-positive)
are what ``geolocate.project_mask_to_world_ortho`` inverts, so a regression here
silently mislocates every detection.
"""

import numpy as np

from maps.perception.observation import view_square_corners_xy


def test_due_north_corners():
    # At yaw=0 (facing North) with half-side 100 m about (500, 500):
    # FL = North-West, FR = North-East, BR = South-East, BL = South-West.
    fl, fr, br, bl = view_square_corners_xy((500.0, 500.0), 100.0, 0.0)
    np.testing.assert_allclose(fl, (400.0, 600.0), atol=1e-6)  # NW
    np.testing.assert_allclose(fr, (600.0, 600.0), atol=1e-6)  # NE
    np.testing.assert_allclose(br, (600.0, 400.0), atol=1e-6)  # SE
    np.testing.assert_allclose(bl, (400.0, 400.0), atol=1e-6)  # SW


def test_facing_east_rotates_cw():
    # yaw=90 faces East: the front edge swings to the East side.
    fl, fr, br, bl = view_square_corners_xy((0.0, 0.0), 10.0, 90.0)
    np.testing.assert_allclose(fl, (10.0, 10.0), atol=1e-6)  # front-left = NE
    np.testing.assert_allclose(fr, (10.0, -10.0), atol=1e-6)  # front-right = SE
    np.testing.assert_allclose(br, (-10.0, -10.0), atol=1e-6)  # back-right = SW
    np.testing.assert_allclose(bl, (-10.0, 10.0), atol=1e-6)  # back-left = NW


def test_world_window_is_symmetric_square():
    corners = view_square_corners_xy((123.0, 456.0), 50.0, 37.0)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    # A rotated square's axis-aligned bbox is centered on the pose.
    assert abs((min(xs) + max(xs)) / 2 - 123.0) < 1e-6
    assert abs((min(ys) + max(ys)) / 2 - 456.0) < 1e-6
