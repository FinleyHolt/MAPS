"""A declared view-up rotates a look crop, with the handedness pinned."""

from __future__ import annotations

from maps.render.look_render import render_candidate_crops
from maps.search.tests.fakes import ORTHO_MAP as MAP


def test_framed_crop_rotates(ortho_dir):
    # frame_up rides the request; the framed crop's pixels differ from the
    # unframed render of the identical window.
    req = {"center_fid": "c1", "center_xy": [60.0, 140.0],
           "center_label": "car c1", "center_boundary": None, "marks": []}
    base = render_candidate_crops([req], map_name=MAP, rgbd_dir=ortho_dir)[0]
    framed = render_candidate_crops(
        [{**req, "frame_up": [1.0, 0.0], "frame_note": "image-up = east"}],
        map_name=MAP, rgbd_dir=ortho_dir)[0]
    assert framed["frame_up"] == [1.0, 0.0]
    assert framed["frame_note"] == "image-up = east"
    assert framed["jpeg"] != base["jpeg"]


def test_frame_up_angle_sign_lock():
    # Pinned both ways like every handedness fact. The native crop is
    # north-up, so north (0, 1) is the identity; south-up is the 180 flip;
    # east-up is +90 — derived: world east's screen vector is (1, 0)
    # (screen x = world x, screen y = -world y), and the cv2 angle sending
    # screen (sx, sy) to image-up (0, -1) is atan2(sx, -sy) = atan2(ux, uy).
    # The cv2 matrix at +90 really sends the east screen-vector to
    # screen-up.
    import cv2

    from maps.render.look_render import frame_up_angle_deg
    assert abs(frame_up_angle_deg(0.0, 1.0)) < 1e-9
    assert abs(abs(frame_up_angle_deg(0.0, -1.0)) - 180.0) < 1e-9
    assert abs(frame_up_angle_deg(1.0, 0.0) - 90.0) < 1e-9
    M = cv2.getRotationMatrix2D((0.0, 0.0), frame_up_angle_deg(1.0, 0.0), 1.0)
    vx, vy = M[0, 0] * 1.0 + M[0, 1] * 0.0, M[1, 0] * 1.0 + M[1, 1] * 0.0
    assert abs(vx) < 1e-9 and abs(vy + 1.0) < 1e-9, (vx, vy)


def test_a_turned_crop_over_full_coverage_counts_no_missing_data(ortho_dir):
    # The black corners a turn leaves are outside the window, not raster
    # no-data: the valid fraction reads the window itself.
    req = {"center_fid": "c1", "center_xy": [100.0, 100.0],
           "center_label": "car c1", "center_boundary": None, "marks": [],
           "frame_up": [0.7071, 0.7071]}
    rec = render_candidate_crops([req], map_name=MAP, rgbd_dir=ortho_dir)[0]
    assert rec["window_valid_frac"] == 1.0, rec["window_valid_frac"]
