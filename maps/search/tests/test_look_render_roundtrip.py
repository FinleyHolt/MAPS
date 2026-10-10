"""A persisted look-crop record re-renders to the exact bytes the model saw.

The whole sha-verified audit doctrine in one test: ``render_candidate_crops``
resolves the window and delegates the pixels to ``render_look_from_spec``;
the record it returns (minus ``jpeg``) IS the manifest crop and IS the
re-render spec. Feeding that record back through ``render_look_from_spec`` —
the same function, the same env — must reproduce the jpeg byte-for-byte and
hash to the recorded ``sha256``. If this breaks, the dashboard's "verified
identical" badge is lying; bump ``LOOK_RENDERER_VERSION`` instead of ever
loosening the assert.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maps.search.render.look_render import (
    LOOK_RENDERER_VERSION,
    render_candidate_crops,
    render_look_from_spec,
)

MAP = "unit_test_look_roundtrip_block"
GROUND_Z = 10.0

# The re-render contract: every key a manifest crop record must carry for
# the dashboard route (spec + verification), pinned by name.
SPEC_KEYS = (
    "map_name", "center_fid", "center_label", "center_boundary", "marks",
    "world_xy", "intended_xy", "pose_xy", "context_m",
)
VERIFY_KEYS = ("sha256", "renderer_version", "renderer_env")


@pytest.fixture
def ortho_dir(tmp_path, monkeypatch):
    """A 200 m x 200 m synthetic ortho (same bake as the adaptive-window
    tests) with non-uniform pixels, so an off-by-one window shift changes
    bytes."""
    from maps.perception.ortho import camera_model
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    n = 2000
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    rgb[..., 0] = (np.arange(n, dtype=np.uint32)[:, None] % 256).astype(np.uint8)
    rgb[..., 1] = (np.arange(n, dtype=np.uint32)[None, :] % 256).astype(np.uint8)
    rgb[..., 2] = 96
    out = tmp_path / "ortho"
    out.mkdir()
    np.savez(
        out / f"{MAP}_ortho_m0.1.npz",
        rgb=rgb,
        height=np.full((n, n), GROUND_Z + 5.0, dtype=np.float32),
        alpha=np.full((n, n), 255, dtype=np.uint8),
        x_min=np.float64(0.0),
        y_min=np.float64(0.0),
        px_per_m=np.float64(10.0),
    )
    monkeypatch.setitem(GROUND_LEVEL, MAP, GROUND_Z)
    camera_model._ORTHO_RASTER_CACHE.clear()
    return tmp_path


def _requests() -> list[dict]:
    """A marked relational crop + a lone tight crop — both workflows."""
    boundary = [[96.0, 96.0], [104.0, 96.0], [104.0, 104.0], [96.0, 104.0]]
    return [
        {
            "center_fid": "c0",
            "center_xy": [100.0, 100.0],
            "center_label": "car c0",
            "center_boundary": boundary,
            "annotate": True,
            "marks": [
                {"fid": "m0", "xy": [88.0, 100.0], "boundary": None,
                 "label": "car m0", "is_target": False},
                {"fid": "tgt", "xy": [112.0, 100.0],
                 "boundary": [[110.0, 98.0], [114.0, 98.0], [114.0, 102.0]],
                 "label": "TARGET car", "is_target": True},
            ],
        },
        {
            "center_fid": "c1",
            "center_xy": [60.0, 140.0],
            "center_label": "car c1",
            "center_boundary": None,
            "annotate": False,
            "marks": [],
        },
    ]


def test_records_reproduce_their_bytes_and_carry_the_contract(ortho_dir):
    recs = render_candidate_crops(_requests(), map_name=MAP, rgbd_dir=ortho_dir)
    # Vacuity floor: the doctrine test proves nothing over zero crops.
    assert len(recs) == 2, recs

    for rec in recs:
        for key in SPEC_KEYS + VERIFY_KEYS:
            assert key in rec, f"crop record lost its {key!r}"
        assert rec["renderer_version"] == LOOK_RENDERER_VERSION
        assert rec["sha256"] == hashlib.sha256(rec["jpeg"]).hexdigest()

        # The round trip: the record (what the manifest persists) back
        # through the same renderer — byte-identical, sha-verified.
        spec = {k: v for k, v in rec.items() if k != "jpeg"}
        rerendered = render_look_from_spec(spec, rgbd_dir=ortho_dir)
        assert rerendered is not None
        assert rerendered["jpeg"] == rec["jpeg"]
        assert rerendered["sha256"] == rec["sha256"]


def test_a_malformed_spec_returns_none_not_a_raise(ortho_dir):
    assert render_look_from_spec(
        {"map_name": MAP, "pose_xy": [1.0, 2.0]}, rgbd_dir=ortho_dir) is None
    assert render_look_from_spec(
        {"map_name": "no_such_map", "pose_xy": [1.0, 2.0],
         "world_xy": [1.0, 2.0], "context_m": 15.0},
        rgbd_dir=ortho_dir) is None


def test_framed_crop_rotates_and_reproduces_bytes(ortho_dir):
    # The declared-view rotation: frame_up rides the spec, so the framed
    # crop honors the SAME re-render contract, and its pixels differ from
    # the unframed render of the identical window.
    reqs = _requests()[1:]  # the lone tight crop
    base = render_candidate_crops(reqs, map_name=MAP, rgbd_dir=ortho_dir)[0]
    reqs[0]["frame_up"] = [1.0, 0.0]   # up = east
    reqs[0]["frame_note"] = "image-up = east"
    framed = render_candidate_crops(reqs, map_name=MAP, rgbd_dir=ortho_dir)[0]
    assert framed["frame_up"] == [1.0, 0.0]
    assert framed["frame_note"] == "image-up = east"
    assert framed["sha256"] != base["sha256"]
    spec = {k: v for k, v in framed.items() if k != "jpeg"}
    again = render_look_from_spec(spec, rgbd_dir=ortho_dir)
    assert again["sha256"] == framed["sha256"], "framed re-render diverged"


def test_frame_up_angle_sign_lock():
    # Pinned both ways like every handedness fact. The native crop is
    # north-up, so north (0, 1) is the identity; south-up is the 180 flip;
    # east-up is +90 — derived: world east's screen vector is (1, 0)
    # (screen x = world x, screen y = -world y), and the cv2 angle sending
    # screen (sx, sy) to image-up (0, -1) is atan2(sx, -sy) = atan2(ux, uy).
    # The cv2 matrix at +90 really sends the east screen-vector to
    # screen-up — the round-trip oracle below is the load-bearing check.
    import cv2
    from maps.search.render.look_render import frame_up_angle_deg
    assert abs(frame_up_angle_deg(0.0, 1.0)) < 1e-9
    assert abs(abs(frame_up_angle_deg(0.0, -1.0)) - 180.0) < 1e-9
    assert abs(frame_up_angle_deg(1.0, 0.0) - 90.0) < 1e-9
    M = cv2.getRotationMatrix2D((0.0, 0.0), frame_up_angle_deg(1.0, 0.0), 1.0)
    vx, vy = M[0, 0] * 1.0 + M[0, 1] * 0.0, M[1, 0] * 1.0 + M[1, 1] * 0.0
    assert abs(vx) < 1e-9 and abs(vy + 1.0) < 1e-9, (vx, vy)
