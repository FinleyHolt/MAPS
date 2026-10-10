"""The perspective path reads a detection's region, never the frame.

``project_mask_to_world`` and ``project_mask_to_tile_cells`` used to unproject
the WHOLE depth buffer and then index it with a whole-frame boolean mask —
once per detection. On a real camera frame that is several full-resolution
float64 temporaries per object, computing the same thing for every object in
the frame. This is the field path: postflight builds an
Observation with a real ``sensor`` and ``depth``, so it runs here and not
through the ortho projectors.

Narrowing both passes to the mask's own box is only legitimate if the result
is bit-identical, which is what this pins. The subtle part is that the pixel
grid is the one term that depends on absolute position, so a crop must carry
its ``origin_rc`` or every world point silently shifts — the first test is
there to fail loudly if that offset is ever dropped.
"""

from __future__ import annotations

import numpy as np

from maps.perception.ortho.camera_model import CameraIntrinsics
from maps.perception.geometry.geolocate import (
    TileFrame,
    project_mask_to_tile_cells,
    project_mask_to_world,
    project_pixels_to_world,
)
from maps.util.mask_region import MaskRegion
from maps.perception.geometry.sensor_spec import SensorSpec

_H, _W = 96, 128
_TILE = TileFrame(x_min=-200.0, y_min=-200.0, cell_size_m=2.0, grid_shape=(200, 200))


def _sensor(max_range_m: float = 200.0) -> SensorSpec:
    return SensorSpec(
        name="field_depth",
        intrinsics=CameraIntrinsics(
            hfov_deg=90.0, vfov_deg=60.0, image_width=_W, image_height=_H
        ),
        mount_pitch_deg=90.0,
        max_range_m=max_range_m,
    )


def _pose(**over):
    kw = dict(
        pose_xy=(12.0, -7.0),
        altitude_m=60.0,
        yaw_deg=31.0,
        sensor=_sensor(),
        airframe_roll_deg=2.5,
        airframe_pitch_deg=-1.5,
    )
    kw.update(over)
    return kw


def _depth(rng):
    return (55.0 + rng.random((_H, _W)) * 20.0).astype(np.float32)


def _mask_and_box(r0, c0, h, w, rng, fill=0.75):
    m = np.zeros((_H, _W), dtype=bool)
    m[r0 : r0 + h, c0 : c0 + w] = rng.random((h, w)) < fill
    m[r0, c0] = m[r0 + h - 1, c0 + w - 1] = True  # pin the box corners
    rr, cc = np.where(m)
    return m, (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max()))


def test_cropped_unprojection_equals_the_frame_slice():
    """A crop with its origin must reproduce the frame's values exactly.

    This is the load-bearing invariant: drop ``origin_rc`` and every point
    still looks plausible while sitting metres from where it belongs.
    """
    rng = np.random.default_rng(19)
    depth = _depth(rng)
    full = project_pixels_to_world(depth, **_pose())

    for r0, c0, r1, c1 in [(0, 0, 20, 30), (33, 41, 70, 99), (60, 90, 95, 127)]:
        sub = project_pixels_to_world(
            depth[r0 : r1 + 1, c0 : c1 + 1], origin_rc=(r0, c0), **_pose()
        )
        assert np.array_equal(
            sub, full[r0 : r1 + 1, c0 : c1 + 1], equal_nan=True
        ), (r0, c0)

    # Vacuity floor: without the offset the crop must DISAGREE, or this test
    # would pass against a projector that ignores absolute pixel position.
    naive = project_pixels_to_world(depth[33:71, 41:100], **_pose())
    assert not np.allclose(naive, full[33:71, 41:100], equal_nan=True)


def test_projection_reads_only_the_region():
    """A region projects to the same world points a whole-frame scan of the
    same mask would, and the box it carries is a real narrowing.

    The old whole-frame path is gone, so the reference here is the geometry
    itself: unproject the frame, take the mask's points, compare.
    """
    rng = np.random.default_rng(23)
    depth = _depth(rng)
    for placement in [(0, 0, 18, 22), (40, 55, 25, 30), (70, 100, 26, 28)]:
        mask, box = _mask_and_box(*placement, rng)
        region = MaskRegion.from_full(mask)
        assert region.bbox == box

        got = project_mask_to_world(
            region, depth=depth, return_xy_cloud=True, tile=_TILE, **_pose()
        )
        assert got is not None, placement

        # Independent reference: unproject the whole frame, mask it.
        world = project_pixels_to_world(depth, **_pose())
        pts = world[mask]
        finite = np.isfinite(pts).all(axis=-1)
        pts = pts[finite]
        assert got.n_valid_pixels == int(finite.sum())
        assert got.world_xy == (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        assert np.array_equal(got.world_xy_cloud, pts[:, :2].astype(np.float32))

        box_px = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        assert box_px < mask.size / 3  # the region is a real narrowing


def test_tile_cells_match_the_world_points_they_come_from():
    rng = np.random.default_rng(29)
    depth = _depth(rng)
    mask, _box = _mask_and_box(30, 44, 24, 26, rng)
    region = MaskRegion.from_full(mask)

    tp = project_mask_to_tile_cells(region, depth=depth, tile=_TILE, **_pose())
    pm = project_mask_to_world(
        region, depth=depth, return_xy_cloud=True, tile=_TILE, **_pose()
    )
    assert tp is not None and pm is not None
    # The tile-cell projector and the world projector rasterise the same cloud.
    assert np.array_equal(tp.rows, pm.tile_cells[:, 0])
    assert np.array_equal(tp.cols, pm.tile_cells[:, 1])
    assert 0 < tp.cell_count < int(mask.sum())  # cells really collapse pixels


def test_invalid_depth_drops_the_same_pixels_the_frame_would():
    """NaN / non-positive / out-of-range depth gate validity, and the
    under-threshold drop counts survivors. A region that examined the wrong
    pixels would keep or drop detections the geometry does not."""
    rng = np.random.default_rng(31)
    depth = _depth(rng)
    mask, _box = _mask_and_box(20, 30, 20, 24, rng)
    region = MaskRegion.from_full(mask)

    depth[22:34, 32:50] = np.nan
    depth[34:36, 32:50] = -1.0
    depth[36:38, 32:50] = 1e6  # beyond max_range_m

    world = project_pixels_to_world(depth, **_pose())
    expected_valid = int(np.isfinite(world[mask]).all(axis=-1).sum())

    for min_px in (1, 8, 16, 64, 4096):
        got = project_mask_to_world(
            region, depth=depth, min_valid_depth_pixels=min_px, **_pose()
        )
        if min_px > expected_valid:
            assert got is None, min_px
        else:
            assert got is not None and got.n_valid_pixels == expected_valid, min_px

    # Vacuity floor: the sweep has to cross the threshold, and the invalid
    # depth has to have actually removed pixels.
    assert 0 < expected_valid < int(mask.sum())
    assert 1 <= expected_valid < 4096
