"""Pixel <-> world projection.

Every ``Observation`` is orthographic and carries its view-square ground
corners; a mask pixel's world (x, y) is an exact affine over them (the
``*_ortho`` projectors and ``world_contour_from_region``).
``project_pixels_to_world`` / ``project_world_to_pixels`` are the perspective
pinhole pair the flight's footprint, frame registration and locator use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from maps.perception.geometry.sensor_spec import SensorSpec, body_tilt_matrix, mount_rotation_matrix
from maps.perception.mask_region import MaskRegion


@dataclass
class TileFrame:
    """Minimal tile georef needed to rasterize a projected mask."""

    x_min: float
    y_min: float
    cell_size_m: float
    grid_shape: tuple[int, int]  # (H, W)


def frame_world_xy(
    view_corners_xy: tuple, rows: np.ndarray, cols: np.ndarray, frame_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, x, y)`` of frame pixels ``(rows, cols)``: normalised frame
    coordinates ``u = col / (w - 1)``, ``v = row / (h - 1)`` and the world
    (x, y) there through the view-square corners ``[FL, FR, BR, BL]`` (==
    image ``[TL, TR, BR, BL]``). ``world = FL + u*(FR-FL) + v*(BL-FL)`` is
    exact: the square is a parallelogram, so the bilinear map is affine.
    """
    h, w = frame_shape
    u = cols.astype(np.float64) / max(w - 1, 1)
    v = rows.astype(np.float64) / max(h - 1, 1)
    flx, fly = float(view_corners_xy[0][0]), float(view_corners_xy[0][1])
    frx, fry = float(view_corners_xy[1][0]), float(view_corners_xy[1][1])
    blx, bly = float(view_corners_xy[3][0]), float(view_corners_xy[3][1])
    xs = flx + u * (frx - flx) + v * (blx - flx)
    ys = fly + u * (fry - fly) + v * (bly - fly)
    return u, v, xs, ys


# Below this many pixels (set, or with valid depth) a mask's world position
# is noise, not a measurement, so the detection is dropped rather than placed
# badly.
_MIN_MASK_PIXELS = 16


def _r_bc(sensor: SensorSpec, airframe_roll_deg: float, airframe_pitch_deg: float) -> np.ndarray:
    """Body <- camera rotation: the static mount, with the airframe's tilt
    (roll/pitch deviation from level) composed on top, ``R_eff = (level-body
    <- body) @ (body <- camera)``. The heading rotation assumes a level body,
    so the tilt carries the ray into the level frame first; the mount alone
    when both tilt angles are 0."""
    r = mount_rotation_matrix(float(sensor.mount_pitch_deg), sensor.mount_roll_deg, float(sensor.mount_yaw_deg))
    if airframe_roll_deg or airframe_pitch_deg:
        r = body_tilt_matrix(airframe_roll_deg, airframe_pitch_deg) @ r
    return r


def _invalid(z_cam: np.ndarray, sensor: SensorSpec) -> np.ndarray:
    """The points neither direction keeps: non-finite or non-positive camera
    depth, or depth beyond a finite ``sensor.max_range_m``."""
    invalid = ~np.isfinite(z_cam) | (z_cam <= 0.0)
    max_range = float(sensor.max_range_m)
    if math.isfinite(max_range):
        invalid = invalid | (z_cam > max_range)
    return invalid


def project_pixels_to_world(
    depth: np.ndarray,
    *,
    pose_xy: tuple[float, float],
    altitude_m: float,
    yaw_deg: float,
    sensor: SensorSpec,
    airframe_roll_deg: float = 0.0,
    airframe_pitch_deg: float = 0.0,
    origin_rc: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Per-pixel back-projection of a depth buffer (meters along camera +Z)
    to world (east, north, up), through a plain pinhole:

        x_cam = (u + 0.5 - cx) / fx * depth
        y_cam = (v + 0.5 - cy) / fy * depth
        z_cam = depth

    Camera frame +X right, +Y down, +Z forward; body <- camera is the mount
    (:func:`_r_bc`); heading ``yaw_deg`` (0 = north, CW-positive) plus
    ``sensor.world_yaw_deg``. A pixel with non-finite or non-positive depth,
    or depth beyond ``sensor.max_range_m``, is NaN. ``origin_rc`` is
    ``depth``'s ``(row, col)`` offset in the frame, so a crop unprojects to
    the world points the full frame would give there.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W); got shape {depth.shape}")
    H, W = depth.shape
    intr = sensor.intrinsics
    r_off, c_off = (int(origin_rc[0]), int(origin_rc[1]))
    us = np.arange(W, dtype=np.float64) + 0.5 + c_off
    vs = np.arange(H, dtype=np.float64) + 0.5 + r_off
    u_grid, v_grid = np.meshgrid(us, vs)
    z_cam = np.asarray(depth, dtype=np.float64)
    x_cam = (u_grid - intr.cx) / intr.fx * z_cam
    y_cam = (v_grid - intr.cy) / intr.fy * z_cam

    R_bc = _r_bc(sensor, airframe_roll_deg, airframe_pitch_deg)
    body_x = R_bc[0, 0] * x_cam + R_bc[0, 1] * y_cam + R_bc[0, 2] * z_cam  # forward
    body_y = R_bc[1, 0] * x_cam + R_bc[1, 1] * y_cam + R_bc[1, 2] * z_cam  # left
    body_z = R_bc[2, 0] * x_cam + R_bc[2, 1] * y_cam + R_bc[2, 2] * z_cam  # up

    right_m = -body_y
    forward_m = body_x
    heading_rad = math.radians(float(yaw_deg) + float(sensor.world_yaw_deg))
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    east_m = s * forward_m + c * right_m
    north_m = c * forward_m - s * right_m

    world_x = pose_xy[0] + east_m
    world_y = pose_xy[1] + north_m
    world_z = altitude_m + body_z

    out = np.stack([world_x, world_y, world_z], axis=-1).astype(np.float32)
    out[_invalid(z_cam, sensor)] = np.nan
    return out


def project_world_to_pixels(
    world_xyz: np.ndarray,
    *,
    pose_xy: tuple[float, float],
    altitude_m: float,
    yaw_deg: float,
    sensor: SensorSpec,
    airframe_roll_deg: float = 0.0,
    airframe_pitch_deg: float = 0.0,
) -> np.ndarray:
    """The exact inverse of :func:`project_pixels_to_world`: ``(..., 3)``
    world (east, north, up) to ``(..., 2)`` full-frame ``(u, v)`` pixel
    coordinates (cell ``[0, 0]``'s center is ``(0.5, 0.5)``). A point behind
    the camera, non-finite, or beyond ``sensor.max_range_m`` is NaN, the
    forward's drop rule. Each step is the forward's run backwards: the
    heading rotation is a symmetric involution and the mount rotation is
    orthonormal, so its inverse is its transpose.
    """
    pts = np.asarray(world_xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"world_xyz must be (..., 3); got shape {pts.shape}")

    east_m = pts[..., 0] - float(pose_xy[0])
    north_m = pts[..., 1] - float(pose_xy[1])
    body_z = pts[..., 2] - float(altitude_m)

    heading_rad = math.radians(float(yaw_deg) + float(sensor.world_yaw_deg))
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    forward_m = s * east_m + c * north_m
    right_m = c * east_m - s * north_m

    body_x = forward_m
    body_y = -right_m

    R_bc = _r_bc(sensor, airframe_roll_deg, airframe_pitch_deg)
    # Orthonormal, so camera <- body is the transpose.
    x_cam = R_bc[0, 0] * body_x + R_bc[1, 0] * body_y + R_bc[2, 0] * body_z
    y_cam = R_bc[0, 1] * body_x + R_bc[1, 1] * body_y + R_bc[2, 1] * body_z
    z_cam = R_bc[0, 2] * body_x + R_bc[1, 2] * body_y + R_bc[2, 2] * body_z

    intr = sensor.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.cx + intr.fx * (x_cam / z_cam)
        v = intr.cy + intr.fy * (y_cam / z_cam)

    out = np.stack([u, v], axis=-1)
    out[_invalid(z_cam, sensor)] = np.nan
    return out


def project_mask_to_tile_cells_ortho(
    region: MaskRegion,
    *,
    view_corners_xy: tuple,
    tile: TileFrame,
) -> np.ndarray | None:
    """A mask's tile cells through the view-square affine (no depth): the
    ``(N, 2) int32`` (row, col) cells its pixels land in, clipped to ``tile``
    and deduplicated (``np.unique`` on ``row * W + col`` gives row-major
    order); ``None`` with fewer than ``_MIN_MASK_PIXELS`` set pixels or no
    cell inside ``tile``."""
    rr, cc = region.rows_cols()
    if rr.size < _MIN_MASK_PIXELS:
        return None
    _u, _v, xs, ys = frame_world_xy(view_corners_xy, rr, cc, region.frame_shape)
    col_idx = np.rint((xs - tile.x_min) / tile.cell_size_m - 0.5).astype(np.int64)
    row_idx = np.rint((ys - tile.y_min) / tile.cell_size_m - 0.5).astype(np.int64)
    H, W = tile.grid_shape
    in_bounds = (row_idx >= 0) & (row_idx < H) & (col_idx >= 0) & (col_idx < W)
    if not np.any(in_bounds):
        return None
    flat = np.unique(row_idx[in_bounds] * W + col_idx[in_bounds])
    return np.stack(np.divmod(flat, W), axis=1).astype(np.int32)


# A traced contour is simplified to ~1.5% of its perimeter, then decimated to
# at most this many vertices.
_CONTOUR_EPSILON_FRAC = 0.015
_CONTOUR_MAX_VERTICES = 16


def world_contour_from_region(
    region: MaskRegion,
    view_corners_xy: tuple,
) -> list[list[float]] | None:
    """The mask's world footprint: its largest external contour traced at
    pixel precision (finer than the tile grid, so a car keeps its
    orientation), simplified, decimated to ``_CONTOUR_MAX_VERTICES``, mapped
    through the view-square affine, rounded to 2 dp and returned closed (the
    chain ends on its first vertex, which is how a reader tells a ring from a
    way). ``None`` for an empty mask or fewer than 3 vertices.
    """
    if not region.crop.any():
        return None

    # Trace the crop and shift back: cv2's contour is translation-invariant.
    contours, _hier = cv2.findContours(
        region.crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    eps = _CONTOUR_EPSILON_FRAC * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
    if approx.shape[0] < 3:
        return None
    if approx.shape[0] > _CONTOUR_MAX_VERTICES:
        stride = int(np.ceil(approx.shape[0] / _CONTOUR_MAX_VERTICES))
        approx = approx[::stride]
        if approx.shape[0] < 3:
            return None

    # cv2 contour points are (x=col, y=row), crop-relative until shifted.
    r0, c0 = region.bbox[0], region.bbox[1]
    _u, _v, xs, ys = frame_world_xy(view_corners_xy, approx[:, 1] + r0, approx[:, 0] + c0, region.frame_shape)
    ring = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs, ys)]
    return ring + [list(ring[0])]


def project_mask_to_world_ortho(
    region: MaskRegion,
    *,
    depth: np.ndarray | None,
    view_corners_xy: tuple,
) -> tuple[float, float] | None:
    """A mask's world centroid, the mean (east, north) through the
    view-square affine over the pixels with finite, positive depth (depth
    sampled nearest-neighbour at each pixel's normalised ``(u, v)``, so its
    grid may differ from the mask's). ``None`` with no depth or fewer than ``_MIN_MASK_PIXELS``
    valid pixels.
    """
    if depth is None:
        return None
    depth_arr = np.asarray(depth, dtype=np.float32)
    depth2d = depth_arr if depth_arr.ndim == 2 else depth_arr[:, :, 0]
    rr, cc = region.rows_cols()
    if rr.size == 0:
        return None
    u, v, xs, ys = frame_world_xy(view_corners_xy, rr, cc, region.frame_shape)

    # Nearest-neighbour: depth must not blend across building/ground edges.
    hd, wd = depth2d.shape
    dr = np.clip(np.rint(v * (hd - 1)).astype(np.int64), 0, hd - 1)
    dc = np.clip(np.rint(u * (wd - 1)).astype(np.int64), 0, wd - 1)
    d = depth2d[dr, dc]
    finite = np.isfinite(d) & (d > 0.0)
    if int(finite.sum()) < _MIN_MASK_PIXELS:
        return None
    return (float(xs[finite].mean()), float(ys[finite].mean()))
