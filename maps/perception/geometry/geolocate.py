"""Pixel <-> world projection.

The bench and the flight observe orthographically: every ``Observation``
carries its view-square ground corners, and a mask pixel's world (x, y) is
an exact affine over them (``project_mask_to_world_ortho``,
``project_mask_to_tile_cells_ortho``, ``world_contour_from_region``). A
mask pixel with no finite, positive depth is dropped, and a detection with
fewer than ``_MIN_VALID_DEPTH_PIXELS`` valid pixels is dropped entirely
rather than placed from a flat-ground fallback.

``project_pixels_to_world`` / ``project_world_to_pixels`` are the
perspective pinhole pair (camera-frame -> body -> world) the flight's
footprint, frame registration and locator use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from maps.perception.geometry.sensor_spec import SensorSpec, body_tilt_matrix, mount_rotation_matrix
from maps.util.mask_region import MaskRegion


@dataclass
class TileProjectedMask:
    """Mask rasterized onto the tile grid, as the occupied cells only.

    ``rows`` / ``cols`` are the deduplicated occupied cells in row-major
    order — the same order (and the same cells) a dense ``(tile_H,
    tile_W)`` bool raster would yield under ``np.where``. Sparse is the
    only representation that scales with the tile: a full-city tile is
    ~10 M cells (10 MB per detection as a dense bool), while a real
    detection footprint is a few hundred cells.
    """

    rows: np.ndarray  # (N,) int32 tile rows of the occupied cells
    cols: np.ndarray  # (N,) int32 tile cols, paired with ``rows``


@dataclass
class ProjectedMask:
    """Per-mask projection result.

    ``world_xy`` is the mean (east, north) of the projected mask pixels.
    ``tile_cells`` is the projected mask rasterized onto the tile grid as
    an ``(N, 2) int32`` array of (row, col) positions; populated when the
    caller passes a ``tile`` (TileFrame) and ``None`` otherwise. The
    downstream mask-IoU re-identification reads ``tile_cells``; AABB
    envelopes are no longer carried because thin diagonal features have
    AABBs that swallow unrelated detections.
    """

    world_xy: tuple[float, float]
    # Tile-grid (row, col) cells the mask rasterizes into, ``(N, 2)
    # int32``. Populated only when the projector received a ``tile``.
    tile_cells: np.ndarray | None = None


@dataclass
class TileFrame:
    """Minimal tile georef needed to rasterize a projected mask."""

    x_min: float
    y_min: float
    cell_size_m: float
    grid_shape: tuple[int, int]  # (H, W)


def _frame_world_xy(
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


def _tile_mask_from_world(
    xs: np.ndarray, ys: np.ndarray, tile: TileFrame
) -> TileProjectedMask | None:
    """Rasterize world points onto ``tile`` as a ``TileProjectedMask``.

    Clip to the tile and collapse the many pixels that land in one cell.
    Dedupe runs on the flat index ``row * W + col``
    through ``np.unique``, which returns ascending order — identical cells
    in identical order to a dense bool raster scanned by ``np.where``,
    without materializing the (H, W) raster.
    """
    col_idx = np.rint((xs - tile.x_min) / tile.cell_size_m - 0.5).astype(np.int64)
    row_idx = np.rint((ys - tile.y_min) / tile.cell_size_m - 0.5).astype(np.int64)
    H, W = tile.grid_shape
    in_bounds = (row_idx >= 0) & (row_idx < H) & (col_idx >= 0) & (col_idx < W)
    if not np.any(in_bounds):
        return None

    flat = np.unique(row_idx[in_bounds] * W + col_idx[in_bounds])
    gr_rows, gr_cols = np.divmod(flat, W)
    return TileProjectedMask(rows=gr_rows.astype(np.int32, copy=False), cols=gr_cols.astype(np.int32, copy=False))


# Below this many valid depth pixels a mask's world position is noise, not a
# measurement, so the detection is dropped rather than placed badly.
_MIN_VALID_DEPTH_PIXELS = 16


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
    """Per-pixel back-projection of a depth buffer to world (east, north, up).

    ``depth`` is meters along the camera +Z axis, matching the pinhole the
    depth source ingests via ``CameraIntrinsics``. The intrinsics carry no
    fisheye coefficients (a plain perspective pinhole), so unprojection is:

        x_cam = (u + 0.5 - cx) / fx * depth
        y_cam = (v + 0.5 - cy) / fy * depth
        z_cam = depth

    Camera frame is +X right, +Y down, +Z forward. The body <- camera
    rotation uses ``sensor.mount_pitch_deg``; the mount is fixed, so the
    optical axis is fixed with it. From body, the yaw convention is
    ``yaw_deg=0`` facing +north, CW-positive, with ``sensor.world_yaw_deg``
    (world-fixed) composing on top. World frame: X=east, Y=north.

    Pixels with non-finite or non-positive depth, or depth beyond
    ``sensor.max_range_m``, are written as NaN. Downstream consumers
    drop NaN cells when computing per-mask aggregates. ``inf`` for
    ``max_range_m`` means range never drops a pixel and only finiteness
    applies.

    ``origin_rc`` is ``depth``'s ``(row, col)`` offset inside the real frame,
    so a CROP unprojects to exactly the world points the full frame would
    have produced there — the pixel grid is the only term that depends on
    absolute position, and it is shifted rather than reconstructed. This is
    what lets a per-detection caller unproject one bounding box instead of
    the whole buffer; the unprojection is linear per pixel, so nothing else
    in the function cares about the crop.
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
    """The exact analytic inverse of :func:`project_pixels_to_world`.

    ``world_xyz`` is ``(..., 3)`` of world (east, north, up) in meters;
    the return is ``(..., 2)`` of continuous ``(u, v)`` pixel coordinates
    in the SAME convention the forward pass consumes -- pixel centers, so
    the center of array cell ``[0, 0]`` is ``(0.5, 0.5)``. Points behind
    the camera, non-finite, or beyond ``sensor.max_range_m`` come back as
    NaN, matching the forward pass's drop rule so a round trip agrees on
    which points exist.

    Every step is the forward's step run backwards, which is what makes
    this exact rather than a fit. The heading rotation inverts to itself:
    the forward applies ``[[sin, cos], [cos, -sin]]``, a symmetric
    involution, so the same matrix carries (east, north) back to
    (forward, right). The body <- camera rotation is orthonormal, so its
    inverse is its transpose. Nothing here iterates.

    Existing depth-buffer callers unproject; this direction is what a
    calibration needs, because it puts the residual in PIXELS. A surveyed
    control point has a known world position and a measured pixel, and
    the measurement noise lives in the image -- so comparing there
    weights every observation correctly, while comparing in world meters
    scales the noise by range and quietly overweights the low passes.

    There is no ``origin_rc``: the return is always in full-frame pixel
    coordinates, and a caller working on a crop subtracts its own
    ``(col, row)`` offset. The forward needs the parameter because it
    builds the pixel grid; this direction is handed the points.
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
) -> TileProjectedMask | None:
    """Rasterize a mask onto ``tile`` through the view-square affine.

    The upstream CityNav (``ortho``) observation is an
    orthographic, yaw-rotated square warp, not a camera frustum, so
    perspective depth unprojection does not apply. Instead each mask pixel
    maps to world (x, y) by the exact view-square affine
    (:func:`_frame_world_xy`). No depth buffer is involved (the ground
    plane is the projection surface).

    Returns ``None`` when the mask has fewer than ``_MIN_VALID_DEPTH_PIXELS``
    set pixels or no projected cell lands inside ``tile``.

    The scan runs over the region's own box, never the frame: the same
    pixels in the same row-major order, for a cost set by the object rather
    than by the sensor's resolution.
    """
    rr, cc = region.rows_cols()
    if rr.size < _MIN_VALID_DEPTH_PIXELS:
        return None
    _u, _v, xs, ys = _frame_world_xy(view_corners_xy, rr, cc, region.frame_shape)
    return _tile_mask_from_world(xs, ys, tile)


# A traced contour is simplified to ~1.5% of its perimeter, then decimated to
# at most this many vertices.
_CONTOUR_EPSILON_FRAC = 0.015
_CONTOUR_MAX_VERTICES = 16


def world_contour_from_region(
    region: MaskRegion,
    view_corners_xy: tuple,
) -> list[list[float]] | None:
    """WORLD_CONTOUR from a native-res crop-pixel mask — ORTHO path only.

    The fine (sub-cell) counterpart of :func:`project_mask_to_tile_cells_ortho`
    for the ``ortho`` backend: instead of rasterizing onto the 2 m
    tile grid (which quantizes a car's orientation to 0/90 garbage), trace the
    mask's largest external contour at NATIVE pixel precision, simplify it
    (``cv2.approxPolyDP``, epsilon ~1.5% of the perimeter), and map crop-px ->
    world meters via the SAME exact view-square affine the ortho projector
    uses (``world = FL + u*(FR-FL) + v*(BL-FL)``, u = col/(w-1),
    v = row/(h-1) — the inverse of upstream ``cropclient.crop_image``'s warp;
    the axis-aligned ``world_window`` alone is NOT sufficient because the
    crop may be yaw-rotated). Decimated to ``_CONTOUR_MAX_VERTICES`` (uniform stride),
    rounded to 2 dp, and returned CLOSED — the chain ends on the vertex it
    started from, which is how a reader tells a ring from a way.

    Returns ``None`` when the mask/corners are degenerate, or when the
    simplified contour has < 3 vertices.
    """
    if not region.crop.any():
        return None

    # Trace the CROP and shift back. cv2's contour is translation-invariant
    # and its border handling matches, verified against a whole-frame trace
    # on blocks, discs, concave shapes and blobs against the frame edge — so
    # this is the same polygon, found without touching the frame.
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
    _u, _v, xs, ys = _frame_world_xy(view_corners_xy, approx[:, 1] + r0, approx[:, 0] + c0, region.frame_shape)
    ring = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs, ys)]
    # A traced mask boundary ENCLOSES, and the chain says so by returning to
    # its first vertex — the one rule that tells a ring from a way
    # (``geom_compute.is_closed_chain``), so nothing downstream has to know
    # this contour came from a pixel mask rather than from a map way.
    return ring + [list(ring[0])]


def project_mask_to_world_ortho(
    region: MaskRegion,
    *,
    depth: np.ndarray | None,
    view_corners_xy: tuple,
    tile: TileFrame | None = None,
) -> ProjectedMask | None:
    """Project a mask to a world centroid through the view-square affine.

    The upstream CityNav (``ortho``) observation is an
    orthographic, yaw-rotated square warp, not a camera frustum, so
    there is no ``SensorSpec`` and the perspective depth unprojection
    does not apply. Each mask pixel's world (x, y) is an exact affine
    over the four view-square ground corners ``[FL, FR, BR, BL]``
    (== image ``[TL, TR, BR, BL]``) — the inverse of
    ``observation.crop_ortho_observation``'s warp. The warp is a purely planar
    2-D affine (the square is a parallelogram, no z-perspective skew), so
    the bilinear map below is exact, not an approximation. See
    ``maps/perception/tests/test_sensor_frame.py`` for
    the forward geometry gate (corner/yaw invariants).

    Depth gates validity (a mask pixel with no finite, positive depth is
    dropped). Upstream
    ships depth at a different resolution than the RGB the mask lives in
    (224×224 RGB, 256×256 depth); both warp the same view square, so each
    mask pixel's normalized ``(u, v)`` is sampled in the depth grid by
    nearest-neighbor rather than requiring equal shapes.

    Returns a ``ProjectedMask`` (``world_xy``, optional ``tile_cells``), or ``None`` when ``depth`` is
    absent or fewer than ``_MIN_VALID_DEPTH_PIXELS`` mask pixels have
    finite, positive depth. There is no ``max_range_m`` gate (no sensor in
    the orthographic path).

    The set-pixel scan runs over the region's box, never the frame. The
    normalised ``(u, v)`` still divide by the FRAME extent, which is why a
    region carries its frame shape: the box narrows the scan, never the
    geometry.
    """
    if depth is None:
        return None
    depth_arr = np.asarray(depth, dtype=np.float32)
    depth2d = depth_arr if depth_arr.ndim == 2 else depth_arr[:, :, 0]
    rr, cc = region.rows_cols()
    if rr.size == 0:
        return None
    u, v, xs, ys = _frame_world_xy(view_corners_xy, rr, cc, region.frame_shape)

    # Upstream CityNav ships the depth raster at a different resolution
    # than the RGB the mask lives in (224x224 RGB vs 256x256 depth). Both
    # warp the *same* view square, so a mask pixel's normalized (u, v)
    # addresses the same world point in either grid. Sample depth at that
    # normalized location by nearest-neighbor (depth must not blend
    # across building/ground discontinuities).
    hd, wd = depth2d.shape
    dr = np.clip(np.rint(v * (hd - 1)).astype(np.int64), 0, hd - 1)
    dc = np.clip(np.rint(u * (wd - 1)).astype(np.int64), 0, wd - 1)
    d = depth2d[dr, dc]
    finite = np.isfinite(d) & (d > 0.0)
    if int(finite.sum()) < _MIN_VALID_DEPTH_PIXELS:
        return None
    xs = xs[finite]
    ys = ys[finite]
    world_xy = (float(xs.mean()), float(ys.mean()))
    tile_cells: np.ndarray | None = None
    if tile is not None:
        tm = _tile_mask_from_world(xs, ys, tile)
        if tm is not None:
            tile_cells = np.stack([tm.rows, tm.cols], axis=1)
    return ProjectedMask(world_xy=world_xy, tile_cells=tile_cells)
