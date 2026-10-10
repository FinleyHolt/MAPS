"""Depth-aware pixel -> world (x, y) projection for the citynav runtime.

Every observation carries a per-pixel depth buffer (the ortho crop's
``altitude − height``, or the field DSM raycast), so the canonical projection is
per-pixel back-projection through that depth.
Any pixel whose depth is non-finite, non-positive, or beyond
``sensor.max_range_m`` is dropped. A detection whose mask has fewer
than ``min_valid_depth_pixels`` valid samples is dropped entirely
rather than have its world coordinates synthesised from a flat-ground
fallback.

Two call surfaces:

- ``project_mask_to_world`` returns ``(world_xy, world_bbox)`` for a
  detection. Used by the SAM 3.1 video adapter and the SegFormer
  adapter to stamp world geometry onto a ``DetectedObject``.
- ``project_mask_to_tile_cells`` rasterizes the same projected pixel
  cloud onto a tile grid, returning a ``TileProjectedMask`` for the
  CityNav per-step belief upsert.

Both wrap ``project_pixels_to_world`` (perspective unprojection with
camera-frame -> body -> world rotation) and apply the same drop rule.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

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
    detection footprint is a few hundred cells, and this object crosses
    a pickled HTTP boundary into the world-state ingest server.
    """

    rows: np.ndarray  # (N,) int32 tile rows of the occupied cells
    cols: np.ndarray  # (N,) int32 tile cols, paired with ``rows``
    cells: list[list[int]]  # list of [row, col] for occupied cells
    centroid_rc: tuple[int, int]
    cell_count: int
    centroid_xy: tuple[float, float]


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
    n_valid_pixels: int
    # Per-valid-mask-pixel world (x, y) — shape ``(n_valid_pixels, 2)``,
    # float32. Used by callers that need to re-project the mask through
    # a different 2D frame (e.g. the satellite ortho) for visualization
    # at pixel fidelity. None means the caller asked the adapter to
    # skip computing it (default in the runtime hot path, where the
    # centroid + tile_cells are all the re-id matcher needs).
    world_xy_cloud: np.ndarray | None = None
    # Tile-grid (row, col) cells the mask rasterizes into, ``(N, 2)
    # int32``. Populated only when ``project_mask_to_world`` received a
    # ``tile``; ``None`` otherwise (no-tile harnesses don't need it).
    tile_cells: np.ndarray | None = None


@dataclass
class TileFrame:
    """Minimal tile georef needed to rasterize a projected mask."""

    x_min: float
    y_min: float
    cell_size_m: float
    grid_shape: tuple[int, int]  # (H, W)


def _tile_mask_from_cell_indices(
    row_idx: np.ndarray, col_idx: np.ndarray, tile: TileFrame
) -> TileProjectedMask | None:
    """Dedupe per-pixel cell indices into a ``TileProjectedMask``.

    The shared tail of both projectors: clip to the tile, collapse the
    many pixels that land in one cell, and derive the centroid. Dedupe
    runs on the flat index ``row * W + col`` through ``np.unique``, which
    returns ascending order — identical cells in identical order to a
    dense bool raster scanned by ``np.where``, without materializing the
    (H, W) raster.
    """
    H, W = tile.grid_shape
    in_bounds = (row_idx >= 0) & (row_idx < H) & (col_idx >= 0) & (col_idx < W)
    if not np.any(in_bounds):
        return None

    flat = np.unique(row_idx[in_bounds] * W + col_idx[in_bounds])
    gr_rows, gr_cols = np.divmod(flat, W)
    gr_rows = gr_rows.astype(np.int32, copy=False)
    gr_cols = gr_cols.astype(np.int32, copy=False)

    centroid_row = int(round(float(gr_rows.mean())))
    centroid_col = int(round(float(gr_cols.mean())))
    world_cx = tile.x_min + (centroid_col + 0.5) * tile.cell_size_m
    world_cy = tile.y_min + (centroid_row + 0.5) * tile.cell_size_m
    cells = [[int(r), int(c)] for r, c in zip(gr_rows.tolist(), gr_cols.tolist())]
    return TileProjectedMask(
        rows=gr_rows,
        cols=gr_cols,
        cells=cells,
        centroid_rc=(centroid_row, centroid_col),
        cell_count=len(cells),
        centroid_xy=(world_cx, world_cy),
    )


# ---------------------------------------------------------------------------
# Depth-aware projection (the runtime ingest path)
# ---------------------------------------------------------------------------


# Below this many valid depth pixels a mask's world position is noise, not a
# measurement, so the detection is dropped rather than placed badly.
_MIN_VALID_DEPTH_PIXELS = 16


def _pitch_rotation_matrix(pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Body <- camera rotation for the given pitch + roll (positive-down pitch,
    body-X roll).

    Delegates to ``sensor_spec.mount_rotation_matrix`` (the single source of
    truth). Camera +Z = optical axis, +X = right, +Y = down; body +X = forward,
    +Y = left, +Z = up.
    """
    return mount_rotation_matrix(pitch_deg, roll_deg, yaw_deg)


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

    R_bc = _pitch_rotation_matrix(
        float(sensor.mount_pitch_deg), sensor.mount_roll_deg, float(sensor.mount_yaw_deg)
    )  # body <- camera
    # Per-frame airframe tilt (roll/pitch deviation from level) composes on top
    # of the static mount: R_eff = (level-body <- body) @ (body <- camera). The
    # heading rotation below assumes a level body, so the tilt must carry the
    # ray into the level frame first. Identity when both are 0 (byte-identical).
    if airframe_roll_deg or airframe_pitch_deg:
        R_bc = body_tilt_matrix(airframe_roll_deg, airframe_pitch_deg) @ R_bc
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
    max_range = float(sensor.max_range_m)
    invalid = ~np.isfinite(z_cam) | (z_cam <= 0.0)
    if math.isfinite(max_range):
        invalid = invalid | (z_cam > max_range)
    out[invalid] = np.nan
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

    R_bc = _pitch_rotation_matrix(float(sensor.mount_pitch_deg), sensor.mount_roll_deg,
                                  float(sensor.mount_yaw_deg))
    if airframe_roll_deg or airframe_pitch_deg:
        R_bc = body_tilt_matrix(airframe_roll_deg, airframe_pitch_deg) @ R_bc
    # Orthonormal, so camera <- body is the transpose.
    x_cam = R_bc[0, 0] * body_x + R_bc[1, 0] * body_y + R_bc[2, 0] * body_z
    y_cam = R_bc[0, 1] * body_x + R_bc[1, 1] * body_y + R_bc[2, 1] * body_z
    z_cam = R_bc[0, 2] * body_x + R_bc[1, 2] * body_y + R_bc[2, 2] * body_z

    intr = sensor.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.cx + intr.fx * (x_cam / z_cam)
        v = intr.cy + intr.fy * (y_cam / z_cam)

    out = np.stack([u, v], axis=-1)
    max_range = float(sensor.max_range_m)
    invalid = ~np.isfinite(z_cam) | (z_cam <= 0.0)
    if math.isfinite(max_range):
        invalid = invalid | (z_cam > max_range)
    out[invalid] = np.nan
    return out


def _world_points_under_mask(
    region: MaskRegion,
    depth: np.ndarray,
    **proj_kw,
) -> np.ndarray:
    """The world points under ``mask``, unprojecting only what it covers.

    Both perspective projectors used to unproject the WHOLE depth buffer and
    then index it with a whole-frame boolean mask — once per detection, for a
    detection covering a few hundred pixels. On a real camera frame that is
    several full-resolution float64 temporaries per object, and the work is
    identical for every object in the frame.

    Given the mask's own inclusive box, both passes shrink to it: the crop
    unprojects to the same absolute world points (``origin_rc`` carries the
    pixel grid), and boolean indexing walks row-major in the box exactly as it
    did in the frame, so the returned point SEQUENCE is unchanged — which is
    what matters, since callers take means over it.

    There is no whole-frame branch: a region always knows its box, which is
    the point of the type.
    """
    r0, c0 = region.bbox[0], region.bbox[1]
    sub_depth = np.asarray(region.window(depth), dtype=np.float32)
    world_xyz = project_pixels_to_world(sub_depth, origin_rc=(r0, c0), **proj_kw)
    return world_xyz[region.crop]


def project_mask_to_world(
    region: MaskRegion,
    *,
    depth: np.ndarray | None,
    pose_xy: tuple[float, float],
    altitude_m: float,
    yaw_deg: float,
    sensor: SensorSpec,
    airframe_roll_deg: float = 0.0,
    airframe_pitch_deg: float = 0.0,
    min_valid_depth_pixels: int | None = None,
    return_xy_cloud: bool = False,
    tile: TileFrame | None = None,
) -> ProjectedMask | None:
    """Project a boolean pixel mask through ``depth`` to a world centroid.

    Returns a ``ProjectedMask`` carrying:
      - ``world_xy``: mean (east, north) of all valid projected pixels
      - ``n_valid_pixels``: count of mask pixels that survived the
        depth filter — load-bearing signal for the engine's
        ``area-ratio > 2x → centroid-distance fallback`` re-id path.
      - ``tile_cells``: rasterized grid (row, col) positions when a
        ``tile`` was provided. ``None`` otherwise. Downstream mask-IoU
        re-identification reads this directly; AABB envelopes are gone.

    Returns ``None`` when ``depth`` is absent, shape-mismatched against
    the region, or has fewer than ``min_valid_depth_pixels`` valid
    samples under the mask. No flat-ground fallback.
    """
    if depth is None:
        return None
    if tuple(region.frame_shape) != tuple(depth.shape[:2]):
        return None
    threshold = (
        int(min_valid_depth_pixels)
        if min_valid_depth_pixels is not None
        else _MIN_VALID_DEPTH_PIXELS
    )
    pts = _world_points_under_mask(
        region,
        depth,
        pose_xy=pose_xy,
        altitude_m=altitude_m,
        yaw_deg=yaw_deg,
        sensor=sensor,
        airframe_roll_deg=airframe_roll_deg,
        airframe_pitch_deg=airframe_pitch_deg,
    )
    finite = np.isfinite(pts).all(axis=-1)
    n_valid = int(finite.sum())
    if n_valid < threshold:
        return None
    pts = pts[finite]
    world_xy = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    cloud: np.ndarray | None = None
    if return_xy_cloud:
        cloud = pts[:, :2].astype(np.float32, copy=False)
    tile_cells: np.ndarray | None = None
    if tile is not None:
        H, W = tile.grid_shape
        col_idx = np.rint((pts[:, 0] - tile.x_min) / tile.cell_size_m - 0.5).astype(
            np.int64
        )
        row_idx = np.rint((pts[:, 1] - tile.y_min) / tile.cell_size_m - 0.5).astype(
            np.int64
        )
        in_bounds = (row_idx >= 0) & (row_idx < H) & (col_idx >= 0) & (col_idx < W)
        if np.any(in_bounds):
            grid = np.zeros((H, W), dtype=bool)
            grid[row_idx[in_bounds], col_idx[in_bounds]] = True
            rr, cc = np.where(grid)
            tile_cells = np.stack(
                [rr.astype(np.int32, copy=False), cc.astype(np.int32, copy=False)],
                axis=1,
            )
    return ProjectedMask(
        world_xy=world_xy,
        n_valid_pixels=n_valid,
        world_xy_cloud=cloud,
        tile_cells=tile_cells,
    )


def project_mask_to_tile_cells(
    region: MaskRegion,
    *,
    depth: np.ndarray | None,
    pose_xy: tuple[float, float],
    altitude_m: float,
    yaw_deg: float,
    sensor: SensorSpec,
    tile: TileFrame,
    airframe_roll_deg: float = 0.0,
    airframe_pitch_deg: float = 0.0,
    min_valid_depth_pixels: int | None = None,
) -> TileProjectedMask | None:
    """Rasterize a boolean pixel mask onto a ``TileFrame`` via per-pixel
    depth back-projection.

    Same drop rule as ``project_mask_to_world``: requires ``depth`` to
    be present and shape-matched, and at least
    ``min_valid_depth_pixels`` valid samples under the mask. Returns
    ``None`` if no projected cell lands inside the tile.

    Row 0 = south (matches the CityNav prebake convention in
    ``benchmarks/citynav/adapter._rasterize_object`` and
    ``maps.sws.graph.builder``).
    """
    if depth is None:
        return None
    if tuple(region.frame_shape) != tuple(depth.shape[:2]):
        return None
    threshold = (
        int(min_valid_depth_pixels)
        if min_valid_depth_pixels is not None
        else _MIN_VALID_DEPTH_PIXELS
    )
    pts = _world_points_under_mask(
        region,
        depth,
        pose_xy=pose_xy,
        altitude_m=altitude_m,
        yaw_deg=yaw_deg,
        sensor=sensor,
        airframe_roll_deg=airframe_roll_deg,
        airframe_pitch_deg=airframe_pitch_deg,
    )
    finite = np.isfinite(pts).all(axis=-1)
    if int(finite.sum()) < threshold:
        return None
    pts = pts[finite]
    xs = pts[:, 0]
    ys = pts[:, 1]

    col_idx = np.rint((xs - tile.x_min) / tile.cell_size_m - 0.5).astype(np.int64)
    row_idx = np.rint((ys - tile.y_min) / tile.cell_size_m - 0.5).astype(np.int64)
    return _tile_mask_from_cell_indices(row_idx, col_idx, tile)


def project_mask_to_tile_cells_ortho(
    region: MaskRegion,
    *,
    view_corners_xy: tuple,
    tile: TileFrame,
    min_pixels: int | None = None,
) -> TileProjectedMask | None:
    """Orthographic counterpart of :func:`project_mask_to_tile_cells`.

    The upstream CityNav (``ortho``) observation is an
    orthographic, yaw-rotated square warp, not a camera frustum, so
    perspective depth unprojection does not apply. Instead each mask pixel
    maps to world (x, y) by an exact affine over the four view-square
    ground corners ``[FL, FR, BR, BL]`` (== image ``[TL, TR, BR, BL]``,
    the inverse of upstream ``cropclient.crop_image``'s warp). No depth
    buffer is involved (the ground plane is the projection surface).

    Returns ``None`` when the mask has fewer than ``min_pixels`` set
    pixels or no projected cell lands inside ``tile`` — same drop
    semantics as the depth path.

    The scan runs over the region's own box, never the frame: the same
    pixels in the same row-major order, for a cost set by the object rather
    than by the sensor's resolution.
    """
    threshold = (
        int(min_pixels) if min_pixels is not None else _MIN_VALID_DEPTH_PIXELS
    )
    rr, cc = region.rows_cols()
    if rr.size < threshold:
        return None

    (flx, fly), (frx, fry), _br, (blx, bly) = (
        (float(view_corners_xy[0][0]), float(view_corners_xy[0][1])),
        (float(view_corners_xy[1][0]), float(view_corners_xy[1][1])),
        view_corners_xy[2],
        (float(view_corners_xy[3][0]), float(view_corners_xy[3][1])),
    )
    h, w = region.frame_shape
    u = cc.astype(np.float64) / max(w - 1, 1)
    v = rr.astype(np.float64) / max(h - 1, 1)
    # world = FL + u*(FR-FL) + v*(BL-FL); exact (the square is a
    # parallelogram, so the bilinear map is affine).
    xs = flx + u * (frx - flx) + v * (blx - flx)
    ys = fly + u * (fry - fly) + v * (bly - fly)

    col_idx = np.rint((xs - tile.x_min) / tile.cell_size_m - 0.5).astype(np.int64)
    row_idx = np.rint((ys - tile.y_min) / tile.cell_size_m - 0.5).astype(np.int64)
    return _tile_mask_from_cell_indices(row_idx, col_idx, tile)


def world_contour_from_region(
    region: MaskRegion,
    view_corners_xy: tuple,
    *,
    max_vertices: int = 16,
    epsilon_frac: float = 0.015,
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
    crop may be yaw-rotated). Decimated to ``max_vertices`` (uniform stride),
    rounded to 2 dp, and returned CLOSED — the chain ends on the vertex it
    started from, which is how a reader tells a ring from a way.

    Returns ``None`` when cv2 is unavailable (graceful degrade: ``cells``
    still flow, the node just gets no fine contour), when the mask/corners
    are degenerate, or when the simplified contour has < 3 vertices.
    """
    try:
        import cv2
    except ImportError:
        return None

    if view_corners_xy is None:
        return None
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
    eps = float(epsilon_frac) * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
    if approx.shape[0] < 3:
        return None
    if approx.shape[0] > max_vertices:
        stride = int(np.ceil(approx.shape[0] / max_vertices))
        approx = approx[::stride]
        if approx.shape[0] < 3:
            return None

    flx, fly = float(view_corners_xy[0][0]), float(view_corners_xy[0][1])
    frx, fry = float(view_corners_xy[1][0]), float(view_corners_xy[1][1])
    blx, bly = float(view_corners_xy[3][0]), float(view_corners_xy[3][1])
    # Normalised by the FRAME, which is why a region carries its frame shape.
    h, w = region.frame_shape
    r0, c0 = region.bbox[0], region.bbox[1]
    # cv2 contour points are (x=col, y=row), crop-relative until shifted.
    u = (approx[:, 0].astype(np.float64) + c0) / max(w - 1, 1)
    v = (approx[:, 1].astype(np.float64) + r0) / max(h - 1, 1)
    xs = flx + u * (frx - flx) + v * (blx - flx)
    ys = fly + u * (fry - fly) + v * (bly - fly)
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
    min_valid_depth_pixels: int | None = None,
    return_xy_cloud: bool = False,
    tile: TileFrame | None = None,
) -> ProjectedMask | None:
    """Orthographic counterpart of :func:`project_mask_to_world`.

    The upstream CityNav (``ortho``) observation is an
    orthographic, yaw-rotated square warp, not a camera frustum, so
    there is no ``SensorSpec`` and the perspective depth unprojection
    does not apply. Each mask pixel's world (x, y) is an exact affine
    over the four view-square ground corners ``[FL, FR, BR, BL]``
    (== image ``[TL, TR, BR, BL]``) — the inverse of
    ``observation.crop_ortho_observation``'s warp. The warp is a purely planar
    2-D affine (the square is a parallelogram, no z-perspective skew), so
    the bilinear map below is exact, not an approximation. See
    ``maps/perception/tests/test_ortho_crop_geometry.py`` for
    the forward geometry gate (corner/yaw invariants).

    Depth gates validity (a mask pixel with no finite, positive depth is
    dropped, same rule as the perspective path). Upstream
    ships depth at a different resolution than the RGB the mask lives in
    (224×224 RGB, 256×256 depth); both warp the same view square, so each
    mask pixel's normalized ``(u, v)`` is sampled in the depth grid by
    nearest-neighbor rather than requiring equal shapes.

    Returns a ``ProjectedMask`` matching the perspective path's shape
    (``world_xy``, ``n_valid_pixels``, optional ``world_xy_cloud`` /
    ``tile_cells``). Drop semantics mirror
    :func:`project_mask_to_world`: ``None`` when ``depth`` is absent or
    fewer than ``min_valid_depth_pixels`` mask pixels have finite,
    positive depth. There is no ``max_range_m`` gate (no sensor in the
    orthographic path).

    The set-pixel scan runs over the region's box, never the frame. The
    normalised ``(u, v)`` still divide by the FRAME extent, which is why a
    region carries its frame shape: the box narrows the scan, never the
    geometry.
    """
    if depth is None:
        return None
    depth_arr = np.asarray(depth, dtype=np.float32)
    depth2d = depth_arr if depth_arr.ndim == 2 else depth_arr[:, :, 0]
    threshold = (
        int(min_valid_depth_pixels)
        if min_valid_depth_pixels is not None
        else _MIN_VALID_DEPTH_PIXELS
    )
    rr, cc = region.rows_cols()
    if rr.size == 0:
        return None

    (flx, fly), (frx, fry), _br, (blx, bly) = (
        (float(view_corners_xy[0][0]), float(view_corners_xy[0][1])),
        (float(view_corners_xy[1][0]), float(view_corners_xy[1][1])),
        view_corners_xy[2],
        (float(view_corners_xy[3][0]), float(view_corners_xy[3][1])),
    )
    h, w = region.frame_shape
    u = cc.astype(np.float64) / max(w - 1, 1)
    v = rr.astype(np.float64) / max(h - 1, 1)
    # world = FL + u*(FR-FL) + v*(BL-FL); exact (the square is a
    # parallelogram, so the bilinear map is affine).
    xs = flx + u * (frx - flx) + v * (blx - flx)
    ys = fly + u * (fry - fly) + v * (bly - fly)

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
    n_valid = int(finite.sum())
    if n_valid < threshold:
        return None
    xs = xs[finite]
    ys = ys[finite]
    world_xy = (float(xs.mean()), float(ys.mean()))
    cloud: np.ndarray | None = None
    if return_xy_cloud:
        cloud = np.stack([xs, ys], axis=1).astype(np.float32, copy=False)
    tile_cells: np.ndarray | None = None
    if tile is not None:
        H, W = tile.grid_shape
        col_idx = np.rint((xs - tile.x_min) / tile.cell_size_m - 0.5).astype(np.int64)
        row_idx = np.rint((ys - tile.y_min) / tile.cell_size_m - 0.5).astype(np.int64)
        in_bounds = (row_idx >= 0) & (row_idx < H) & (col_idx >= 0) & (col_idx < W)
        if np.any(in_bounds):
            grid = np.zeros((H, W), dtype=bool)
            grid[row_idx[in_bounds], col_idx[in_bounds]] = True
            gr, gc = np.where(grid)
            tile_cells = np.stack(
                [gr.astype(np.int32, copy=False), gc.astype(np.int32, copy=False)],
                axis=1,
            )
    return ProjectedMask(
        world_xy=world_xy,
        n_valid_pixels=n_valid,
        world_xy_cloud=cloud,
        tile_cells=tile_cells,
    )
