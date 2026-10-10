"""Pose → cropped semantic + height window.

Given a prebuilt raster (semantic class-id PNG + optional height TIFF) and a
world pose, produce a ``fov_m × fov_m`` crop centered on that pose. Out-of-
bounds regions are zero-padded; the crop retains its world extent so the
orthographic raster channels (semantic/height/alpha) stay axis-aligned to
world NE.

``obs.rgb`` comes from a yaw-rotated nadir warp of the ortho at the drone
heading. ``obs.world_window`` still describes the
axis-aligned raster window; it does NOT describe the yaw-rotated RGB.
Camera-derived pixel -> world projection for the RGB lives in
``perception.geolocate.project_mask_to_world`` and must use
``obs.yaw_deg`` + ``obs.sensor`` + ``obs.depth``.

This module is raster-format agnostic: the caller resolves file paths for a
given map / scene and passes them in. Perception never knows what a CityNav
block or a Matterport scan or a warehouse layout is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maps.perception.geometry.sensor_spec import SensorSpec

# The rasters opened here are our own baked orthos, not untrusted uploads;
# a whole-AOI map (~460 MP for calcity at 0.3 m/px) exceeds PIL's
# decompression-bomb ceiling (~178 MP), so lift it deliberately. Covers
# ``load_block_rasters`` and the ``_read_height_tiff`` PIL fallback, whose
# bomb error would otherwise degrade silently to ``height=None``.
try:
    from PIL import Image as _pil_image

    _pil_image.MAX_IMAGE_PIXELS = None
except ImportError:  # pragma: no cover - PIL is a hard dep in practice
    pass


@dataclass
class BlockObservation:
    """Cached per-scene rasters + world-to-pixel transform."""

    map_name: str
    semantic: np.ndarray  # (H, W) int — class id (or R channel of photo)
    height: np.ndarray | None  # (H, W) float — ground elevation in meters
    x_min: float
    y_min: float
    px_per_m: float  # GSD inverse; 1.0 for a 1 m/px raster
    rgb: np.ndarray | None = None  # (H, W, 3) uint8 — photoreal when present
    alpha: np.ndarray | None = None  # (H, W) uint8 — 0 = nodata

    @property
    def shape(self) -> tuple[int, int]:
        return self.semantic.shape

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        """Return (row, col) for a world (x, y). Row grows north-to-south."""
        col = int(round((x - self.x_min) * self.px_per_m))
        # Row 0 is at the top (north); y_min anchors the south edge.
        H = self.semantic.shape[0]
        row = H - 1 - int(round((y - self.y_min) * self.px_per_m))
        return row, col


@dataclass
class Observation:
    """One cropped view around a pose."""

    pose_xy: tuple[float, float]
    altitude_m: float
    yaw_deg: float  # body heading, 0=N, CW-positive
    semantic: np.ndarray  # (h, w) int
    height: np.ndarray | None  # (h, w) float (None if no height raster)
    fov_m: float
    world_window: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    sensor: SensorSpec | None = None  # required for camera-derived pixel projection
    # The model-input crop the reasoner, SAM, and dashboard all consume. On
    # ortho this is the native-resolution crop resampled to the
    # model-input size (rgb_upscale_px); on the sensor backends it is the
    # render. Marks/labels are drawn on this single frame.
    rgb: np.ndarray | None = None  # (h, w, 3) uint8 when the block has photoreal data
    alpha: np.ndarray | None = None  # (h, w) uint8 — 0 for nodata pixels
    depth: np.ndarray | None = (
        None  # (h, w) float32 meters (altitude − height on the ortho crop)
    )
    seg_ids: np.ndarray | None = (
        None  # (h, w) uint16 SensatUrban class ids; reserved (unused on the ortho path)
    )
    # Optional tile georef. When set, adapters project pixel masks to
    # tile-grid (row, col) cells via ``project_mask_to_tile_cells`` and
    # stamp ``metadata["tile_cells"]`` on each ``DetectedObject`` for the
    # downstream mask-IoU re-identification path. ``None`` means the
    # adapter falls back to ``project_mask_to_world``-only outputs
    # (used by harnesses that have no tile, e.g. raw smoke tests).
    tile: "object | None" = None  # perception.geolocate.TileFrame
    # Upstream CityNav (ortho backend) view-square ground corners
    # in world (x, y), order [FL, FR, BR, BL] == image [TL, TR, BR, BL].
    # Set on the orthographic backend; ``perception.geolocate``'s
    # orthographic projection path uses these instead of depth + sensor
    # (the upstream model is orthographic, so a perspective unprojection
    # does not apply).
    ortho_view_corners_xy: tuple | None = None
    # Source video frame index; reserved for a video-based backend. ``None`` on
    # the ortho-only build.
    frame_idx: int | None = None


def view_square_corners_xy(
    pose_xy: tuple[float, float], half_m: float, yaw_deg: float
) -> list[np.ndarray]:
    """The CityNav view-square corners ``[FL, FR, BR, BL]`` in world (x, y).

    ``half_m = altitude - GROUND_LEVEL[map]`` (the square's half-side). Heading
    convention: 0 = North, CW-positive, so ``front = (sin, cos)`` (east, north)
    and ``left = (-cos, sin)``. At due north (yaw=0) FL is the North-West corner
    ``(x - half, y + half)`` — this is what ``project_mask_to_world_ortho``
    inverts, and the geometry-parity test pins.
    """
    yaw_r = np.deg2rad(float(yaw_deg))
    sin_y, cos_y = float(np.sin(yaw_r)), float(np.cos(yaw_r))
    front = np.array([sin_y, cos_y])
    left = np.array([-cos_y, sin_y])
    c = np.array([float(pose_xy[0]), float(pose_xy[1])])
    return [
        c + half_m * front + half_m * left,  # front-left
        c + half_m * front - half_m * left,  # front-right
        c - half_m * front - half_m * left,  # back-right
        c - half_m * front + half_m * left,  # back-left
    ]


def crop_ortho_observation(
    *,
    map_name: str,
    pose_xy: tuple[float, float],
    altitude_m: float,
    yaw_deg: float,
    ground_z: float,
    rgbd_dir: Path | str | None = None,
    m_per_px: float = 0.10,
    max_px: int = 2048,
    sensor: "SensorSpec | None" = None,
    ortho: dict | None = None,
) -> dict:
    """The single nadir observation renderer: crop the hole-free 2D ortho over
    the view rectangle.

    ``ortho`` is the raster to crop, in hand: ``{rgb, height, alpha, x_min,
    y_min, px_per_m}``, the shape :func:`camera_model.load_ortho_or_rasters`
    returns. Given, nothing is loaded: a flight hands the imagery its own
    run captured (``benchmarks/flight/run_imagery.py``); the bench
    leaves it unset and crops the bake's raster, which is its camera.

    Warps ``rgb`` + ``height`` + ``validity`` from the canonical ortho
    (:func:`camera_model.load_ortho_or_rasters`), then derives
    ``depth = altitude_m - height``.

    The view rectangle is one of two shapes:

    - ``sensor=None`` — the CityNav view square: ``side = 2*(altitude_m -
      ground_z)``, yaw-rotated, corners ``[FL,FR,BR,BL]``, output square-pixel
      at the ortho's own resolution (capped at ``max_px``). Byte-identical to
      the historical contract.
    - a nadir ``SensorSpec`` — the sensor's ground footprint at this AGL
      (``2·AGL·tan(fov/2)`` per axis, via :mod:`maps.perception.ortho.
      sensor_frame`), warped to the sensor's native ``image_width ×
      image_height``. Still an ORTHOGRAPHIC warp — geometrically the nadir
      frame a real camera would see over flat ground, photometrically capped
      by the source imagery — so pixel→world stays on the ortho projector
      over ``view_corners_xy``, never the perspective frustum.

    Returns ``{rgb, depth, validity, view_corners_xy, world_window, fov_m}``.
    Out-of-raster border is ``validity=0`` with ``depth=0`` so the projection's
    validity gate drops it (instead of trusting an ``altitude - 0`` depth).
    """
    import cv2  # noqa: PLC0415

    from maps.perception.ortho.camera_model import load_ortho_or_rasters  # noqa: PLC0415

    if ortho is None:
        ortho = load_ortho_or_rasters(map_name, rgbd_dir=rgbd_dir, m_per_px=m_per_px)
    if ortho is None:
        raise FileNotFoundError(f"No ortho for {map_name!r}: pass rgbd_dir (the block's rasters).")
    rgb_map, height_map, alpha_map = ortho["rgb"], ortho["height"], ortho["alpha"]
    ox_min, oy_min, ppm = (
        float(ortho["x_min"]),
        float(ortho["y_min"]),
        float(ortho["px_per_m"]),
    )
    H_o = rgb_map.shape[0]

    if sensor is not None:
        from maps.perception.ortho.sensor_frame import (  # noqa: PLC0415
            footprint_corners_xy,
            footprint_m,
        )

        agl = max(1.0, float(altitude_m) - float(ground_z))
        width_m, height_m = footprint_m(sensor, agl)
        corners_world = footprint_corners_xy(pose_xy, width_m, height_m, yaw_deg)
        out_w = int(sensor.intrinsics.image_width)
        out_h = int(sensor.intrinsics.image_height)
        view_extent_m = float(width_m)
    else:
        # View square: side = 2*(altitude - ground); corners [FL,FR,BR,BL].
        half = max(1.0, float(altitude_m) - float(ground_z))
        corners_world = view_square_corners_xy(pose_xy, half, yaw_deg)
        view_extent_m = float(2.0 * half)
        out_w = out_h = None  # derived from the ortho span below

    corners_px = np.array(
        [[(x - ox_min) * ppm, (H_o - 1) - (y - oy_min) * ppm] for (x, y) in corners_world],
        dtype=np.float32,
    )
    if out_w is None:
        span = int(
            round(
                max(
                    float(np.hypot(*(corners_px[1] - corners_px[0]))),
                    float(np.hypot(*(corners_px[3] - corners_px[0]))),
                )
            )
        )
        out_w = out_h = max(1, min(span, int(max_px)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners_px, dst)
    # Warp from the window's own neighbourhood of the map, never the whole
    # map. Cal City's raster is 34,035 x 44,853 x 3, past the 2^31 elements
    # one OpenCV Mat addresses, and a warp handed the whole of it read the
    # wrong memory: a black crop with a sliver of picture and validity 255
    # throughout (sortie 20260913T014538Z, the look at prior_4533). The
    # neighbourhood is the corners' box plus INTER_LINEAR's neighbours, and
    # the transform is M composed with the box's offset, so an in-raster
    # window samples exactly the pixels the whole-map warp did; the
    # off-raster remainder is the constant border either way.
    pad = 2
    x0 = max(0, int(np.floor(corners_px[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(corners_px[:, 1].min())) - pad)
    x1 = min(int(rgb_map.shape[1]), int(np.ceil(corners_px[:, 0].max())) + pad + 1)
    y1 = min(int(H_o), int(np.ceil(corners_px[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        rgb = np.zeros((out_h, out_w, 3), np.uint8)
        height = np.zeros((out_h, out_w), np.float32)
        validity = np.zeros((out_h, out_w), np.uint8)
    else:
        shift = np.array([[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]], dtype=np.float64)
        M_box = np.asarray(M, np.float64) @ shift
        rgb = cv2.warpPerspective(
            np.ascontiguousarray(rgb_map[y0:y1, x0:x1]), M_box, (out_w, out_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        height = cv2.warpPerspective(
            np.ascontiguousarray(height_map[y0:y1, x0:x1], np.float32), M_box, (out_w, out_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        validity = cv2.warpPerspective(
            np.ascontiguousarray(alpha_map[y0:y1, x0:x1]), M_box, (out_w, out_h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    depth = (float(altitude_m) - height).astype(np.float32)
    # No-data (out-of-raster / boundary): 0.0 sentinel — the ortho projection
    # gate drops depth <= 0, and 0.0 encodes cleanly to the wedge depth PNG.
    depth[validity == 0] = 0.0
    corners_xy = [(float(x), float(y)) for (x, y) in corners_world]
    xs = [p[0] for p in corners_xy]
    ys = [p[1] for p in corners_xy]
    return {
        "rgb": np.ascontiguousarray(rgb, np.uint8),
        "depth": depth,
        "validity": np.ascontiguousarray(validity, np.uint8),
        "view_corners_xy": corners_xy,
        "world_window": (min(xs), min(ys), max(xs), max(ys)),
        "fov_m": view_extent_m,
    }


def load_block_rasters(
    *,
    map_name: str,
    png_path: Path,
    tiff_path: Path | None,
    x_min: float,
    y_min: float,
    px_per_m: float = 1.0,
    grid_shape: tuple[int, int] | None = None,
) -> BlockObservation:
    """Open a semantic PNG (+ optional height TIFF) and build a BlockObservation.

    If ``png_path`` is missing and ``grid_shape`` is set, returns a zero-filled
    BlockObservation so crop_observation still produces a valid world window
    for synthetic perception (which reads GT directly and ignores the raster).
    Real perception tiers must ensure the PNG exists before calling.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow required to read perception rasters") from e

    if not png_path.exists():
        if grid_shape is None:
            raise FileNotFoundError(
                f"Missing rasterized tile: {png_path} and no grid_shape fallback"
            )
        return BlockObservation(
            map_name=map_name,
            semantic=np.zeros(grid_shape, dtype=np.int32),
            height=None,
            x_min=x_min,
            y_min=y_min,
            px_per_m=px_per_m,
        )

    img = Image.open(png_path)
    arr = np.asarray(img)
    rgb: np.ndarray | None = None
    alpha: np.ndarray | None = None
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(np.uint8)
        if arr.shape[2] == 4:
            alpha = arr[..., 3].astype(np.uint8)
        semantic = arr[..., 0].astype(np.int32)
    else:
        semantic = arr.astype(np.int32)

    height = None
    if tiff_path is not None and tiff_path.exists():
        height = _read_height_tiff(tiff_path)

    return BlockObservation(
        map_name=map_name,
        semantic=semantic,
        height=height,
        x_min=x_min,
        y_min=y_min,
        px_per_m=px_per_m,
        rgb=rgb,
        alpha=alpha,
    )


def _read_height_tiff(path: Path) -> np.ndarray | None:
    """Read a height-map TIFF, tolerating GeoTIFF + plain TIFF variants.

    PIL refuses many scientific (float64, multi-sample) GeoTIFFs. tifffile
    handles both; fall back to rasterio if tifffile missing. Returns None
    if every reader declines.
    """
    try:
        import tifffile

        arr = tifffile.imread(str(path))
        return np.asarray(arr).astype(np.float32)
    except Exception:
        pass
    try:
        import rasterio  # type: ignore[import-not-found]

        with rasterio.open(str(path)) as ds:
            return ds.read(1).astype(np.float32)
    except Exception:
        pass
    try:
        from PIL import Image

        return np.asarray(Image.open(path)).astype(np.float32)
    except Exception:
        return None


def observation_extent_m(
    altitude_m: float, ground_z: float, sensor: SensorSpec | None = None
) -> float:
    """The observation's ground extent in meters — the crop window's honest
    ceiling (``obs_span_m``). Sensorless, it is the view-square side
    ``2·(altitude − ground)``; with a nadir sensor it is the footprint's long
    side at this AGL. One function so the mission engine and the search loop
    cannot disagree about what the observation showed.
    """
    agl = max(1.0, float(altitude_m) - float(ground_z))
    if sensor is None:
        return 2.0 * agl
    from maps.perception.ortho.sensor_frame import footprint_m  # noqa: PLC0415

    width_m, height_m = footprint_m(sensor, agl)
    return float(max(width_m, height_m))


def observation_minor_extent_m(
    altitude_m: float, ground_z: float, sensor: SensorSpec | None = None
) -> float:
    """The observation's SHORT ground side at this AGL — the row pitch a
    2-D coverage tiling can rely on whichever way the footprint is turned.
    Sensorless the view square has one side; with a nadir sensor it is
    ``min`` where ``observation_extent_m`` is ``max``, same source, so the
    two cannot disagree about the footprint they describe.
    """
    agl = max(1.0, float(altitude_m) - float(ground_z))
    if sensor is None:
        return 2.0 * agl
    from maps.perception.ortho.sensor_frame import footprint_m  # noqa: PLC0415

    width_m, height_m = footprint_m(sensor, agl)
    return float(min(width_m, height_m))


def resample_to_model_input(
    rgb: np.ndarray | None, rgb_upscale_px: int | None
) -> np.ndarray | None:
    """Deterministically resample an RGB crop to ``rgb_upscale_px`` square.

    Model-input preprocessing: normalizes the native-resolution crop to the
    fixed model-input size, UP or DOWN as needed (the native pixel span varies
    with altitude). Adds no ground information past native -- it only lifts the
    crop into the input regime the downstream detector / VLM expect, so
    on-image marks and labels stay legible at a consistent resolution.
    Lanczos for upscaling, INTER_AREA for downscaling (the area-average kernel
    anti-aliases shrinks far better than Lanczos). No-op when the target is
    ``None``, non-positive, or already the crop's size.
    """
    if rgb is None or rgb_upscale_px is None:
        return rgb
    target = int(rgb_upscale_px)
    if target <= 0 or (rgb.shape[0] == target and rgb.shape[1] == target):
        return rgb
    import cv2  # noqa: PLC0415  (lazy: cv2 ships with the ortho extra)

    interp = cv2.INTER_AREA if target < rgb.shape[0] else cv2.INTER_LANCZOS4
    return cv2.resize(rgb, (target, target), interpolation=interp)


def crop_observation(
    block: BlockObservation,
    pose_xy: tuple[float, float],
    fov_m: float,
    altitude_m: float = 50.0,
    *,
    yaw_deg: float = 0.0,
    sensor: SensorSpec | None = None,
    rgbd_dir: Path | str | None = None,
    map_name: str | None = None,
    rgb_upscale_px: int | None = None,
) -> Observation:
    """Crop a `fov_m × fov_m` window centered on the pose.

    Out-of-bounds regions are zero-padded. The orthographic raster
    channels (``semantic``, ``height``, ``alpha``) are axis-aligned to
    world NE; ``world_window`` describes their extent.

    ``obs.rgb`` / ``obs.depth``
    and the view geometry come from :func:`crop_ortho_observation` — the
    hole-free 2D ortho cropped over the view rectangle, ``depth = altitude_m −
    height``. With ``sensor=None`` that rectangle is the altitude-driven view
    square (side = ``2·(altitude_m − GROUND_LEVEL[map])``) and ``obs.rgb`` is
    resampled to ``rgb_upscale_px`` square as model-input preprocessing. With a
    nadir ``sensor``, it is the sensor's ground footprint at this AGL warped to
    the sensor's native ``image_width × image_height`` — already model-input
    sized, so ``rgb_upscale_px`` does not apply. Either way the warp is
    ORTHOGRAPHIC and pixel→world goes through the orthographic geolocate path
    keyed off ``ortho_view_corners_xy`` (an ``Observation`` carrying both
    ``sensor`` and ortho corners is an ortho rendering of that sensor's
    footprint — the corners, not the frustum, are its projection contract).
    The belief-grid semantic/height/alpha channels are still cropped from the
    1 m/px raster over the same axis-aligned bbox. The ortho is read from
    the rasters under ``rgbd_dir``.
    """
    H, W = block.shape

    if rgbd_dir is None or map_name is None:
        raise ValueError(
            "crop_observation requires rgbd_dir and map_name "
            "(the 0.1 m/px rasters dir + block name)"
        )
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    if map_name not in GROUND_LEVEL:
        raise KeyError(
            f"citynav ortho: map {map_name!r} not in GROUND_LEVEL "
            f"({len(GROUND_LEVEL)} known blocks)."
        )
    oc = crop_ortho_observation(
        map_name=map_name,
        pose_xy=pose_xy,
        altitude_m=altitude_m,
        yaw_deg=yaw_deg,
        ground_z=float(GROUND_LEVEL[map_name]),
        rgbd_dir=rgbd_dir,
        sensor=sensor,
    )
    # Belief-grid underlay (semantic/height/alpha) cropped over the same
    # world window — unchanged from the vendored path.
    nx0, ny0, nx1, ny1 = oc["world_window"]
    nr_top, nc_left = block.world_to_px(nx0, ny1)
    nr_bot, nc_right = block.world_to_px(nx1, ny0)
    nr_min, nr_max = sorted((nr_top, nr_bot))
    nc_min, nc_max = sorted((nc_left, nc_right))
    nsem = _safe_crop(block.semantic, nr_min, nr_max, nc_min, nc_max, H, W, fill=0)
    nhgt = None
    if block.height is not None:
        nhgt = _safe_crop(
            block.height, nr_min, nr_max, nc_min, nc_max, H, W, fill=0.0
        )
    nalpha = None
    if block.alpha is not None:
        nalpha = _safe_crop(
            block.alpha, nr_min, nr_max, nc_min, nc_max, H, W, fill=0
        )
    return Observation(
        pose_xy=pose_xy,
        altitude_m=altitude_m,
        yaw_deg=yaw_deg,
        sensor=sensor,
        semantic=nsem,
        height=nhgt,
        fov_m=oc["fov_m"],
        world_window=oc["world_window"],
        # A sensor frame is already at the sensor's native input size; the
        # square-normalizing upscale is the sensorless path's contract only.
        rgb=(
            oc["rgb"]
            if sensor is not None
            else resample_to_model_input(oc["rgb"], rgb_upscale_px)
        ),
        alpha=nalpha,
        depth=oc["depth"],
        ortho_view_corners_xy=tuple(oc["view_corners_xy"]),
    )



def _safe_crop(
    arr: np.ndarray,
    r_min: int,
    r_max: int,
    c_min: int,
    c_max: int,
    H: int,
    W: int,
    *,
    fill: float,
) -> np.ndarray:
    """Crop with zero-padding when the window extends outside the raster."""
    out_shape = (r_max - r_min + 1, c_max - c_min + 1)
    out = np.full(out_shape, fill, dtype=arr.dtype)
    r0, r1 = max(0, r_min), min(H - 1, r_max)
    c0, c1 = max(0, c_min), min(W - 1, c_max)
    if r1 < r0 or c1 < c0:
        return out
    dr0, dr1 = r0 - r_min, r1 - r_min + 1
    dc0, dc1 = c0 - c_min, c1 - c_min + 1
    out[dr0:dr1, dc0:dc1] = arr[r0 : r1 + 1, c0 : c1 + 1]
    return out
