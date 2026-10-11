"""Pose -> nadir observation: the ortho crop the segmenter and reasoner see.

Given a block's rasters and a world pose, :func:`crop_observation` warps the
hole-free 2D ortho over the view rectangle (the CityNav view square, or a
nadir sensor's ground footprint) into ``obs.rgb`` + ``obs.depth``, and records
the rectangle's ground corners in ``obs.ortho_view_corners_xy``: pixel ->
world goes through those corners (``geometry.geolocate``), never a frustum.

This module is raster-format agnostic: the caller resolves file paths for a
given map / scene and passes them in. Perception never knows what a CityNav
block or a Matterport scan or a warehouse layout is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maps.perception.geometry.sensor_spec import SensorSpec
from maps.perception.ortho.sensor_frame import footprint_corners_xy, footprint_m, frame_corners_px

#: The sensorless view square's output side cap, in pixels.
_MAX_PX = 2048


@dataclass
class Observation:
    """One nadir view around a pose."""

    pose_xy: tuple[float, float]
    altitude_m: float
    yaw_deg: float  # body heading, 0=N, CW-positive
    world_window: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    sensor: SensorSpec | None = None
    # The model-input crop the reasoner and SAM consume; marks/labels are
    # drawn on this single frame.
    rgb: np.ndarray | None = None  # (h, w, 3) uint8
    alpha: np.ndarray | None = None  # (h, w) uint8 — 0 for nodata pixels (flight imagery)
    depth: np.ndarray | None = None  # (h, w) float32 meters (altitude − height)
    # Tile georef the detections are rasterized onto
    # (``perception.geometry.geolocate.TileFrame``).
    tile: "object | None" = None
    # View-rectangle ground corners in world (x, y), order [FL, FR, BR, BL]
    # == image [TL, TR, BR, BL]; the orthographic projection inverts them.
    ortho_view_corners_xy: tuple | None = None


def crop_ortho_observation(
    *,
    map_name: str,
    pose_xy: tuple[float, float],
    altitude_m: float,
    yaw_deg: float,
    ground_z: float,
    rgbd_dir: Path | str | None = None,
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
      at the ortho's own resolution (capped at ``_MAX_PX``).
    - a nadir ``SensorSpec`` — the sensor's ground footprint at this AGL
      (``2·AGL·tan(fov/2)`` per axis, via :mod:`maps.perception.ortho.
      sensor_frame`), warped to the sensor's native ``image_width ×
      image_height``. Still an ORTHOGRAPHIC warp — geometrically the nadir
      frame a real camera would see over flat ground, photometrically capped
      by the source imagery — so pixel→world stays on the ortho projector
      over ``view_corners_xy``, never the perspective frustum.

    Returns ``{rgb, depth, validity, view_corners_xy, world_window}``.
    Out-of-raster border is ``validity=0`` with ``depth=0`` so the projection's
    validity gate drops it (instead of trusting an ``altitude - 0`` depth).
    """
    import cv2  # noqa: PLC0415

    from maps.perception.ortho.camera_model import load_ortho_or_rasters  # noqa: PLC0415

    if ortho is None:
        ortho = load_ortho_or_rasters(map_name, rgbd_dir=rgbd_dir)
    if ortho is None:
        raise FileNotFoundError(f"No ortho for {map_name!r}: pass rgbd_dir (the block's rasters).")
    rgb_map, height_map, alpha_map = ortho["rgb"], ortho["height"], ortho["alpha"]
    ox_min, oy_min, ppm = (
        float(ortho["x_min"]),
        float(ortho["y_min"]),
        float(ortho["px_per_m"]),
    )
    H_o = rgb_map.shape[0]

    width_m, height_m = _ground_sides_m(altitude_m, ground_z, sensor)
    corners_world = footprint_corners_xy(pose_xy, width_m, height_m, yaw_deg)
    if sensor is not None:
        out_w = int(sensor.intrinsics.image_width)
        out_h = int(sensor.intrinsics.image_height)
    else:
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
        out_w = out_h = max(1, min(span, _MAX_PX))
    M = cv2.getPerspectiveTransform(corners_px, frame_corners_px(out_w, out_h))
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
    }


def _ground_sides_m(
    altitude_m: float, ground_z: float, sensor: SensorSpec | None
) -> tuple[float, float]:
    """The observation's ground ``(width, height)`` in meters at this AGL:
    sensorless the view square, ``2·(altitude − ground)`` a side; with a nadir
    sensor its footprint (``2·AGL·tan(fov/2)`` per axis). The crop, the
    mission engine and the search loop all read the footprint here, so they
    cannot disagree about what the observation showed."""
    agl = max(1.0, float(altitude_m) - float(ground_z))
    if sensor is None:
        return 2.0 * agl, 2.0 * agl
    return footprint_m(sensor, agl)


def observation_extent_m(
    altitude_m: float, ground_z: float, sensor: SensorSpec | None = None
) -> float:
    """The observation's LONG ground side in meters, the crop window's honest
    ceiling (``obs_span_m``)."""
    return float(max(_ground_sides_m(altitude_m, ground_z, sensor)))


def observation_minor_extent_m(
    altitude_m: float, ground_z: float, sensor: SensorSpec | None = None
) -> float:
    """The observation's SHORT ground side in meters, the row pitch a 2-D
    coverage tiling can rely on whichever way the footprint is turned."""
    return float(min(_ground_sides_m(altitude_m, ground_z, sensor)))


def crop_observation(
    pose_xy: tuple[float, float],
    *,
    altitude_m: float,
    rgbd_dir: Path | str,
    map_name: str,
    yaw_deg: float = 0.0,
    sensor: SensorSpec | None = None,
) -> Observation:
    """The observation at a pose: :func:`crop_ortho_observation` over the
    block's ortho, with ``ground_z = GROUND_LEVEL[map_name]``.

    With ``sensor=None`` the view rectangle is the altitude-driven view
    square (side = ``2·(altitude_m − GROUND_LEVEL[map])``); with a nadir
    ``sensor`` it is the sensor's ground footprint at this AGL warped to the
    sensor's native ``image_width × image_height``. Either way the warp is
    ORTHOGRAPHIC and pixel→world goes through the orthographic geolocate path
    keyed off ``ortho_view_corners_xy`` (an ``Observation`` carrying both
    ``sensor`` and ortho corners is an ortho rendering of that sensor's
    footprint — the corners, not the frustum, are its projection contract).
    """
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
    return Observation(
        pose_xy=pose_xy,
        altitude_m=altitude_m,
        yaw_deg=yaw_deg,
        world_window=oc["world_window"],
        sensor=sensor,
        rgb=oc["rgb"],
        depth=oc["depth"],
        ortho_view_corners_xy=tuple(oc["view_corners_xy"]),
    )
