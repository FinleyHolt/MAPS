"""A camera's ground footprint and the frame it sees over a baked ortho.

A pinhole over flat ground is exactly a homography, so the four footprint
corners plus ``cv2.getPerspectiveTransform`` are the perspective render;
relief displacement is not modelled. Every function takes the ortho and its
bounds as arguments.
"""

from __future__ import annotations

import math

import numpy as np

from maps.perception.geometry.sensor_spec import SensorSpec

#: The taps ``INTER_LANCZOS4`` reads either side of a sample, so the margin
#: a rendered window needs around its footprint.
TAP_MARGIN_PX = 4


def footprint_m(sensor, agl_m: float) -> tuple[float, float]:
    """Ground footprint ``(width, height)`` in meters for a nadir ``sensor`` at
    ``agl_m`` above a flat surface: ``2 * agl * tan(fov/2)`` per axis."""
    intr = sensor.intrinsics
    return (
        2.0 * agl_m * math.tan(math.radians(intr.hfov_deg) / 2.0),
        2.0 * agl_m * math.tan(math.radians(intr.vfov_deg) / 2.0),
    )


def footprint_corners_xy(
    pose_xy: tuple[float, float], width_m: float, height_m: float, yaw_deg: float
) -> list[tuple[float, float]]:
    """Footprint corners ``[FL, FR, BR, BL]`` in world (x, y), image order
    ``[TL, TR, BR, BL]``.

    Heading 0 = North, CW-positive, ``front = (sin, cos)``, ``left = (-cos,
    sin)``; ``front`` spans the image height, ``left`` the image width. At
    yaw 0 FL is the North-West corner. The CityNav view square is the case
    ``width_m == height_m == 2·(altitude − ground)``, and this order is what
    ``geolocate``'s ortho projectors invert.
    """
    yaw_r = math.radians(float(yaw_deg))
    sin_y, cos_y = math.sin(yaw_r), math.cos(yaw_r)
    front = np.array([sin_y, cos_y]) * (height_m / 2.0)
    left = np.array([-cos_y, sin_y]) * (width_m / 2.0)
    c = np.array([float(pose_xy[0]), float(pose_xy[1])])
    return [
        tuple(c + front + left),  # front-left  -> image top-left
        tuple(c + front - left),  # front-right -> image top-right
        tuple(c - front - left),  # back-right  -> image bottom-right
        tuple(c - front + left),  # back-left   -> image bottom-left
    ]


def ground_corners_xy(
    sensor,
    pose_xy: tuple[float, float],
    agl_m: float,
    yaw_deg: float,
    *,
    airframe_roll_deg: float = 0.0,
    airframe_pitch_deg: float = 0.0,
    ground_z: float = 0.0,
) -> list[tuple[float, float]]:
    """Where the frame's four corner pixel centres meet flat ground, image
    order ``[TL, TR, BR, BL]``, through the projector (mount angles and
    airframe tilt included); :func:`footprint_corners_xy` is the level,
    mount-aligned case to within half a pixel.
    """
    from maps.perception.geometry.geolocate import project_pixels_to_world  # noqa: PLC0415  (cv2)

    intr = sensor.intrinsics
    w, h = intr.image_width, intr.image_height
    camera = np.array([float(pose_xy[0]), float(pose_xy[1]), float(ground_z) + float(agl_m)])
    # The ray is taken at ground range, not unit depth: the projector hands
    # back float32 world points, and a unit-depth point beside a camera at
    # a few thousand metres carries a 1e-4 m quantisation that the range
    # then scales into a tenth of a metre at the corner.
    reach = np.full((1, 1), max(float(agl_m), 1.0))
    out: list[tuple[float, float]] = []
    for r, c in ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)):
        # The point at ground range along the ray fixes the ray; the ground
        # plane fixes where along it the corner lands.
        p1 = project_pixels_to_world(
            reach,
            pose_xy=pose_xy,
            altitude_m=camera[2],
            yaw_deg=yaw_deg,
            sensor=sensor,
            airframe_roll_deg=airframe_roll_deg,
            airframe_pitch_deg=airframe_pitch_deg,
            origin_rc=(r, c),
        )[0, 0]
        ray = np.asarray(p1, dtype=np.float64) - camera
        if ray[2] >= 0.0:
            raise ValueError("a frame corner looks at or above the horizon; no ground under it")
        t = (camera[2] - float(ground_z)) / -ray[2]
        ground = camera + t * ray
        out.append((float(ground[0]), float(ground[1])))
    return out


def render_frame(
    ortho_rgb: np.ndarray,
    bounds,
    px_per_m: float,
    *,
    sensor,
    pose_xy: tuple[float, float],
    agl_m: float,
    yaw_deg: float = 0.0,
    airframe_roll_deg: float = 0.0,
    airframe_pitch_deg: float = 0.0,
) -> np.ndarray:
    """The frame ``sensor`` sees over ``ortho_rgb`` (north-up, row 0 at
    ``bounds.y_max``) through the footprint :func:`ground_corners_xy` gives.
    Raises when a corner falls outside ``bounds``: a black-bordered frame
    would misstate coverage.
    """
    corners = ground_corners_xy(
        sensor,
        pose_xy,
        agl_m,
        yaw_deg,
        airframe_roll_deg=airframe_roll_deg,
        airframe_pitch_deg=airframe_pitch_deg,
    )
    for x, y in corners:
        if not (bounds.x_min <= x <= bounds.x_max and bounds.y_min <= y <= bounds.y_max):
            raise ValueError(
                f"footprint corner ({x:.1f}, {y:.1f}) is outside the block "
                f"{tuple(round(v, 1) for v in bounds)}: the frame would carry "
                f"no-data border. Move the aim point inward or lower the altitude."
            )
    src = ortho_px(corners, bounds.x_min, bounds.y_min, px_per_m, ortho_rgb.shape[0])
    out_w, out_h = sensor.intrinsics.image_width, sensor.intrinsics.image_height
    return np.ascontiguousarray(warp_window(ortho_rgb, src, out_w, out_h), np.uint8)


def ortho_px(points_xy, x_min: float, y_min: float, px_per_m: float, rows: int) -> np.ndarray:
    """World ``(x, y)`` points as ``float32`` pixel ``(col, row)`` of a north-up
    ortho ``rows`` tall whose south-west corner is ``(x_min, y_min)``."""
    return np.array(
        [[(x - x_min) * px_per_m, (rows - 1) - (y - y_min) * px_per_m] for (x, y) in points_xy],
        dtype=np.float32,
    )


def source_map(src_px: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """``(map_x, map_y)``: the ortho pixel every pixel of a ``width x height``
    frame reads (the inverted four-corner homography, float64 evaluated,
    float32 returned)."""
    import cv2  # noqa: PLC0415

    inv = np.linalg.inv(
        np.asarray(cv2.getPerspectiveTransform(src_px, frame_corners_px(width, height)), np.float64)
    )
    x, y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    w = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2]
    return (
        ((inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]) / w).astype(np.float32),
        ((inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]) / w).astype(np.float32),
    )


def warp_window(ortho_rgb: np.ndarray, src_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """The frame the corners ``src_px`` cut from ``ortho_rgb``, reading only
    the window under them (plus :data:`TAP_MARGIN_PX`): OpenCV's ``remap``
    refuses a source of 32,767 px or more. The window offset is a whole
    number of pixels, so every tap reads what a whole-raster warp would.
    """
    import cv2  # noqa: PLC0415

    map_x, map_y = source_map(src_px, width, height)
    x0 = max(0, int(np.floor(map_x.min())) - TAP_MARGIN_PX)
    y0 = max(0, int(np.floor(map_y.min())) - TAP_MARGIN_PX)
    x1 = min(ortho_rgb.shape[1], int(np.ceil(map_x.max())) + TAP_MARGIN_PX + 1)
    y1 = min(ortho_rgb.shape[0], int(np.ceil(map_y.max())) + TAP_MARGIN_PX + 1)
    return cv2.remap(
        np.ascontiguousarray(ortho_rgb[y0:y1, x0:x1]),
        map_x - np.float32(x0),
        map_y - np.float32(y0),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def frame_corners_px(width: int, height: int) -> np.ndarray:
    """The frame's own corner pixels in image order ``[TL, TR, BR, BL]``, the
    destination of every homography onto a frame."""
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def served_description(spec: SensorSpec, *, agl_m: float | None) -> str:
    """The rig's description as served. A description that states a footprint
    states the sensor's own (:func:`footprint_m`) at ``agl_m``, the
    height above ground the mission flies; one that states none is served as
    written."""
    if "{" not in spec.description:
        return spec.description
    if agl_m is None:
        raise ValueError(
            f"{spec.name} states its footprint at the height the mission flies: "
            "no height above ground was given"
        )
    w, h = footprint_m(spec, agl_m)
    gsd = w / spec.intrinsics.image_width
    return spec.description.format(
        agl_ft=agl_m / 0.3048, w=w, h=h, gsd=gsd, px_5m=5.0 / gsd, px_half_m=0.5 / gsd
    )
