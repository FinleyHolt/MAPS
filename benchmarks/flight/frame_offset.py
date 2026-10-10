"""A frame's offset from the map rendered at its pose, as a pose correction.

The map rendered at the frame's own pose, its centre taken as one template,
located in the whole frame by the locator's edge structure at a quarter of
the frame's resolution (:func:`offset`). The shift it finds is the ground the
pose is off by, through the projector's own Jacobian at the nadir point
(:func:`ground_correction`); its peak says how sure the frame is of it.
:func:`similarity` reads the same with the content turned as well, which is
how a whole north-up mosaic is registered as one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from benchmarks.flight import locator

#: The frame and the render are read at this fraction of their resolution.
DOWN = 4
#: The part of the render, about its centre, that is the template.
TEMPLATE_FRACTION = 0.5

#: The frame's structure and the render's, computed side by side.
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame_offset")


def reach_px(w: int, h: int) -> tuple[float, float]:
    """How far :func:`offset` can read, in frame pixels either way: a peak there is the search's wall."""
    return w * (1.0 - TEMPLATE_FRACTION) / 2.0, h * (1.0 - TEMPLATE_FRACTION) / 2.0


def structure(gray: np.ndarray, down: int = DOWN) -> np.ndarray:
    """The edge structure at ``1/down`` of the image's size, float32 ``(3, h, w)``:
    averaged down first and the operator run there. ``down=1`` reads an image
    already at that size."""
    import cv2  # noqa: PLC0415

    small = np.asarray(gray) if down == 1 else cv2.resize(
        np.asarray(gray), (gray.shape[1] // down, gray.shape[0] // down), interpolation=cv2.INTER_AREA)
    return np.stack(locator.edge_maps(small, template_half=locator.TEMPLATE_HALF)).astype(np.float32)


def render_sensor(sensor, down: int = DOWN):
    """``sensor`` at ``1/down`` of its pixels, the same fields of view: the
    expected view is rendered straight onto the grid the structure is read at."""
    from maps.perception.ortho.camera_model import CameraIntrinsics  # noqa: PLC0415

    intr = sensor.intrinsics
    return replace(sensor, intrinsics=CameraIntrinsics(hfov_deg=intr.hfov_deg, vfov_deg=intr.vfov_deg,
                                                       image_width=intr.image_width // down,
                                                       image_height=intr.image_height // down))


def offset(frame_rgb: np.ndarray, render_rgb: np.ndarray) -> tuple[float, float, float]:
    """``(du, dv, response)``: the frame's content sits ``(du, dv)`` frame pixels from where the render puts it.

    ``frame(p) == render(p - d)``. The response is the normalised correlation
    at the peak, near one for the same ground and near zero for none; a render
    with no structure to locate is no match, ``(0, 0, 0)``. The render is the
    frame's size or ``1/DOWN`` of it (:func:`render_sensor`).
    """
    fh, rh = np.asarray(frame_rgb).shape[0], np.asarray(render_rgb).shape[0]
    render_down = DOWN if rh == fh else 1 if rh == fh // DOWN else None
    if render_down is None:
        raise ValueError(f"the render is {rh} rows against a frame of {fh}: the frame's size or 1/{DOWN} of it")
    fa = _POOL.submit(structure, locator.gray(render_rgb), render_down)
    fb = _POOL.submit(structure, locator.gray(frame_rgb))
    a, b = fa.result(), fb.result()
    _c, h, w = a.shape
    th, tw = int(h * TEMPLATE_FRACTION), int(w * TEMPLATE_FRACTION)
    y0, x0 = (h - th) // 2, (w - tw) // 2
    score = locator._joint_ncc(b, a[:, y0:y0 + th, x0:x0 + tw])
    if score is None:
        return 0.0, 0.0, 0.0
    at, peak, _margin = locator._peak(score, max(1, locator.UNIQUE_RADIUS // DOWN))
    return float(at[0] - x0) * DOWN, float(at[1] - y0) * DOWN, float(peak)


def similarity(frame_rgb: np.ndarray, render_rgb: np.ndarray, *, yaw_reach_deg: float, yaw_step_deg: float = 1.0
               ) -> tuple[float, float, float, float]:
    """``(du, dv, yaw_deg, response)``: :func:`offset` with the frame's content turned as well.

    The yaw is the turn of the content about its centre against the render,
    in the heading's sense, the sweep's best response over ``+-yaw_reach_deg``
    at ``yaw_step_deg`` and again at a quarter step about that; ``du`` and
    ``dv`` are the offset of the content turned back by it.
    """
    import cv2  # noqa: PLC0415

    h, w = np.asarray(frame_rgb).shape[:2]
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)

    def read(yaw: float) -> tuple[float, float, float, float]:
        # cv2's positive angle turns the picture counter-clockwise on screen; a
        # heading-sense turn is clockwise there, so turning BACK by yaw is -yaw.
        turned = cv2.warpAffine(np.asarray(frame_rgb), cv2.getRotationMatrix2D(centre, -yaw, 1.0), (w, h),
                                flags=cv2.INTER_LINEAR)
        du, dv, response = offset(turned, render_rgb)
        return response, du, dv, yaw

    best = max(read(float(y)) for y in np.arange(-yaw_reach_deg, yaw_reach_deg + 1e-9, yaw_step_deg))
    fine = np.arange(best[3] - yaw_step_deg, best[3] + yaw_step_deg + 1e-9, yaw_step_deg / 4.0)
    best = max([best] + [read(float(y)) for y in fine if abs(y - best[3]) > 1e-9])
    response, du, dv, yaw = best
    return du, dv, yaw, response


def _jacobian(pose, sensor, *, altitude_m: float, ground_z: float, at_xy: tuple[float, float]
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """``(pixel of at_xy, d pixel / d (east, north))`` through the projector at ``pose``, or None off it.

    ``pose`` carries ``pose_xy``, ``yaw_deg``, ``roll_deg`` and ``pitch_deg``
    in aerospace signs; the projector's pitch is nose down, so it enters negated.
    """
    from maps.perception.geometry.geolocate import project_world_to_pixels  # noqa: PLC0415

    x, y = at_xy
    world = np.asarray([(x, y, ground_z), (x + 1.0, y, ground_z), (x, y + 1.0, ground_z)], dtype=np.float64)
    uv = np.asarray(project_world_to_pixels(
        world, pose_xy=tuple(pose.pose_xy), altitude_m=float(altitude_m), yaw_deg=float(pose.yaw_deg),
        sensor=sensor, airframe_roll_deg=float(pose.roll_deg), airframe_pitch_deg=-float(pose.pitch_deg)),
        dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(uv)):
        return None
    o = uv[0]
    return o, np.column_stack([uv[1] - o, uv[2] - o])


def ground_correction(pose, sensor, *, altitude_m: float, ground_z: float, d_px: tuple[float, float]
                      ) -> tuple[float, float]:
    """The east and north metres to ADD to the pose so the render lands on the frame.

    A feature at world W is at render pixel ``q = J (W - N) + c`` under the
    nominal pose N and at frame pixel ``q + d`` under the true pose T, so
    ``T - N = -J^-1 d``.
    """
    jac = _jacobian(pose, sensor, altitude_m=altitude_m, ground_z=ground_z, at_xy=tuple(pose.pose_xy))
    if jac is None:
        return float("nan"), float("nan")
    _o, j = jac
    try:
        de, dn = -np.linalg.solve(j, np.asarray(d_px, dtype=np.float64))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(de), float(dn)
