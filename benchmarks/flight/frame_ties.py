"""Tie points between two overlapping frames: the same ground seen twice, located by the edge matcher.

The partner frame is warped onto this frame's work grid through the floor, as
the poses in hand place the two, and templates of the warped partner's edge
structure are located in this frame's (:func:`locator.locate_edges`). What a
peak finds is one ground point in both frames' pixels, and the adjustment
(:mod:`tie_adjust`) reads the two frames' relative placement off it.

A pinhole over flat ground is a homography exactly, so :func:`floor_homography`
carries any pixel to the floor with no iteration, and a yaw's lever is a
central difference of the same map. The frame is binned to
:data:`locator.WORK_GSD_M` first, so the windows and the gate mean the ground
they were fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from benchmarks.flight import locator
from maps.perception.ortho.sensor_frame import frame_corners_px, ground_corners_xy

#: Candidate templates are laid this far apart on the work grid.
TIE_PITCH_PX = 160
#: The warped partner's footprint is eroded by the template's half plus the
#: bilinear taps' reach, so no template straddles the warp's black border.
ERODE_PX = locator.EDGE_TEMPLATE_HALF + 2
#: The yaw step of the lever's central difference, degrees.
YAW_STEP_DEG = 0.1
#: A pair keeps its ties only where at least two agree on the displacement to
#: this many work pixels: a lone lock on a grid town's repeating structure is a
#: tie to the wrong block.
TIE_AGREE_PX = 3.0
TIE_AGREE_MIN = 2
#: Templates kept a pair: past twelve the tie rows out-vote the anchors and
#: the cap refuses its own measurements, so eight sits a step inside that.
TIES_PER_PAIR = 8
#: Frames whose work grid and edge maps are kept: the pose graph's
#: ``TIE_NEIGHBOURS`` plus two.
CACHED_FRAMES = 3


@dataclass(frozen=True)
class Pose:
    """A camera pose as the floor geometry reads it, attitude in aerospace signs."""

    pose_xy: tuple[float, float]
    alt_msl_m: float
    yaw_deg: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class Tie:
    """One ground point seen in two frames: its pixel in each, in the frames' own pixels."""

    uv_i: tuple[float, float]
    uv_j: tuple[float, float]


def floor_homography(pose: Pose, sensor, floor: float) -> np.ndarray:
    """The 3x3 map from the frame's pixels (array indices) to bake-local metres on the floor at ``pose``."""
    import cv2  # noqa: PLC0415

    intr = sensor.intrinsics
    corners = ground_corners_xy(sensor, tuple(pose.pose_xy), pose.alt_msl_m - floor, pose.yaw_deg,
                                airframe_roll_deg=pose.roll_deg, airframe_pitch_deg=-pose.pitch_deg, ground_z=floor)
    src = np.asarray(frame_corners_px(int(intr.image_width), int(intr.image_height)), dtype=np.float32)
    # cv2 takes float32 points and bake metres run to 13 km, so the map is
    # solved about the footprint's centroid and the centroid put back in float64.
    dst = np.asarray(corners, dtype=np.float64)
    centre = dst.mean(axis=0)
    relative = np.asarray(cv2.getPerspectiveTransform(src, (dst - centre).astype(np.float32)), dtype=np.float64)
    back = np.array([[1.0, 0.0, centre[0]], [0.0, 1.0, centre[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    return back @ relative


def ground_of_pixels(homography: np.ndarray, uv) -> np.ndarray:
    """``(N, 2)`` points through a 3x3 map from ``(N, 2)`` points."""
    import cv2  # noqa: PLC0415

    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def work_grid(sensor, agl_m: float) -> tuple[int, int, np.ndarray]:
    """``(width, height, S)``: the frame's work grid and the 3x3 map from its
    pixel indices onto it, exact for the area binning (``u_b = (u + 1/2) s - 1/2``)."""
    binned, scale = locator.work_sensor(sensor, agl_m)
    shift = (scale - 1.0) / 2.0
    s = np.array([[scale, 0.0, shift], [0.0, scale, shift], [0.0, 0.0, 1.0]], dtype=np.float64)
    return int(binned.intrinsics.image_width), int(binned.intrinsics.image_height), s


def candidates(warp: np.ndarray, size_j: tuple[int, int], size_i: tuple[int, int]) -> list[tuple[float, float]]:
    """Template centres on frame i's work grid: a :data:`TIE_PITCH_PX` grid inside the
    search border and inside the partner's warped footprint eroded by :data:`ERODE_PX`."""
    import cv2  # noqa: PLC0415

    wj, hj = size_j
    wi, hi = size_i
    corners = np.asarray(frame_corners_px(wj, hj), dtype=np.float64).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, warp).reshape(-1, 2).astype(np.float32)
    out = []
    for v in range(locator.SEARCH_HALF + TIE_PITCH_PX // 2, hi - locator.SEARCH_HALF, TIE_PITCH_PX):
        for u in range(locator.SEARCH_HALF + TIE_PITCH_PX // 2, wi - locator.SEARCH_HALF, TIE_PITCH_PX):
            if cv2.pointPolygonTest(poly, (float(u), float(v)), True) >= ERODE_PX:
                out.append((float(u), float(v)))
    return out


def _keep(hits: list[tuple[tuple[float, float], tuple[float, float], float, float]]) -> list:
    """The hits past the edge gate that agree with the pair's median
    displacement to ``TIE_AGREE_PX``, at least ``TIE_AGREE_MIN`` of them or
    none, the strongest margins first, at most ``TIES_PER_PAIR``."""
    kept = [(c, f, p, m) for c, f, p, m in hits if m >= locator.MIN_EDGE_MARGIN]
    if len(kept) < TIE_AGREE_MIN:
        return []
    d = np.asarray([(f[0] - c[0], f[1] - c[1]) for c, f, _p, _m in kept], dtype=np.float64)
    med = np.median(d, axis=0)
    kept = [h for h, dd in zip(kept, d) if np.hypot(*(dd - med)) <= TIE_AGREE_PX]
    if len(kept) < TIE_AGREE_MIN:
        return []
    kept.sort(key=lambda h: -h[3])
    return kept[:TIES_PER_PAIR]


class TieMatcher:
    """Ties between frames of one sensor over the map's ground, each frame's
    work grid and edge maps cached by the frame and its grid.

    ``ground`` is the height under a nadir as a callable of ``(x, y)``, so each
    frame's floor homography sits on its own ground.
    """

    def __init__(self, sensor, ground) -> None:
        self.sensor, self.ground = sensor, ground
        self._cache: dict = {}

    def _cached(self, key, make):
        """``make()``'s result for ``key`` as a Future on the locator's pool: a
        frame warmed ahead (:meth:`warm`) and asked for at its pair share one computation."""
        if key not in self._cache:
            if len(self._cache) >= CACHED_FRAMES:
                del self._cache[next(iter(self._cache))]
            self._cache[key] = locator._pool.submit(make)
        return self._cache[key]

    def warm(self, key, pose: Pose, gray: np.ndarray) -> None:
        """Start this frame's work grid and edge maps now, at ``pose``, ahead of
        the pair its successor opens."""
        self._frame(key, gray, pose.alt_msl_m - self.ground(tuple(pose.pose_xy)))

    def _frame(self, key, gray: np.ndarray, agl_m: float):
        """``(w, h, S, future)``: the frame binned to its work grid and its edge maps."""
        import cv2  # noqa: PLC0415

        w, h, s = work_grid(self.sensor, agl_m)

        def make():
            small = gray if (w, h) == (gray.shape[1], gray.shape[0]) else cv2.resize(
                gray, (w, h), interpolation=cv2.INTER_AREA)
            return small, locator.edge_maps(small)

        return w, h, s, self._cached((key, w, h), make)

    def ties(self, key_i, pose_i: Pose, gray_i: np.ndarray, pose_j: Pose, gray_j: np.ndarray) -> list[Tie]:
        """The ties between frame i (cached under ``key_i``) and frame j, in the frames' own pixels."""
        import cv2  # noqa: PLC0415

        g_i, g_j = self.ground(tuple(pose_i.pose_xy)), self.ground(tuple(pose_j.pose_xy))
        w, h, s_i = work_grid(self.sensor, pose_i.alt_msl_m - g_i)
        h_i = floor_homography(pose_i, self.sensor, g_i)
        h_j = floor_homography(pose_j, self.sensor, g_j)
        warp = s_i @ np.linalg.inv(h_i) @ h_j
        centres = candidates(warp, (gray_j.shape[1], gray_j.shape[0]), (w, h))
        if not centres:
            return []
        _w, _h, _s, own = self._frame(key_i, gray_i, pose_i.alt_msl_m - g_i)
        e = cv2.warpPerspective(gray_j, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        edges_e = locator.edge_maps(e)
        _a, edges_a = own.result()
        found = locator._pool.map(lambda c: locator.locate_edges(edges_a, edges_e, c), centres)
        hits = [(c, hit[0], hit[1], hit[2]) for c, hit in zip(centres, found) if hit is not None]
        inv_s, inv_warp = np.linalg.inv(s_i), np.linalg.inv(warp)
        out = []
        for centre, at, _peak, _margin in _keep(hits):
            uv_i = ground_of_pixels(inv_s, [at])[0]
            uv_j = ground_of_pixels(inv_warp, [centre])[0]
            out.append(Tie((float(uv_i[0]), float(uv_i[1])), (float(uv_j[0]), float(uv_j[1]))))
        return out


class FrameGeometry:
    """One frame's floor geometry, built once: the ground under any pixel through
    the chain pose, the yaw's lever and the scale's arm there, and the solve's
    node (the nadir and the footprint's half-diagonal)."""

    def __init__(self, pose: Pose, sensor, floor: float) -> None:
        intr = sensor.intrinsics
        self.pose = pose
        self.h = floor_homography(pose, sensor, floor)
        self._ahead = floor_homography(replace(pose, yaw_deg=pose.yaw_deg + YAW_STEP_DEG), sensor, floor)
        self._behind = floor_homography(replace(pose, yaw_deg=pose.yaw_deg - YAW_STEP_DEG), sensor, floor)
        corners = ground_of_pixels(self.h, [(0.0, 0.0), (intr.image_width - 1.0, intr.image_height - 1.0)])
        self.nadir_xy = (float(pose.pose_xy[0]), float(pose.pose_xy[1]))
        self.half_diagonal_m = float(np.hypot(*(corners[1] - corners[0])) / 2.0)
        self.centre = (intr.cx - 0.5, intr.cy - 0.5)

    def ground(self, uv) -> tuple[float, float]:
        g = ground_of_pixels(self.h, [uv])[0]
        return float(g[0]), float(g[1])

    def lever(self, uv) -> tuple[float, float]:
        """Metres a degree of yaw moves the ground under ``uv``."""
        j = (ground_of_pixels(self._ahead, [uv])[0] - ground_of_pixels(self._behind, [uv])[0]) / (2.0 * YAW_STEP_DEG)
        return float(j[0]), float(j[1])

    def radial(self, uv) -> tuple[float, float]:
        """Metres a unit of scale moves the ground under ``uv``: its arm from the nadir."""
        g = self.ground(uv)
        return g[0] - self.nadir_xy[0], g[1] - self.nadir_xy[1]
