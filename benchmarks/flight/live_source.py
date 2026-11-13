"""What the aircraft's recording holds over an area, as one observation the mission's projector reads.

:meth:`LiveSource.coverage` reads off the ring's index how much of an area
the recording holds (the arrival rule); :meth:`LiveSource.observe_area` fuses
the frames that saw it nearest nadir into one north-up orthophoto on the map
grid, each frame registered into the pose graph as it arrives
(:meth:`LiveSource.start_corrector`) and placed through the calibration:
the ``boresight.json`` a calibration solve writes, passed to :func:`open_live`
as ``boresight_path``, else the datasheet camera (the twin's).
"""

from __future__ import annotations

import functools
import json
import logging
import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.bench import missing_bake
from benchmarks.flight import frame_offset, frame_ties, locator, map_render
from benchmarks.flight.airframe import C12_LEVER_M
from benchmarks.flight.footprint import in_frame_xy, off_nadir_lonlat
from benchmarks.flight.nav_track import Pose, camera_pose_local, tag_pose
from benchmarks.flight.pose_graph import INNOVATION_CAP_M, PoseGraph, innovation_past_cap
from benchmarks.flight.tile_georef import FLOOR_MSL_M, lonlat_to_local
from maps.perception.geometry.sensor_spec import SensorSpec, resolve_sensor
from maps.perception.observation import Observation
from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.perception.ortho.sensor_frame import footprint_m, ground_corners_xy

log = logging.getLogger(__name__)

#: The mosaic's grid cell where the frames report none, metres.
GSD_FLOOR_M = 0.3

#: How far either way the mosaic's yaw is swept against the map: a crab in a
#: crosswind runs past what one frame's marks can carry.
MOSAIC_YAW_REACH_DEG = 8.0

#: How often the corrector asks for the newest frame: the camera's own rate.
CORRECTOR_PERIOD_S = 0.1

#: A whole-frame correlation under this is refused: the frame met structure the
#: map does not hold there, and its peak is a false one more often than not.
FRAME_RESPONSE_MIN = 0.4

#: How far, in degrees, a frame's pose may sit from an area's centroid and
#: still be scored against it: about four kilometres, several times a frame's
#: reach at the C-12's height.
HOLD_NEAR_DEG = 0.04


def ortho_observation(
    rgb: np.ndarray,
    alpha: np.ndarray,
    extent: tuple[float, float, float, float],
    camera_z: float,
    *,
    tile_ground_z: float,
) -> Observation:
    """A north-up image on the map grid as the observation the bench reads
    orthographically: ``extent`` is ``(x0, y0, x1, y1)`` in bake-local metres,
    ``camera_z`` the camera's height in the tile frame."""
    x0, y0, x1, y1 = (float(v) for v in extent)
    depth = np.where(alpha > 0, np.float32(camera_z - tile_ground_z), np.float32(0.0)).astype(
        np.float32
    )
    return Observation(
        pose_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        altitude_m=float(camera_z),
        world_window=(x0, y0, x1, y1),
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        # [FL, FR, BR, BL] == image [TL, TR, BR, BL]; north-up, so FL is the
        # north-west corner, the pixel the grid starts at.
        ortho_view_corners_xy=((x0, y1), (x1, y1), (x1, y0), (x0, y0)),
    )


def coverage_record(best: list[tuple[float, int] | None], frames: int) -> dict[str, Any]:
    """How much of an area a source holds, as ``coverage`` returns it: per
    point ``best`` (``(off-nadir degrees, seq)``, or None where nothing holds
    it), the fraction held, and how many ``frames`` were asked."""
    held = sum(1 for b in best if b is not None)
    return {
        "points": len(best),
        "held": held,
        "fraction": (held / len(best)) if best else 0.0,
        "best": best,
        "frames": frames,
    }


@functools.lru_cache(maxsize=4)
def _radius(h: int, w: int) -> np.ndarray:
    """Each source pixel's distance from the frame's centre: the seam rule's cost."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0).astype(np.float32)


def composite(
    corners: np.ndarray, images, *, gsd_m: float
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], int]:
    """``(rgb, alpha, (x0, y0, x1, y1), placed)``: a north-up mosaic of posed
    frames on one ground grid. ``corners`` is ``(n, 4, 2)`` floor corners in
    bake-local metres for the n RGB ``images``; an overlapped ground pixel
    takes the frame whose centre it is nearest, so seams fall mid-overlap."""
    import cv2

    allc = np.concatenate(corners)
    x0, y0 = np.floor(allc.min(axis=0) / gsd_m) * gsd_m
    x1, y1 = np.ceil(allc.max(axis=0) / gsd_m) * gsd_m
    width, height = int(round((x1 - x0) / gsd_m)), int(round((y1 - y0) / gsd_m))
    rgb = np.zeros((height, width, 3), np.uint8)
    nearest = np.full((height, width), np.inf, np.float32)
    placed = 0
    for img, frame_corners in zip(images, corners):
        h, w = img.shape[:2]
        xy = np.asarray(frame_corners, dtype=np.float64)
        dst = np.column_stack([(xy[:, 0] - x0) / gsd_m, (y1 - xy[:, 1]) / gsd_m]).astype(np.float32)
        c0 = np.maximum(np.floor(dst.min(axis=0)).astype(int), 0)
        c1 = np.minimum(np.ceil(dst.max(axis=0)).astype(int) + 1, [width, height])
        if np.any(c1 <= c0):
            continue
        size = (int(c1[0] - c0[0]), int(c1[1] - c0[1]))
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        hom = cv2.getPerspectiveTransform(src, dst - c0.astype(np.float32))
        warped = cv2.warpPerspective(
            img, hom, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        dist = cv2.warpPerspective(
            _radius(h, w),
            hom,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(np.finfo(np.float32).max),
        )
        # The pixels the frame's quad lands on: only these are imaged.
        on = (
            cv2.warpPerspective(
                np.ones((h, w), np.uint8),
                hom,
                size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        win = (slice(c0[1], c1[1]), slice(c0[0], c1[0]))
        take = on & (dist < nearest[win])
        nearest[win][take] = dist[take]
        rgb[win][take] = warped[take]
        placed += 1
    alpha = np.where(np.isfinite(nearest), 255, 0).astype(np.uint8)
    return rgb, alpha, (float(x0), float(y0), float(x1), float(y1)), placed


def corrected_ortho(
    rgb: np.ndarray,
    alpha: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    shift_e_m: float,
    shift_n_m: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """The product turned about its window's centre by ``yaw_deg`` (heading
    sense) and moved by the shift, resampled north-up on the grid it was on:
    ``(rgb, alpha, extent)``, the extent grown to hold the turned corners."""
    import cv2

    x0, y0, x1, y1 = (float(v) for v in extent)
    h, w = rgb.shape[:2]
    g = (x1 - x0) / w
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    a = math.radians(yaw_deg)
    cos, sin = math.cos(a), math.sin(a)

    def truth(px: np.ndarray) -> np.ndarray:
        # The ground a pixel truly shows: its grid point turned about the centre, then moved.
        e, n = x0 + px[:, 0] * g - cx, y1 - px[:, 1] * g - cy
        return np.column_stack(
            [cx + e * cos + n * sin + shift_e_m, cy - e * sin + n * cos + shift_n_m]
        )

    box = truth(np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64))
    nx0, ny0 = np.floor(box.min(axis=0) / g) * g
    nx1, ny1 = np.ceil(box.max(axis=0) / g) * g
    nw, nh = int(round((nx1 - nx0) / g)), int(round((ny1 - ny0) / g))
    # Three new-grid pixels and the old pixels they read: the affine between them is the warp.
    new_px = np.array([[0, 0], [nw, 0], [0, nh]], dtype=np.float64)
    ground = np.column_stack(
        [nx0 + new_px[:, 0] * g - cx - shift_e_m, ny1 - new_px[:, 1] * g - cy - shift_n_m]
    )
    e, n = ground[:, 0] * cos - ground[:, 1] * sin, ground[:, 0] * sin + ground[:, 1] * cos
    old_px = np.column_stack([(cx + e - x0) / g, (y1 - (cy + n)) / g])
    m = cv2.getAffineTransform(new_px.astype(np.float32), old_px.astype(np.float32))
    out = cv2.warpAffine(
        np.ascontiguousarray(rgb), m, (nw, nh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    out_alpha = cv2.warpAffine(
        np.ascontiguousarray(alpha), m, (nw, nh), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    return out, out_alpha, (float(nx0), float(ny0), float(nx1), float(ny1))


def register_mosaic(
    rgb: np.ndarray,
    alpha: np.ndarray,
    extent: tuple[float, float, float, float],
    render: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], dict[str, Any]]:
    """``(rgb, alpha, extent, record)``: the mosaic registered on the map as
    ONE similarity and corrected by it when the fit passes the gates, else as
    it was with ``why``. A crab or a mount's yaw shows here."""
    t0 = time.perf_counter()
    x0, y0, x1, y1 = (float(v) for v in extent)
    h, w = rgb.shape[:2]
    g = (x1 - x0) / w
    du, dv, yaw, response = frame_offset.similarity(rgb, render, yaw_reach_deg=MOSAIC_YAW_REACH_DEG)
    record: dict[str, Any] = {
        "applied": False,
        "response": round(float(response), 4),
        "shift_yaw_deg": round(float(yaw), 2),
    }
    reach = frame_offset.reach_px(w, h)
    if response < FRAME_RESPONSE_MIN:
        record["why"] = (
            f"the mosaic does not register: response {response:.2f} under {FRAME_RESPONSE_MIN}"
        )
    elif abs(du) >= reach[0] - frame_offset.DOWN or abs(dv) >= reach[1] - frame_offset.DOWN:
        record["why"] = "the mosaic's peak sits at the search's wall"
    else:
        # The content sits (du, dv) pixels from where the map has it: the
        # ground it truly shows is that far the other way, north up.
        record.update(applied=True, shift_e_m=round(-du * g, 2), shift_n_m=round(dv * g, 2))
        rgb, alpha, extent = corrected_ortho(
            rgb, alpha, extent, shift_e_m=-du * g, shift_n_m=dv * g, yaw_deg=yaw
        )
    record["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    return rgb, alpha, extent, record


@dataclass(frozen=True)
class Boresight:
    """What the calibration solve wrote, as the flight applies it."""

    sensor: SensorSpec
    gnss_bias_e_m: float = 0.0
    gnss_bias_n_m: float = 0.0
    time_offset_s: float = 0.0
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> Boresight:
        doc = json.loads(Path(path).read_text())
        sensor = resolve_sensor(doc["sensor"])
        if sensor is None:
            raise ValueError(f"{path} carries a sensor override nothing resolves")
        if "nav_chain" not in doc:
            raise ValueError(f"{path} carries no nav_chain: not a boresight solve")
        chain = doc["nav_chain"]
        return cls(
            sensor=sensor,
            gnss_bias_e_m=float(chain.get("gnss_bias_e_m", 0.0)),
            gnss_bias_n_m=float(chain.get("gnss_bias_n_m", 0.0)),
            time_offset_s=float(chain.get("time_offset_s", 0.0)),
            path=Path(path),
        )


@dataclass(frozen=True)
class Registration:
    """The map each frame is registered against: ``ortho`` the raster
    (:func:`map_render.load_raster`) and ``render`` the expected view off it."""

    ortho: tuple
    render: Callable


def fused_registration(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One fused observation's frame registrations as one record: ``applied``,
    ``sources`` counted, and when none applied the commonest ``why``."""
    out: dict[str, Any] = {
        "applied": any(r.get("applied") for r in records),
        "frames": len(records),
        "sources": dict(Counter(r.get("source") or "refused" for r in records)),
    }
    if not out["applied"]:
        whys = Counter(str(r["why"]) for r in records if r.get("why")).most_common(1)
        out["why"] = whys[0][0] if whys else "no frame to register"
    return out


def register_frame(
    register: Registration, rgb, pose: frame_ties.Pose, sensor, ground: float
) -> tuple[dict[str, Any], tuple[float, float, float] | None]:
    """``(record, measurement)``: the frame's ground residual on its own
    pixels, its edges located on the map's render at ``pose`` and the shift
    taken to the ground through the projector's Jacobian; the record carries
    the shift or ``why``, the measurement is ``(east, north, response)`` or
    None. ``ground`` is the height under the nadir in ``pose``'s datum."""
    record: dict[str, Any] = {"applied": False, "source": None}
    measured: tuple[float, float, float] | None = None
    try:
        intr = sensor.intrinsics
        expected = register.render(
            locator.sized(
                sensor,
                intr.image_width // frame_offset.DOWN,
                intr.image_height // frame_offset.DOWN,
            ),
            pose.pose_xy,
            pose.alt_msl_m - ground,
            pose.yaw_deg,
            pose.roll_deg,
            -pose.pitch_deg,
        )
    except ValueError:
        expected = None
    if expected is None:
        record["why"] = "the frame reaches off the map"
    else:
        du, dv, response = frame_offset.offset(rgb, expected)
        de, dn = frame_offset.ground_correction(
            pose, sensor, altitude_m=pose.alt_msl_m, ground_z=ground, d_px=(du, dv)
        )
        if not (math.isfinite(de) and math.isfinite(dn)):
            record["why"] = "the frame's offset put the camera nowhere"
        elif response < FRAME_RESPONSE_MIN:
            record["why"] = (
                f"the frame does not register: response {response:.2f} under {FRAME_RESPONSE_MIN}"
            )
        else:
            record.update(applied=True, source="measured", shift_e_m=de, shift_n_m=dn)
            measured = (de, dn, response)
    return record, measured


class LiveSource:
    """The vehicle's recorded frames, posed and registered, as coverage and
    observations. ``floor_msl_m`` is the ground in the receiver's orthometric
    datum, ``tile_ground_z`` the tile frame's floor (``GROUND_LEVEL``).
    ponytail: two heights above ground: the ownship, the frame tests and the
    arrival rule (``ownship``, ``in_frame``, ``frame_transit_s``,
    ``coverage``) stand on the bake's floor, the projector (``_posed``) and
    the mission's track on the ground under the camera (``ground_msl``); one
    camera-now height on ``ground_msl(xy)`` makes them one rule."""

    def __init__(
        self,
        *,
        tiles_dir: str | Path,
        map_name: str,
        boresight: Boresight,
        vehicle,
        tile_ground_z: float,
        floor_msl_m: float,
        register: Registration | None = None,
    ) -> None:
        self.tiles_dir = Path(tiles_dir)
        self.map_name = map_name
        self.boresight = boresight
        self.register = register
        self.vehicle = vehicle
        self.tile_ground_z = float(tile_ground_z)
        self.floor_msl_m = float(floor_msl_m)
        #: The map's bare-earth surface, None where it carries none (:func:`map_render.terrain`).
        self.terrain = map_render.terrain(map_name, str(self.tiles_dir))
        self.last_refusal = ""
        #: Provenance of the last observation: the recorded frames it stands on.
        self.last: dict[str, Any] = {}
        #: The frames placed together, built on the first frame at its size.
        self.graph: PoseGraph | None = None
        self._correcting = threading.Lock()
        #: One ask of the vehicle at a time: the corrector's thread asks too,
        #: and ``vehicle.last_error`` is the reason for the last ask either made.
        self._asking = threading.Lock()
        #: Set to stop the wall-clock corrector; None until it starts.
        self._halt: threading.Event | None = None
        #: The log time past the newest frame a simulated vehicle's clock has corrected.
        self._clocked_since: int | None = None

    def start_corrector(self) -> None:
        """Register frames as the vehicle records them: a simulated vehicle
        (``on_sleep``) corrects every frame up to its clock before a sleep
        returns; a real one runs a wall-clock thread that each
        ``CORRECTOR_PERIOD_S`` takes the newest frame, so when it falls
        behind it skips to the newest."""
        hooks = getattr(self.vehicle, "on_sleep", None)
        if hooks is not None:
            if self._correct_through_now not in hooks:
                hooks.append(self._correct_through_now)
        elif self._halt is None:
            halt = self._halt = threading.Event()

            def run() -> None:
                since = None
                while not halt.is_set():
                    t0 = time.perf_counter()
                    since = self._correct_since(since, newest_only=True)
                    # What is left of the period after the frame, so the rate is the period's.
                    halt.wait(max(0.0, CORRECTOR_PERIOD_S - (time.perf_counter() - t0)))

            threading.Thread(target=run, name="corrector", daemon=True).start()

    def _ask(self, call: Callable, *args) -> tuple[Any, str]:
        """``(answer, why)``: one ask of the vehicle and its ``last_error`` read together."""
        with self._asking:
            return call(*args), self.vehicle.last_error

    def _correct_through_now(self) -> None:
        self._clocked_since = self._correct_since(self._clocked_since)

    def _correct_since(self, since: int | None, *, newest_only: bool = False) -> int | None:
        """:meth:`correct` each frame the index holds past log time ``since``,
        in log-time order (only the newest with ``newest_only``); the log time
        past the last frame asked (fetched, with ``newest_only``), else ``since``."""
        rows = sorted(
            self._ask(self.vehicle.index, since)[0] or [], key=lambda r: int(r["log_time"])
        )
        for row in rows[-1:] if newest_only else rows:
            got, _why = self._ask(self.vehicle.frame, int(row["seq"]))
            # The wall clock asks the newest frame again until it is served;
            # the simulated clock asks each frame once.
            if got is not None or not newest_only:
                since = int(row["log_time"]) + 1
            if got is not None:
                self.correct(got)
        return since

    def close(self) -> None:
        if self._halt is not None:
            self._halt.set()

    def _posed(
        self, jpeg: bytes, t_host: float, *, pose: Pose | None = None
    ) -> SimpleNamespace | str:
        """The frame decoded and its camera posed through the calibrated chain
        at ``pose`` (default :meth:`pose_now` at ``t_host``), or why it cannot
        be: a string, so the corrector's thread leaves ``last_refusal`` alone.
        ``cam`` is the camera pose the registration reads, ``ground_tile`` the
        ground under it in the tile frame."""
        why = ""
        if pose is None:
            pose, why = self._pose(t_host)
        if pose is None:
            return f"no pose for the frame: {why or 'no fix'}"
        pose_xy, alt_msl, yaw = self.camera_pose(pose)
        cam = frame_ties.Pose(
            tuple(pose_xy), float(alt_msl), float(yaw), float(pose.roll_deg), float(pose.pitch_deg)
        )
        try:
            rgb = np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            return f"the frame does not decode: {exc}"
        h, w = rgb.shape[:2]
        sensor = locator.sized(self.boresight.sensor, w, h)
        camera_z = self.tile_ground_z + (alt_msl - self.floor_msl_m)
        ground_tile = self.tile_ground_z + (self.ground_msl(pose_xy) - self.floor_msl_m)
        agl = camera_z - ground_tile
        try:
            corners = self._corners(sensor, cam, agl, ground_tile)
        except ValueError as exc:
            return f"no ground under the frame: {exc}"
        return SimpleNamespace(
            pose=pose,
            cam=cam,
            rgb=rgb,
            sensor=sensor,
            agl=agl,
            camera_z=camera_z,
            corners=corners,
            ground_tile=ground_tile,
        )

    @staticmethod
    def _corners(
        sensor, cam: frame_ties.Pose, agl: float, ground_tile: float
    ) -> list[tuple[float, float]]:
        """Where the frame of a camera at ``cam``, ``agl`` over ``ground_tile``, meets the ground."""
        # ponytail: flat ground at the nadir's height (the bake's DEM, else its floor); relief
        # within a frame needs a height raster under ground_corners_xy. The projector's pitch is nose down.
        return ground_corners_xy(
            sensor,
            cam.pose_xy,
            agl,
            cam.yaw_deg,
            airframe_roll_deg=cam.roll_deg,
            airframe_pitch_deg=-cam.pitch_deg,
            ground_z=ground_tile,
        )

    def ground_msl(self, xy) -> float:
        """The ground under ``xy`` in the receiver's orthometric datum: the
        map's surface where it carries one, else ``floor_msl_m``."""
        return map_render.ground_msl(self.terrain, self.floor_msl_m, xy)

    def correct(self, got: tuple[bytes, float, int, int, dict]) -> None:
        """Register one recorded frame into the pose graph, posed by the
        vehicle's pose at its stamp, as the corrector does for every one."""
        if self.register is None:
            return
        jpeg, t_host, _log_time, seq, _tag = got
        posed = self._posed(jpeg, t_host)
        if isinstance(posed, str):
            log.debug("corrector: frame %s refused: %s", seq, posed)
            return
        with self._correcting:
            if self.graph is None:
                self.graph = PoseGraph(posed.sensor, self.ground_msl)
            _record, measured = register_frame(
                self.register,
                posed.rgb,
                posed.cam,
                posed.sensor,
                self.ground_msl(posed.cam.pose_xy),
            )
            self.graph.offer(seq, t_host, posed.cam, locator.gray(posed.rgb), measured)

    def _register_ring(self, seq: int, posed, t: float) -> dict[str, Any]:
        """A ring frame registered on its own pixels: a measurement past the
        cap (:func:`pose_graph.innovation_past_cap`) from the graph's node
        nearest in time is refused, and a refused or unmeasured frame takes that node."""
        record, _measured = register_frame(
            self.register, posed.rgb, posed.cam, posed.sensor, self.ground_msl(posed.cam.pose_xy)
        )
        record["seq"] = seq
        with self._correcting:
            est = self.graph.estimate(t) if self.graph is not None else None
        far = (
            innovation_past_cap(est, record["shift_e_m"], record["shift_n_m"])
            if record["applied"]
            else None
        )
        if far is not None:
            record.update(
                applied=False,
                source=None,
                why=f"disagrees with the graph's nearest node by {far:.1f} m, past {INNOVATION_CAP_M:.0f}",
            )
        if not record["applied"] and est is not None:
            record.update(
                applied=True, source="nearest", shift_e_m=est["e_m"], shift_n_m=est["n_m"]
            )
        return record

    def pose_now(self, t: float | None = None) -> Pose | None:
        """The antenna's pose at ``t`` (now by default), the timing offset applied."""
        return self._pose(t)[0]

    def _pose(self, t: float | None = None) -> tuple[Pose | None, str]:
        """:meth:`pose_now` and why it is None."""
        return self._ask(
            self.vehicle.pose,
            (self.vehicle.now() if t is None else t) + self.boresight.time_offset_s,
        )

    def camera_pose(self, pose: Pose) -> tuple[tuple[float, float], float, float]:
        """``(pose_xy, alt_msl, yaw)`` of the camera in the tile frame, the solved GNSS bias applied."""
        pose_xy, alt_msl, yaw = camera_pose_local(
            pose, self.tiles_dir, self.map_name, lever=C12_LEVER_M
        )
        return (
            (pose_xy[0] + self.boresight.gnss_bias_e_m, pose_xy[1] + self.boresight.gnss_bias_n_m),
            alt_msl,
            yaw,
        )

    def in_frame(self, x: float, y: float) -> bool | None:
        """Whether ground ``(x, y)`` is in the frame at the pose now; None
        without a pose. ponytail: field of view only, no mount angles (tenths
        of a degree against a 15 degree frame)."""
        pose = self.pose_now()
        if pose is None:
            return None
        pose_xy, alt_msl, yaw = self.camera_pose(pose)
        intr = self.boresight.sensor.intrinsics
        return in_frame_xy(
            pose_xy,
            alt_msl - self.floor_msl_m,
            yaw,
            pose.roll_deg,
            pose.pitch_deg,
            (x, y),
            hfov_deg=intr.hfov_deg,
            vfov_deg=intr.vfov_deg,
        )

    def frame_transit_s(self) -> float | None:
        """How long a point stays in the frame as the aircraft flies over it:
        the along-track ground one frame covers over the speed made good.
        None without a pose, and none when the aircraft is not moving."""
        pose = self.pose_now()
        if pose is None or float(pose.speed_mps or 0.0) <= 0.1:
            return None
        _xy, alt_msl, _yaw = self.camera_pose(pose)
        agl = alt_msl - self.floor_msl_m
        if agl <= 0.0:
            return None
        return footprint_m(self.boresight.sensor, agl)[1] / pose.speed_mps

    def ownship(self) -> dict[str, Any] | None:
        """Where the aircraft IS: the camera in the tile frame, its height
        above ground, and the receiver's UTC on this pose; None without a fix."""
        pose = self.pose_now()
        if pose is None:
            return None
        pose_xy, alt_msl, _yaw = self.camera_pose(pose)
        return {
            "x": pose_xy[0],
            "y": pose_xy[1],
            "agl_m": alt_msl - self.floor_msl_m,
            "utc": pose.utc,
        }

    def coverage(
        self, points: list[tuple[float, float]], after: float | None = None
    ) -> dict[str, Any] | None:
        """How much of an area (``points``, lon/lat) the ring's index holds,
        over the frames stamped after ``after`` (vehicle seconds) when given:
        ``fraction``, and ``best`` per point ``(off-nadir degrees, seq)`` or
        None. None with ``last_refusal`` when the ring cannot be asked."""
        rows = self._ring_rows(after)
        return None if rows is None else self._held_by(rows, points)

    def coverage_of_each(
        self, areas: list[list[tuple[float, float]]]
    ) -> list[dict[str, Any] | None] | None:
        """:meth:`coverage` of several areas off ONE read of the ring's index,
        in the areas' order. None when the ring cannot be asked."""
        rows = self._ring_rows()
        return None if rows is None else [self._held_by(rows, area) for area in areas]

    def _ring_rows(self, after: float | None = None) -> list[dict[str, Any]] | None:
        """The ring's index, the frames stamped after ``after`` (vehicle seconds) when given."""
        rows, why = self._ask(self.vehicle.index)
        if rows is None:
            self.last_refusal = f"the recorder's ring cannot be asked: {why or 'no index'}"
        elif after is not None:
            rows = [r for r in rows if float(r["t_host"]) > after]
        return rows

    def _held_by(
        self, rows: list[dict[str, Any]], points: list[tuple[float, float]]
    ) -> dict[str, Any]:
        intr = self.boresight.sensor.intrinsics
        # Only a frame near the area can hold it: rows more than HOLD_NEAR_DEG
        # off are skipped before any geometry.
        if points:
            clon = sum(p[0] for p in points) / len(points)
            clat = sum(p[1] for p in points) / len(points)
            rows = [
                r
                for r in rows
                if abs(float((r.get("pose") or {}).get("lat") or 0.0) - clat) <= HOLD_NEAR_DEG
                and abs(float((r.get("pose") or {}).get("lon") or 0.0) - clon) <= HOLD_NEAR_DEG
            ]
        best: list[tuple[float, int] | None] = [None] * len(points)
        # Newest first, so at one angle the fresher frame wins; the sequence
        # restarts with the camera, the stamp does not.
        for row in sorted(rows, key=lambda r: -int(r["log_time"])):
            tag = row.get("pose") or {}
            if tag.get("alt_msl_m") is None or tag.get("heading_deg") is None:
                continue
            seq = int(row["seq"])
            agl = float(tag["alt_msl_m"]) - self.floor_msl_m
            for i, (lon, lat) in enumerate(points):
                angle = off_nadir_lonlat(
                    tag["lat"],
                    tag["lon"],
                    agl,
                    tag["heading_deg"],
                    tag.get("roll_deg") or 0.0,
                    tag.get("pitch_deg") or 0.0,
                    lat,
                    lon,
                    hfov_deg=intr.hfov_deg,
                    vfov_deg=intr.vfov_deg,
                )
                if angle is None:
                    continue
                if best[i] is None or angle < best[i][0]:
                    best[i] = (float(angle), seq)
        self.last_refusal = ""
        return coverage_record(best, len(rows))

    def observe_area(
        self, points: list[tuple[float, float]], after: float | None = None
    ) -> Observation | None:
        """The ground the pass held over an AREA as one north-up observation:
        the frames (after ``after`` when given) nearest nadir over each point,
        fused; a partial pass is served as the part it held. None with
        ``last_refusal`` when no frame held any of it."""
        rows = self._ring_rows(after)
        if rows is None:
            return None
        cov = self._held_by(rows, points)
        if not cov["held"]:
            self.last_refusal = (
                f"no frame over any of the area's {cov['points']} points in the "
                f"recorder's ring ({cov['frames']} frames asked)"
            )
            return None
        chosen: dict[int, float] = {}
        for b in cov["best"]:
            if b is not None:
                chosen[b[1]] = min(chosen.get(b[1], b[0]), b[0])
        # At one angle the frame stamped later goes first, as it holds a point in _held_by.
        stamped: dict[int, int] = {}
        for r in rows:
            stamped[int(r["seq"])] = max(stamped.get(int(r["seq"]), 0), int(r["log_time"]))
        want = [
            (chosen[seq], seq) for seq in sorted(chosen, key=lambda s: (chosen[s], -stamped[s]))
        ]
        obs = self._fused(want, ranked=cov["frames"], about=f"the area's {cov['points']} points")
        if obs is not None:
            self.last["coverage"] = {"points": cov["points"], "held": cov["held"]}
        return obs

    def _fused(
        self, want: list[tuple[float, int]], *, ranked: int, about: str
    ) -> Observation | None:
        """``want``'s frames, ``(off-nadir degrees, seq)``, fetched, each placed
        by its pose and registration, composited at their median GSD."""
        placed: list[Any] = []
        records: list[dict[str, Any]] = []
        unfetched = 0
        for angle, asked in want:
            got, why = self._ask(self.vehicle.frame, asked)
            if got is None:
                # The link, not the ring: the index said the frame is there.
                unfetched += 1
                self.last_refusal = f"ring frame {asked} not fetched: {why or 'no reason recorded'}"
                continue
            jpeg, t_host, log_time, seq, tag = got
            # Posed as the corrector poses it, so its registration and the
            # graph's nodes share a time base; the tag where the vehicle has no pose then.
            pose = self.pose_now(t_host) or tag_pose(tag)
            if pose is None:
                self.last_refusal = f"ring frame {seq} carries no pose"
                continue
            posed = self._posed(jpeg, t_host, pose=pose)
            if isinstance(posed, str):
                self.last_refusal = posed
                continue
            if self.register is not None:
                record = self._register_ring(seq, posed, t_host)
                # The heading the frame was placed at and the track beside it: a crab is their difference.
                record.update(
                    heading_deg=posed.pose.heading_deg,
                    track_deg=posed.pose.track_deg,
                    attitude=posed.pose.attitude,
                )
                records.append(record)
                if record["applied"]:
                    moved = replace(
                        posed.cam,
                        pose_xy=(
                            posed.cam.pose_xy[0] + record["shift_e_m"],
                            posed.cam.pose_xy[1] + record["shift_n_m"],
                        ),
                    )
                    posed.corners = self._corners(posed.sensor, moved, posed.agl, posed.ground_tile)
            posed.angle, posed.seq, posed.t_host, posed.log_time = angle, seq, t_host, log_time
            placed.append(posed)
        if not placed:
            self.last_refusal = (
                f"none of the {len(want)} frames over {about} could be placed: "
                f"{self.last_refusal or 'no reason recorded'}"
            )
            return None
        # The grid's cell is what the FRAMES carry: resampling onto a finer grid
        # invents resolution the sensor never had, onto a coarser one throws it away.
        gsd = [
            footprint_m(f.sensor, f.agl)[0] / f.sensor.intrinsics.image_width
            for f in placed
            if f.agl > 0.0
        ]
        gsd_m = float(np.median(gsd)) if gsd else GSD_FLOOR_M
        corners = np.asarray([[(float(c[0]), float(c[1])) for c in f.corners] for f in placed])
        rgb, alpha, (x0, y0, x1, y1), n = composite(corners, (f.rgb for f in placed), gsd_m=gsd_m)
        fit = None
        if self.register is not None:
            # The frames placed together, registered on the map as one: the yaw
            # they share is read off the whole mosaic.
            render = map_render.window_render((x0, y0, x1, y1), rgb.shape[:2], self.register.ortho)
            if render is None:
                fit = {"applied": False, "why": "the mosaic reaches off the map"}
            else:
                rgb, alpha, (x0, y0, x1, y1), fit = register_mosaic(
                    rgb, alpha, (x0, y0, x1, y1), render
                )
            if fit["applied"]:
                log.info(
                    "mosaic registered on the map: %+.1f E %+.1f N %+.2f deg, response %.2f, %.0f ms",
                    fit["shift_e_m"],
                    fit["shift_n_m"],
                    fit["shift_yaw_deg"],
                    fit["response"],
                    fit["ms"],
                )
            else:
                log.info("mosaic not registered on the map: %s", fit["why"])
        obs = ortho_observation(
            rgb,
            alpha,
            (x0, y0, x1, y1),
            float(np.median([f.camera_z for f in placed])),
            tile_ground_z=self.tile_ground_z,
        )
        # The sequence restarts with the camera; the stamp does not.
        newest = max(placed, key=lambda f: f.log_time)
        self.last = {
            "log_time": newest.log_time,
            "seq": newest.seq,
            "t_host": newest.t_host,
            "age_s": self.vehicle.now() - newest.t_host,
            "attitude": newest.pose.attitude,
            "utc": newest.pose.utc,
            "boresight": self.boresight.path.name if self.boresight.path else "datasheet",
            "registration": fused_registration(records) if self.register is not None else None,
            "corners_xy": list(obs.ortho_view_corners_xy),
            "mosaic": {
                "placed": n,
                "asked": len(want),
                "ranked": ranked,
                "gsd_m": round(gsd_m, 3),
                "fetch_failed": unfetched,
                "fit": fit,
                "seqs": [f.seq for f in placed],
                "off_nadir_deg": [round(f.angle, 2) for f in placed],
                "registrations": [
                    {
                        k: r.get(k)
                        for k in (
                            "seq",
                            "applied",
                            "source",
                            "why",
                            "shift_e_m",
                            "shift_n_m",
                            "heading_deg",
                            "track_deg",
                            "attitude",
                        )
                    }
                    for r in records
                ],
            },
            "off_nadir_deg": round(min(f.angle for f in placed), 2),
        }
        self.last_refusal = ""
        return obs


def check_map(tiles_dir, map_name: str) -> None:
    """Refuse a map a fix cannot be placed on (no georef under ``tiles_dir``),
    an episode cannot load on (a bake file missing) or a floor cannot be read
    for, before any run folder is opened."""
    if lonlat_to_local(Path(tiles_dir), map_name, 0.0, 0.0) is None:
        raise ValueError(
            f"{map_name} carries no georef under {tiles_dir}: a fix cannot be placed in the tile frame"
        )
    if why := missing_bake(str(Path(tiles_dir) / map_name)):
        raise ValueError(why)
    if map_name not in GROUND_LEVEL:
        raise ValueError(f"{map_name} has no floor in GROUND_LEVEL: no bench declares it")
    if map_name not in FLOOR_MSL_M:
        raise ValueError(f"{map_name} has no orthometric floor in tile_georef.FLOOR_MSL_M")


def open_live(
    *,
    vehicle,
    tiles_dir,
    map_name: str,
    sensor: SensorSpec,
    boresight_path: str | Path | None = None,
    register_raster: str | Path | None = None,
) -> LiveSource:
    """A live source over ``vehicle``, failing on the ramp rather than on the
    first flight. ``boresight_path`` is a calibration solve's
    ``boresight.json``; with ``register_raster`` (the raster each frame is registered
    against) the corrector starts."""
    check_map(tiles_dir, map_name)
    boresight = Boresight.load(boresight_path) if boresight_path else Boresight(sensor=sensor)
    # The twin (a simulated vehicle, ``on_sleep``) renders through the datasheet camera itself.
    if boresight_path is None and getattr(vehicle, "on_sleep", None) is None:
        log.warning("live source: no boresight.json; flying the datasheet camera")
    register = None
    if register_raster is not None:
        ortho = map_render.load_raster(register_raster, map_name)
        if ortho is None:
            raise ValueError(
                f"no {map_name} raster under {register_raster} to register the frames against: "
                f"a raster directory holds {map_render.raster_pair(map_name)}"
            )
        register = Registration(ortho=ortho, render=map_render.renderer(ortho))
    source = LiveSource(
        tiles_dir=tiles_dir,
        map_name=map_name,
        boresight=boresight,
        vehicle=vehicle,
        tile_ground_z=float(GROUND_LEVEL[map_name]),
        floor_msl_m=FLOOR_MSL_M[map_name],
        register=register,
    )
    if register is not None:
        source.start_corrector()
    return source
