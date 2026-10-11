"""What the aircraft's recording holds over an area, as one observation the mission's projector reads.

The live counterpart of the ortho crop the bench perceives with. The vehicle
records frames, each tagged with the pose the aircraft had at its stamp, and
serves the ring's index (one pose a frame and no pixels) and any frame's
pixels by sequence (``benchmarks.flight.vehicle``). This module answers two
questions off that:

* :meth:`LiveSource.coverage`: how much of an area the recording holds, off
  the index alone. This is the one arrival rule: a leg has arrived when the
  pass holds the area it was flown for, whichever pass held it.
* :meth:`LiveSource.observe_area`: the ground the pass held over the area,
  as ONE north-up orthophoto on the map grid. The frames that saw each of
  the area's points nearest nadir are fetched, each placed through the
  solved boresight and its own registration (:func:`register_frame` against
  the map's render, gated by the corrector's :class:`PoseGraph`), composited
  (:func:`composite`), registered on the map as one similarity
  (:func:`register_mosaic`), and handed over as an orthographic observation
  on flat ground at the bake's floor (:func:`ortho_observation`).

What the calibration solved is applied here and nowhere else: the sensor
override from ``boresight.json``, the GNSS bias on the position, the timing
offset on the frame's time. A station started with no boresight flies the
datasheet camera.

Every frame the vehicle records is registered as it arrives by a thread the
live source starts (:class:`Corrector`), and the frames are placed together
(:mod:`benchmarks.flight.pose_graph`). A frame off the ring that is older
than the graph's window is registered on its own pixels, gated by the graph's
node nearest it in time (:meth:`LiveSource._register_ring`).

Signs: a pose carries aerospace roll and pitch (right wing down and nose up
positive). The projector's pitch is nose down, so it is carried negated.

Two floors, one datum each: ``floor_msl_m`` is the ground under the aircraft
in the receiver's orthometric datum, so their difference is the height above
ground; ``tile_ground_z`` is the tile frame's own floor (``GROUND_LEVEL``),
and every altitude an observation carries is that floor plus the height
above ground.

ponytail: flat ground at the bake's floor under the composite; relief needs a
height raster under ``ground_corners_xy``.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from PIL import Image

from benchmarks.flight import frame_offset, frame_ties, locator, map_render
from benchmarks.flight.airframe import C12_LEVER_M
from benchmarks.flight.footprint import ground_gsd_m, in_frame_xy, off_nadir_lonlat
from benchmarks.flight.nav_track import Pose, camera_pose_local, tag_pose
from benchmarks.flight.pose_graph import INNOVATION_CAP_M, PoseGraph
from benchmarks.flight.tile_georef import FLOOR_MSL_M, lonlat_to_local
from maps.perception.geometry.sensor_spec import SensorSpec, resolve_sensor
from maps.perception.observation import Observation
from maps.perception.ortho.sensor_frame import ground_corners_xy

log = logging.getLogger(__name__)

#: The quality a waypoint's observation is written at (``observations/<sha>.jpg``).
OBSERVATION_JPEG_QUALITY = 90

#: The mosaic's grid cell where the frames report none, metres.
GSD_FLOOR_M = 0.3

#: How far either way the mosaic's yaw is swept against the map: a crab in a
#: crosswind runs past what one frame's marks can carry.
MOSAIC_YAW_REACH_DEG = 8.0

#: How often the corrector asks for the newest frame: the camera's own rate.
CORRECTOR_PERIOD_S = 0.1

#: A whole-frame correlation under this is refused: the frame met structure the
#: map does not hold there, and its peak is a false one more often than not.
#: Read against a leg's median on a flown cut, a frame under 0.4 sat within
#: 6 m of the leg 16% of the time, 0.4 to 0.5 45%, 0.5 and over 83%; end to
#: end, a 0.4 floor admits forty percent more frames than 0.5 at barely worse
#: marks each, and the graph holds the tail.
FRAME_RESPONSE_MIN = 0.4

#: How far, in degrees, a frame's pose may sit from an area's centroid and
#: still be scored against it: about four kilometres, several times a frame's
#: reach at the C-12's height.
HOLD_NEAR_DEG = 0.04


def ortho_observation(rgb: np.ndarray, alpha: np.ndarray, extent: tuple[float, float, float, float],
                      camera_z: float, *, tile_ground_z: float, tile: object | None) -> Observation:
    """A north-up image on the map grid as the observation the bench reads
    orthographically: ``extent`` is ``(x0, y0, x1, y1)`` in bake-local metres,
    ``camera_z`` the camera's height in the tile frame."""
    x0, y0, x1, y1 = (float(v) for v in extent)
    depth = np.where(alpha > 0, np.float32(camera_z - tile_ground_z), np.float32(0.0)).astype(np.float32)
    return Observation(
        pose_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        altitude_m=float(camera_z),
        yaw_deg=0.0,
        world_window=(x0, y0, x1, y1),
        sensor=None,          # ortho corners outrank a sensor, and there is no one camera here
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        tile=tile,
        # [FL, FR, BR, BL] == image [TL, TR, BR, BL]; north-up, so FL is the
        # north-west corner, the pixel the grid starts at.
        ortho_view_corners_xy=((x0, y1), (x1, y1), (x1, y0), (x0, y0)),
    )


@functools.lru_cache(maxsize=4)
def _radius(h: int, w: int) -> np.ndarray:
    """Each source pixel's distance from the frame's centre: the seam rule's cost."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0).astype(np.float32)


def composite(corners: np.ndarray, images, *, gsd_m: float
              ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], int]:
    """A north-up mosaic of posed frames on one ground grid, by direct georeferencing.

    ``corners`` is ``(n, 4, 2)``, each frame's corner pixel centres on the
    floor in bake-local metres; ``images`` yields the same n frames as RGB
    arrays. Where frames overlap a ground pixel takes the frame whose centre
    it is nearest, in that frame's own pixels, so seams fall mid-overlap.
    Returns ``(rgb, alpha, (x0, y0, x1, y1), placed)``.
    """
    import cv2  # noqa: PLC0415

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
        warped = cv2.warpPerspective(img, hom, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        dist = cv2.warpPerspective(_radius(h, w), hom, size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.finfo(np.float32).max))
        # The pixels the frame's quad lands on: only these are imaged.
        on = cv2.warpPerspective(np.ones((h, w), np.uint8), hom, size, flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        win = (slice(c0[1], c1[1]), slice(c0[0], c1[0]))
        take = on & (dist < nearest[win])
        nearest[win][take] = dist[take]
        rgb[win][take] = warped[take]
        placed += 1
    alpha = np.where(np.isfinite(nearest), 255, 0).astype(np.uint8)
    return rgb, alpha, (float(x0), float(y0), float(x1), float(y1)), placed


def corrected_ortho(rgb: np.ndarray, alpha: np.ndarray, extent: tuple[float, float, float, float], *,
                    shift_e_m: float, shift_n_m: float, yaw_deg: float
                    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """The product turned about its window's centre by ``yaw_deg`` (heading
    sense) and moved by the shift, resampled north-up on the grid it was on:
    ``(rgb, alpha, extent)``, the extent grown to hold the turned corners."""
    import cv2  # noqa: PLC0415

    x0, y0, x1, y1 = (float(v) for v in extent)
    h, w = rgb.shape[:2]
    g = (x1 - x0) / w
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    a = math.radians(yaw_deg)
    cos, sin = math.cos(a), math.sin(a)

    def truth(px: np.ndarray) -> np.ndarray:
        # The ground a pixel truly shows: its grid point turned about the centre, then moved.
        e, n = x0 + px[:, 0] * g - cx, y1 - px[:, 1] * g - cy
        return np.column_stack([cx + e * cos + n * sin + shift_e_m, cy - e * sin + n * cos + shift_n_m])

    box = truth(np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64))
    nx0, ny0 = np.floor(box.min(axis=0) / g) * g
    nx1, ny1 = np.ceil(box.max(axis=0) / g) * g
    nw, nh = int(round((nx1 - nx0) / g)), int(round((ny1 - ny0) / g))
    # Three new-grid pixels and the old pixels they read: the affine between them is the warp.
    new_px = np.array([[0, 0], [nw, 0], [0, nh]], dtype=np.float64)
    ground = np.column_stack([nx0 + new_px[:, 0] * g - cx - shift_e_m, ny1 - new_px[:, 1] * g - cy - shift_n_m])
    e, n = ground[:, 0] * cos - ground[:, 1] * sin, ground[:, 0] * sin + ground[:, 1] * cos
    old_px = np.column_stack([(cx + e - x0) / g, (y1 - (cy + n)) / g])
    m = cv2.getAffineTransform(new_px.astype(np.float32), old_px.astype(np.float32))
    out = cv2.warpAffine(np.ascontiguousarray(rgb), m, (nw, nh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    out_alpha = cv2.warpAffine(np.ascontiguousarray(alpha), m, (nw, nh),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    return out, out_alpha, (float(nx0), float(ny0), float(nx1), float(ny1))


def register_mosaic(rgb: np.ndarray, alpha: np.ndarray, extent: tuple[float, float, float, float],
                    render: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float],
                                                 dict[str, Any]]:
    """The whole mosaic registered on the map at its window as ONE similarity
    (:func:`frame_offset.similarity`), and corrected by it when the fit passes
    the corrector's gates: ``(rgb, alpha, extent, record)``, the product as it
    was and ``why`` otherwise. What the frames share, a crab or a mount's yaw,
    shows here as the streets turned against the prior."""
    t0 = time.perf_counter()
    x0, y0, x1, y1 = (float(v) for v in extent)
    h, w = rgb.shape[:2]
    g = (x1 - x0) / w
    du, dv, yaw, response = frame_offset.similarity(rgb, render, yaw_reach_deg=MOSAIC_YAW_REACH_DEG)
    record: dict[str, Any] = {"applied": False, "response": round(float(response), 4),
                              "shift_yaw_deg": round(float(yaw), 2)}
    reach = frame_offset.reach_px(w, h)
    if response < FRAME_RESPONSE_MIN:
        record["why"] = f"the mosaic does not register: response {response:.2f} under {FRAME_RESPONSE_MIN}"
    elif abs(du) >= reach[0] - frame_offset.DOWN or abs(dv) >= reach[1] - frame_offset.DOWN:
        record["why"] = "the mosaic's peak sits at the search's wall"
    else:
        # The content sits (du, dv) pixels from where the map has it: the
        # ground it truly shows is that far the other way, north up.
        record.update(applied=True, shift_e_m=round(-du * g, 2), shift_n_m=round(dv * g, 2))
        rgb, alpha, extent = corrected_ortho(rgb, alpha, extent, shift_e_m=-du * g, shift_n_m=dv * g,
                                             yaw_deg=yaw)
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
    def load(cls, path: str | Path) -> "Boresight":
        doc = json.loads(Path(path).read_text())
        sensor = resolve_sensor(doc["sensor"])
        if sensor is None:
            raise ValueError(f"{path} carries a sensor override nothing resolves")
        if "nav_chain" not in doc:
            raise ValueError(f"{path} carries no nav_chain: not a boresight solve")
        chain = doc["nav_chain"]
        return cls(sensor=sensor, gnss_bias_e_m=float(chain.get("gnss_bias_e_m", 0.0)),
                   gnss_bias_n_m=float(chain.get("gnss_bias_n_m", 0.0)),
                   time_offset_s=float(chain.get("time_offset_s", 0.0)), path=Path(path))


@dataclass(frozen=True)
class Registration:
    """The map each frame is registered against: ``ortho`` the raster
    (:func:`map_render.load_raster`), ``render`` the expected view off it, and
    the ground under a nadir, the map's ``terrain`` where it carries one, else
    ``floor_msl_m``. Heights are the receiver's orthometric datum."""

    ortho: tuple
    render: Callable
    floor_msl_m: float
    terrain: object | None = None

    def ground_at(self, xy) -> float:
        return map_render.ground_msl(self.terrain, self.floor_msl_m, xy)


def fused_registration(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The frames of one fused observation, each registered on its own pixels,
    as one record: ``applied`` when any was placed, ``sources`` how many by
    each (``measured``, ``tied``, ``nearest``, ``refused``), and when none was,
    ``why`` the reason most of them gave."""
    sources: dict[str, int] = {}
    for r in records:
        key = r.get("source") or "refused"
        sources[key] = sources.get(key, 0) + 1
    out: dict[str, Any] = {"applied": any(r.get("applied") for r in records), "frames": len(records),
                           "sources": sources}
    if not out["applied"]:
        whys: dict[str, int] = {}
        for r in records:
            if r.get("why"):
                whys[str(r["why"])] = whys.get(str(r["why"]), 0) + 1
        out["why"] = max(whys, key=whys.get) if whys else "no frame to register"
    return out


def register_frame(register: Registration, rgb, pose_xy, altitude_m, yaw, roll, pitch_proj, sensor,
                   graph: PoseGraph | None = None, seq: int | None = None, t: float | None = None
                   ) -> tuple[tuple[float, float], dict[str, Any]]:
    """This frame's ground residual: the corrected pose and its record, or the pose as it was and why.

    The whole frame is the one mark: its edge structure located on the map's
    render at this pose (:func:`frame_offset.offset`), the shift taken to the
    ground through the projector's Jacobian. ``altitude_m`` is the camera's
    orthometric altitude and ``pitch_proj`` the projector's pitch. With a
    ``graph`` and the frame's sequence and time, the frame joins the graph
    with its measurement (or none) and takes the graph's placement; without
    one the measurement stands on its own. A banked frame is asked like a
    level one: its peak is noisier, and the graph's vote over the window is
    what reads it.
    """
    pose = frame_ties.Pose(tuple(pose_xy), float(altitude_m), float(yaw), float(roll), -float(pitch_proj))
    record: dict[str, Any] = {"applied": False, "source": None, "shift_yaw_deg": 0.0}
    measured: tuple[float, float, float] | None = None
    try:
        ground = register.ground_at(pose_xy)
        intr = sensor.intrinsics
        expected = register.render(locator.sized(sensor, intr.image_width // frame_offset.DOWN,
                                                 intr.image_height // frame_offset.DOWN),
                                   tuple(pose_xy), altitude_m - ground,
                                   yaw, roll, pitch_proj)
    except ValueError:
        expected = None
    if expected is None:
        record["why"] = "the frame reaches off the map"
    else:
        du, dv, response = frame_offset.offset(rgb, expected)
        record["response"] = round(response, 4)
        de, dn = frame_offset.ground_correction(pose, sensor, altitude_m=altitude_m, ground_z=ground, d_px=(du, dv))
        if not (math.isfinite(de) and math.isfinite(dn)):
            record["why"] = "the frame's offset put the camera nowhere"
        elif response < FRAME_RESPONSE_MIN:
            record["why"] = f"the frame does not register: response {response:.2f} under {FRAME_RESPONSE_MIN}"
        else:
            record.update(shift_e_m=de, shift_n_m=dn)
            measured = (de, dn, response)
    if graph is not None and seq is not None and t is not None:
        placed = graph.offer(seq, t, pose, locator.gray(rgb), measured)
        if placed.get("why") and not record.get("why"):
            record["why"] = placed["why"]
        record.update({k: v for k, v in placed.items() if k != "why"})
        if record["applied"]:
            pose_xy = (pose_xy[0] + record["shift_e_m"], pose_xy[1] + record["shift_n_m"])
    elif measured is not None:
        pose_xy = (pose_xy[0] + measured[0], pose_xy[1] + measured[1])
        record.update(applied=True, source="measured", shift_e_m=measured[0], shift_n_m=measured[1])
    return pose_xy, record


class Corrector(threading.Thread):
    """Registers every frame the vehicle records, as it arrives, into the pose
    graph. When it falls behind it takes the newest frame."""

    def __init__(self, source: "LiveSource") -> None:
        super().__init__(name="corrector", daemon=True)
        self.source = source
        self._halt = threading.Event()
        #: The log time past the newest frame registered: the index is asked for what came since.
        self._since: int | None = None

    def run(self) -> None:
        while not self._halt.is_set():
            t0 = time.perf_counter()
            rows = self.source.vehicle.index(since=self._since)
            if rows:
                newest = max(rows, key=lambda r: int(r["log_time"]))
                got = self.source.vehicle.frame(int(newest["seq"]))
                if got is not None:
                    self._since = int(newest["log_time"]) + 1
                    self.source.correct(got)
            # What is left of the period after the frame, so the rate is the period's.
            self._halt.wait(max(0.0, CORRECTOR_PERIOD_S - (time.perf_counter() - t0)))

    def stop(self) -> None:
        self._halt.set()


class LiveSource:
    """The vehicle's recorded frames, posed and registered, as coverage and observations."""

    def __init__(self, *, tiles_dir: str | Path, map_name: str, boresight: Boresight, vehicle,
                 tile_ground_z: float, floor_msl_m: float, register: Registration | None = None,
                 tile: object | None = None) -> None:
        self.tiles_dir = Path(tiles_dir)
        self.map_name = map_name
        self.boresight = boresight
        self.register = register
        self.vehicle = vehicle
        self.tile = tile
        self.tile_ground_z = float(tile_ground_z)
        self.floor_msl_m = float(floor_msl_m)
        #: The map's bare-earth surface, None where it carries none (:func:`map_render.terrain`).
        self.terrain = map_render.terrain(map_name, str(self.tiles_dir))
        self.last_refusal = ""
        #: The ring's index as :meth:`coverage_of_each` has read it, and the
        #: newest log time, so each later read asks for the frames since.
        self._index_rows: list[dict[str, Any]] = []
        self._index_newest: int | None = None
        #: Provenance of the last observation: the recorded frames it stands on.
        self.last: dict[str, Any] = {}
        #: The frames placed together, built on the first frame at its size.
        self.graph: PoseGraph | None = None
        self._correcting = threading.Lock()
        self.corrector: Corrector | None = None
        #: The log time past the newest frame a simulated vehicle's clock has corrected.
        self._clocked_since: int | None = None

    def start_corrector(self) -> None:
        """Register frames into the pose graph as the vehicle records them. A
        simulated vehicle (one that carries ``on_sleep``) moves its clock only
        when the harness sleeps, so it corrects every frame up to that clock,
        in order, before the sleep returns; a real vehicle runs the
        wall-clock :class:`Corrector`."""
        hooks = getattr(self.vehicle, "on_sleep", None)
        if hooks is not None:
            if self._correct_through_now not in hooks:
                hooks.append(self._correct_through_now)
        elif self.corrector is None:
            self.corrector = Corrector(self)
            self.corrector.start()

    def _correct_through_now(self) -> None:
        rows = self.vehicle.index(since=self._clocked_since) or []
        for row in sorted(rows, key=lambda r: int(r["log_time"])):
            self._clocked_since = int(row["log_time"]) + 1
            got = self.vehicle.frame(int(row["seq"]))
            if got is not None:
                self.correct(got)

    def close(self) -> None:
        if self.corrector is not None:
            self.corrector.stop()

    def _posed(self, jpeg: bytes, t_host: float, *, pose: Pose | None = None):
        """The frame decoded and its camera posed through the calibrated chain,
        or None with ``last_refusal`` set. ``pose`` is the frame's own tag;
        without one, the vehicle's pose at ``t_host``."""
        pose = pose if pose is not None else self.pose_now(t_host)
        if pose is None:
            self.last_refusal = f"no pose for the frame: {self.vehicle.last_error or 'no fix'}"
            return None
        roll = float(pose.roll_deg)
        pitch_proj = -float(pose.pitch_deg)
        pose_xy, alt_msl, yaw = self.camera_pose(pose)
        rgb = np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        sensor = locator.sized(self.boresight.sensor, w, h)
        camera_z = self.tile_ground_z + (alt_msl - self.floor_msl_m)
        ground_tile = self.ground_tile_z(pose_xy)
        agl = camera_z - ground_tile
        try:
            corners = ground_corners_xy(sensor, pose_xy, agl, yaw, airframe_roll_deg=roll,
                                        airframe_pitch_deg=pitch_proj, ground_z=ground_tile)
        except ValueError as exc:
            self.last_refusal = f"no ground under the frame: {exc}"
            return None
        return SimpleNamespace(pose=pose, pose_xy=pose_xy, alt_msl=alt_msl, yaw=yaw, roll=roll,
                               pitch_proj=pitch_proj, rgb=rgb, sensor=sensor, agl=agl, camera_z=camera_z,
                               corners=corners, ground_tile=ground_tile)

    def ground_tile_z(self, pose_xy) -> float:
        """The ground under ``pose_xy`` in the tile frame: the map's surface
        over its floor where the map carries one, else the tile's floor."""
        return self.tile_ground_z + (map_render.ground_msl(self.terrain, self.floor_msl_m, pose_xy) - self.floor_msl_m)

    def correct(self, got: tuple[bytes, float, int, int, dict]) -> dict[str, Any] | None:
        """Register one recorded frame into the pose graph, posed by the
        vehicle's pose at its stamp, as the corrector does for every one."""
        if self.register is None:
            return None
        jpeg, t_host, _log_time, seq, _tag = got
        posed = self._posed(jpeg, t_host)
        if posed is None:
            return None
        with self._correcting:
            if self.graph is None:
                self.graph = PoseGraph(posed.sensor, self.register.ground_at)
            _xy, record = register_frame(self.register, posed.rgb, posed.pose_xy, posed.alt_msl, posed.yaw,
                                         posed.roll, posed.pitch_proj, posed.sensor,
                                         graph=self.graph, seq=seq, t=t_host)
        return record

    def _register_ring(self, seq: int, posed, t: float) -> dict[str, Any]:
        """A frame off the ring, registered on its own pixels.

        The frame a mission is handed at a waypoint is seconds to tens of
        seconds old, so the graph's window has moved on and its own pixels are
        the one measurement that belongs to it. The graph's node nearest it in
        time is the gate: a ring measurement further from it than
        :data:`pose_graph.INNOVATION_CAP_M` is a false lock and is refused. A
        refused or unmeasured ring frame takes that nearest node (``nearest``);
        without one the pose stands and the record says why.
        """
        _xy, record = register_frame(self.register, posed.rgb, posed.pose_xy, posed.alt_msl, posed.yaw,
                                     posed.roll, posed.pitch_proj, posed.sensor)
        record.update(seq=seq, ring=True)
        with self._correcting:
            est = self.graph.estimate(t) if self.graph is not None else None
        if record["applied"] and est is not None:
            far = math.hypot(record["shift_e_m"] - est["e_m"], record["shift_n_m"] - est["n_m"])
            if far > INNOVATION_CAP_M:
                record.update(applied=False, source=None, measured_e_m=record["shift_e_m"],
                              measured_n_m=record["shift_n_m"],
                              why=f"disagrees with the graph's nearest node by {far:.1f} m, past {INNOVATION_CAP_M:.0f}")
        if not record["applied"] and est is not None:
            # The node's shift and no yaw: the record's own yaw is the refused fit's.
            record.update(applied=True, source="nearest", shift_e_m=est["e_m"], shift_n_m=est["n_m"],
                          shift_yaw_deg=0.0, nearest_s=round(abs(t - est["t"]), 2))
        return record

    def pose_now(self, t: float | None = None) -> Pose | None:
        """The antenna's pose at ``t`` (now by default), the timing offset applied."""
        at = (self.vehicle.now() if t is None else t) + self.boresight.time_offset_s
        return self.vehicle.pose(at)

    def camera_pose(self, pose: Pose) -> tuple[tuple[float, float], float, float]:
        """``(pose_xy, alt_msl, yaw)`` of the camera in the tile frame, the solved GNSS bias applied."""
        pose_xy, alt_msl, yaw = camera_pose_local(pose, self.tiles_dir, self.map_name, lever=C12_LEVER_M)
        return (pose_xy[0] + self.boresight.gnss_bias_e_m, pose_xy[1] + self.boresight.gnss_bias_n_m), alt_msl, yaw

    def in_frame(self, x: float, y: float) -> bool | None:
        """Whether the ground point ``(x, y)`` is inside the camera's frame at
        the pose now, read with the solved sensor's field of view; None
        without a pose. ponytail: the solve's mount angles are not in it,
        tenths of a degree against a frame 15 degrees tall."""
        pose = self.pose_now()
        if pose is None:
            return None
        pose_xy, alt_msl, yaw = self.camera_pose(pose)
        intr = self.boresight.sensor.intrinsics
        return in_frame_xy(pose_xy, alt_msl - self.floor_msl_m, yaw, pose.roll_deg, pose.pitch_deg, (x, y),
                           hfov_deg=intr.hfov_deg, vfov_deg=intr.vfov_deg)

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
        return 2.0 * agl * math.tan(math.radians(self.boresight.sensor.intrinsics.vfov_deg) / 2.0) / pose.speed_mps

    def ownship(self) -> dict[str, Any] | None:
        """Where the aircraft IS: the camera in the tile frame, its height
        above ground, and the receiver's UTC on this pose; None without a fix."""
        pose = self.pose_now()
        if pose is None:
            return None
        pose_xy, alt_msl, _yaw = self.camera_pose(pose)
        return {"x": pose_xy[0], "y": pose_xy[1], "agl_m": alt_msl - self.floor_msl_m, "utc": pose.utc}

    def coverage(self, points: list[tuple[float, float]]) -> dict[str, Any] | None:
        """How much of an area the recording holds. ``points`` are the area's
        (lon, lat) samples; ``fraction`` is the share some frame held, and
        ``best`` per point the ``(off-nadir degrees, sequence)`` of the frame
        that saw it nearest nadir, None for a point no frame held. Read off the
        ring's index over every frame, through the solved camera's field of
        view. None with ``last_refusal`` when the
        ring cannot be asked."""
        rows = self._ring_rows(None)
        if rows is None:
            return None
        return self._held_by(rows, points)

    def coverage_of_each(self, areas: list[list[tuple[float, float]]]) -> list[dict[str, Any] | None] | None:
        """:meth:`coverage` of several areas off ONE read of the ring's index,
        in the areas' order. None when the ring cannot be asked."""
        newest = self._index_newest
        rows = self._ring_rows(None if newest is None else newest + 1)
        if rows is None:
            return None
        self._index_rows.extend(rows)
        stamps = [int(r["log_time"]) for r in rows if r.get("log_time") is not None]
        if stamps:
            self._index_newest = max(stamps) if newest is None else max(newest, max(stamps))
        return [self._held_by(self._index_rows, area) for area in areas]

    def _ring_rows(self, since: int | None) -> list[dict[str, Any]] | None:
        rows = self.vehicle.index(since=since)
        if rows is None:
            self.last_refusal = f"the recorder's ring cannot be asked: {self.vehicle.last_error or 'no index'}"
        return rows

    def _held_by(self, rows: list[dict[str, Any]], points: list[tuple[float, float]]) -> dict[str, Any]:
        intr = self.boresight.sensor.intrinsics
        # Only a frame near the area can hold it: rows more than HOLD_NEAR_DEG
        # off are skipped before any geometry.
        if points:
            clon = sum(p[0] for p in points) / len(points)
            clat = sum(p[1] for p in points) / len(points)
            rows = [r for r in rows
                    if abs(float((r.get("pose") or {}).get("lat") or 0.0) - clat) <= HOLD_NEAR_DEG
                    and abs(float((r.get("pose") or {}).get("lon") or 0.0) - clon) <= HOLD_NEAR_DEG]
        best: list[tuple[float, int] | None] = [None] * len(points)
        for row in rows:
            tag = row.get("pose") or {}
            if tag.get("alt_msl_m") is None or tag.get("heading_deg") is None:
                continue
            seq = int(row["seq"])
            agl = float(tag["alt_msl_m"]) - self.floor_msl_m
            for i, (lon, lat) in enumerate(points):
                angle = off_nadir_lonlat(tag["lat"], tag["lon"], agl, tag["heading_deg"],
                                         tag.get("roll_deg") or 0.0, tag.get("pitch_deg") or 0.0,
                                         lat, lon, hfov_deg=intr.hfov_deg, vfov_deg=intr.vfov_deg)
                if angle is None:
                    continue
                # At one angle the fresher frame wins.
                if best[i] is None or (angle, -seq) < (best[i][0], -best[i][1]):
                    best[i] = (float(angle), seq)
        held = sum(1 for b in best if b is not None)
        self.last_refusal = ""
        return {"points": len(points), "held": held,
                "fraction": (held / len(points)) if points else 0.0, "best": best, "frames": len(rows)}

    def observe_area(self, points: list[tuple[float, float]]) -> Observation | None:
        """The ground the pass held over an AREA, as one north-up observation on
        the map grid: the frames that saw each of ``points`` nearest nadir,
        fused (:meth:`_fused`). ``last["coverage"]`` says how much of the area
        was held and by which frames; a partial pass is served as the part it
        held. None with ``last_refusal`` when no frame held any of it.

        Off nadir is the rank: a flown sortie's marks binned by the angle the
        frame saw each at read a CE50 of under a metre and a half within 2.5
        degrees and two to thirteen metres past 7.5. Attitude is in the angle
        already and height does not move within one pass, so nothing else
        ranks. Only the frames that win a point are fetched, so choosing costs
        no frame bytes.
        """
        cov = self.coverage(points)
        if cov is None:
            return None
        if not cov["held"]:
            self.last_refusal = (f"no frame over any of the area's {cov['points']} points in the "
                                 f"recorder's ring ({cov['frames']} frames asked)")
            return None
        chosen: dict[int, float] = {}
        for b in cov["best"]:
            if b is not None:
                chosen[b[1]] = min(chosen.get(b[1], b[0]), b[0])
        want = sorted((angle, -seq) for seq, angle in chosen.items())
        obs = self._fused(want, ranked=cov["frames"], about=f"the area's {cov['points']} points")
        if obs is not None:
            self.last["coverage"] = {"fraction": round(cov["fraction"], 3), "points": cov["points"],
                                     "held": cov["held"], "seqs": [-neg for _a, neg in want],
                                     "fetch_failed": self.last["mosaic"]["fetch_failed"]}
        return obs

    def _fused(self, want: list[tuple[float, int]], *, ranked: int, about: str) -> Observation | None:
        """``want``'s frames, ``(off-nadir degrees, -sequence)`` nearest nadir
        first, fetched off the ring, each placed by its own tagged pose
        through the boresight and its own registration, composited on one
        north-up grid at the frames' median ground sample."""
        placed: list[Any] = []
        records: list[dict[str, Any]] = []
        unfetched = 0
        for angle, negative_seq in want:
            got = self.vehicle.frame(-negative_seq)
            if got is None:
                # The link, not the ring: the index said the frame is there.
                unfetched += 1
                self.last_refusal = (f"ring frame {-negative_seq} not fetched: "
                                     f"{self.vehicle.last_error or 'no reason recorded'}")
                continue
            jpeg, t_host, log_time, seq, tag = got
            pose = tag_pose(tag, t_host)
            if pose is None:
                self.last_refusal = f"ring frame {seq} carries no pose"
                continue
            posed = self._posed(jpeg, t_host, pose=pose)
            if posed is None:
                continue
            corners = posed.corners
            if self.register is not None:
                record = self._register_ring(seq, posed, t_host)
                # The heading the frame was placed at and the track beside it: a crab is their difference.
                record.update(heading_deg=posed.pose.heading_deg, track_deg=posed.pose.track_deg,
                              attitude=posed.pose.attitude)
                records.append(record)
                if record["applied"]:
                    pose_xy = (posed.pose_xy[0] + record["shift_e_m"], posed.pose_xy[1] + record["shift_n_m"])
                    corners = ground_corners_xy(posed.sensor, pose_xy, posed.agl, posed.yaw,
                                                airframe_roll_deg=posed.roll, airframe_pitch_deg=posed.pitch_proj,
                                                ground_z=posed.ground_tile)
            placed.append(SimpleNamespace(angle=angle, seq=seq, t_host=t_host, log_time=log_time,
                                          pose=posed.pose, corners=corners, rgb=posed.rgb,
                                          camera_z=posed.camera_z, agl=posed.agl, sensor=posed.sensor))
        if not placed:
            self.last_refusal = (f"none of the {len(want)} frames over {about} could be placed: "
                                 f"{self.last_refusal or 'no reason recorded'}")
            return None
        # The grid's cell is what the FRAMES carry: resampling onto a finer grid
        # invents resolution the sensor never had, onto a coarser one throws it away.
        gsd = [ground_gsd_m(f.agl, hfov_deg=f.sensor.intrinsics.hfov_deg,
                            image_width=f.sensor.intrinsics.image_width) for f in placed]
        gsd_m = float(np.median([g for g in gsd if g])) if any(gsd) else GSD_FLOOR_M
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
                rgb, alpha, (x0, y0, x1, y1), fit = register_mosaic(rgb, alpha, (x0, y0, x1, y1), render)
            if fit["applied"]:
                log.info("mosaic registered on the map: %+.1f E %+.1f N %+.2f deg, response %.2f, %.0f ms",
                         fit["shift_e_m"], fit["shift_n_m"], fit["shift_yaw_deg"], fit["response"], fit["ms"])
            else:
                log.info("mosaic not registered on the map: %s", fit["why"])
        camera_z = float(np.median([f.camera_z for f in placed]))
        newest = max(placed, key=lambda f: f.seq)
        self.last = {
            "log_time": newest.log_time, "seq": newest.seq, "t_host": newest.t_host,
            "age_s": self.vehicle.now() - newest.t_host, "attitude": newest.pose.attitude,
            "gap_s": newest.pose.gap_s, "utc": newest.pose.utc,
            "boresight": self.boresight.path.name if self.boresight.path else "datasheet",
            "registration": fused_registration(records) if self.register is not None else None,
            "corners_xy": [(x0, y1), (x1, y1), (x1, y0), (x0, y0)],
            "mosaic": {"placed": n, "asked": len(want), "ranked": ranked, "gsd_m": round(gsd_m, 3),
                       "fetch_failed": unfetched, "fit": fit, "seqs": [f.seq for f in placed],
                       "off_nadir_deg": [round(f.angle, 2) for f in placed],
                       "registrations": [{k: r.get(k) for k in ("seq", "applied", "source", "why", "shift_e_m",
                                                                 "shift_n_m", "shift_yaw_deg", "heading_deg",
                                                                 "track_deg", "attitude")} for r in records]},
            "off_nadir_deg": round(min(f.angle for f in placed), 2),
        }
        self.last_refusal = ""
        return ortho_observation(rgb, alpha, (x0, y0, x1, y1), camera_z, tile_ground_z=self.tile_ground_z,
                                 tile=self.tile)


def open_live(*, vehicle, tiles_dir, map_name: str, sensor: SensorSpec, boresight_path: str | Path | None = None,
              register_raster: str | Path | None = None, tile=None, corrector: bool = True) -> LiveSource:
    """A live source over ``vehicle``, failing on the ramp rather than on the first flight.

    No boresight file flies the datasheet camera. ``register_raster`` names
    the raster directory each frame is registered against
    (:func:`map_render.load_raster`); with it the corrector starts unless
    ``corrector`` is False.
    """
    from maps.perception.ortho.ground_level import GROUND_LEVEL  # noqa: PLC0415

    if lonlat_to_local(Path(tiles_dir), map_name, 0.0, 0.0) is None:
        raise ValueError(f"{map_name} carries no georef under {tiles_dir}: a fix cannot be placed in the tile frame")
    if map_name not in GROUND_LEVEL:
        raise ValueError(f"{map_name} has no floor in GROUND_LEVEL: the bench's bridge for it is not active")
    if map_name not in FLOOR_MSL_M:
        raise ValueError(f"{map_name} has no orthometric floor in tile_georef.FLOOR_MSL_M")
    boresight = Boresight.load(boresight_path) if boresight_path else Boresight(sensor=sensor)
    if boresight_path is None:
        log.warning("live source: no boresight.json; flying the datasheet camera")
    register = None
    if register_raster is not None:
        ortho = map_render.load_raster(register_raster, map_name)
        if ortho is None:
            raise ValueError(f"no {map_name} raster under {register_raster} to register the frames against")
        register = Registration(ortho=ortho, render=map_render.renderer(ortho), floor_msl_m=FLOOR_MSL_M[map_name],
                                terrain=map_render.terrain(map_name, str(tiles_dir)))
    source = LiveSource(tiles_dir=tiles_dir, map_name=map_name, boresight=boresight, vehicle=vehicle, tile=tile,
                        tile_ground_z=float(GROUND_LEVEL[map_name]), floor_msl_m=FLOOR_MSL_M[map_name],
                        register=register)
    if register is not None and corrector:
        source.start_corrector()
    return source
