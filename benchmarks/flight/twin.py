"""A kinematic twin of the aircraft over a supplied ortho, and the CLI that flies one Cal City mission through it.

    python -m benchmarks.flight.twin --config configs/calcity.yaml --episode <id> --ortho <raster dir>

The twin is a :class:`benchmarks.flight.vehicle.Vehicle`: it flies each posted
leg straight on through the rest of the pass at the C-12's speed, wings level,
and records a pose-tagged frame at the camera's rate, its pixels rendered on
request off the ortho (``<map>_rgb.tif`` beside ``<map>.tif``) at the camera
pose the live source places it by. The endpoints come from the environment as
the bench's do; the run lands under ``--out/<UTC stamp>/`` with the run
records a flown run carries.

ponytail: the twin's clock moves only while the harness waits on it, so the
aircraft holds still while the model thinks and a twin sortie's time to
identification and SPL path leave the thinking out; a real-time clock is the
upgrade.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.flight import map_render
from benchmarks.flight.airframe import C12_LEVER_M, C12_SPEED_MPS
from benchmarks.flight.nav_track import Pose, camera_pose_local, pose_tag
from benchmarks.flight.tile_georef import FLOOR_MSL_M, local_to_lonlat
from benchmarks.flight.vehicle import Leg

#: The recorder's frame rate: two frames a second.
CAMERA_HZ = 2.0
#: A twin frame's JPEG quality, the recorder's own.
FRAME_JPEG_QUALITY = 90
#: The receiver's clock at the twin's time zero.
EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)


class Twin:
    """The aircraft flown kinematically, its frames rendered off ``ortho``.

    ``start_xy`` is where it starts in the tile frame, heading north at
    ``agl_m`` over the ground; ``sensor`` is the camera its frames are
    rendered through, ``terrain`` the map's surface (flat at its floor without one).
    """

    def __init__(self, *, tiles_dir, map_name: str, ortho, sensor, start_xy: tuple[float, float], agl_m: float,
                 terrain=None) -> None:
        self.tiles_dir, self.map_name, self.sensor = Path(tiles_dir), map_name, sensor
        self.floor_msl_m = FLOOR_MSL_M[map_name]
        self.terrain = terrain
        self.agl_m = float(agl_m)
        self._render = map_render.renderer(ortho)
        self.t = 0.0
        #: Time slept and not yet flown: less than one camera tick.
        self._owed = 0.0
        self.xy = (float(start_xy[0]), float(start_xy[1]))
        self.heading_deg = 0.0
        self.route: list[tuple[float, float]] = []
        #: Called after each sleep, once the clock has moved (the live source's corrector).
        self.on_sleep: list = []
        #: Every frame recorded: ``{"seq", "log_time", "t_host", "pose"}``.
        self.rows: list[dict[str, Any]] = []
        #: The track at every camera tick, ``(t, x, y, heading_deg)``, which ``pose(t)`` interpolates.
        self._track: list[tuple[float, float, float, float]] = [(0.0, *self.xy, 0.0)]
        self.last_error = ""
        self._record()

    def ground(self, xy) -> float:
        return map_render.ground_msl(self.terrain, self.floor_msl_m, xy)

    def post(self, leg: Leg) -> None:
        self.route = [tuple(leg.point)] + [tuple(p) for p in leg.ahead]

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        """Fly ``seconds`` of the route on the twin's clock, a camera tick at a time."""
        dt_s = 1.0 / CAMERA_HZ
        self._owed += max(0.0, float(seconds))
        while self._owed >= dt_s - 1e-9:
            self._advance(dt_s)
            self._owed -= dt_s
            self.t = len(self._track) / CAMERA_HZ
            self._track.append((self.t, *self.xy, self.heading_deg))
            self._record()
        for hook in self.on_sleep:
            hook()

    def _advance(self, dt_s: float) -> None:
        """Straight at the route's next point at the airframe's speed, through it
        and on to the next; past the last, on along the same heading."""
        step = C12_SPEED_MPS * dt_s
        while self.route:
            tx, ty = self.route[0]
            dx, dy = tx - self.xy[0], ty - self.xy[1]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                self.heading_deg = math.degrees(math.atan2(dx, dy)) % 360.0
            if d > step:
                break
            self.xy, step = (tx, ty), step - d
            self.route.pop(0)
        h = math.radians(self.heading_deg)
        self.xy = (self.xy[0] + step * math.sin(h), self.xy[1] + step * math.cos(h))

    def _pose_at_track(self, t: float, x: float, y: float, heading_deg: float) -> Pose | None:
        lonlat = local_to_lonlat(self.tiles_dir, self.map_name, x, y)
        if lonlat is None:
            return None
        alt = self.ground((x, y)) + self.agl_m + C12_LEVER_M[2]
        utc = (EPOCH + dt.timedelta(seconds=t)).isoformat()
        return Pose(t=t, lat=lonlat[1], lon=lonlat[0], alt_msl_m=alt, track_deg=heading_deg,
                    speed_mps=C12_SPEED_MPS, utc=utc, heading_deg=heading_deg,
                    attitude="fusion")

    def _record(self) -> None:
        pose = self._pose_at_track(self.t, *self.xy, self.heading_deg)
        if pose is not None:
            self.rows.append({"seq": len(self.rows), "log_time": int(round(self.t * 1e9)), "t_host": self.t,
                              "pose": pose_tag(pose)})

    def pose(self, t: float | None = None) -> Pose | None:
        t = self.t if t is None else float(t)
        track = self._track
        if t >= track[-1][0]:
            return self._pose_at_track(t, *track[-1][1:])
        i = max(0, min(len(track) - 2, int(t * CAMERA_HZ)))
        (t0, x0, y0, h0), (t1, x1, y1, h1) = track[i], track[i + 1]
        w = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        return self._pose_at_track(t, x0 + (x1 - x0) * w, y0 + (y1 - y0) * w, h1 if w >= 0.5 else h0)

    def index(self, since: int | None = None) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rows if since is None or r["log_time"] >= since]

    def frame(self, seq: int):
        """``(jpeg, t_host, log_time, seq, tag)``: the frame rendered off the
        ortho at the camera pose the tag places it at; None off the ortho."""
        from PIL import Image  # noqa: PLC0415

        from benchmarks.flight.nav_track import tag_pose  # noqa: PLC0415

        row = dict(self.rows[seq]) if 0 <= seq < len(self.rows) else None
        if row is None:
            self.last_error = f"no frame {seq} in the ring"
            return None
        pose = tag_pose(row["pose"], row["t_host"])
        cam_xy, cam_alt, yaw = camera_pose_local(pose, self.tiles_dir, self.map_name, lever=C12_LEVER_M)
        try:
            rgb = self._render(self.sensor, cam_xy, cam_alt - self.ground(cam_xy), yaw,
                               pose.roll_deg, -pose.pitch_deg)
        except ValueError as exc:
            self.last_error = f"frame {seq} reaches off the ortho: {exc}"
            return None
        out = BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(out, format="JPEG", quality=FRAME_JPEG_QUALITY)
        self.last_error = ""
        return out.getvalue(), row["t_host"], row["log_time"], seq, row["pose"]


def _xy(text: str) -> tuple[float, float]:
    x, y = (float(v) for v in text.split(","))
    return x, y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/calcity.yaml")
    ap.add_argument("--split", default="calcity")
    ap.add_argument("--episode", required=True, help="the episode id; with --start-xy, an operator-authored mission's")
    ap.add_argument("--ortho", required=True, type=Path, help="the raster directory the twin's frames are rendered off")
    ap.add_argument("--instruction", default=None, help="the operator's words in place of the episode's")
    ap.add_argument("--start-xy", type=_xy, default=None, help="x,y: fly a mission with no row from here, under --instruction")
    ap.add_argument("--then", action="append", default=[], help="a tasking spoken after each report, in order")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--boresight", type=Path, default=None, help="a boresight solve; unset flies the datasheet camera")
    ap.add_argument("--register", type=Path, default=None,
                    help="the raster directory each frame is registered against; unset registers nothing")
    ap.add_argument("--out", type=Path, default=Path("output/flight"))
    args = ap.parse_args(argv)
    if args.start_xy is not None and not args.instruction:
        ap.error("--start-xy flies an operator-authored mission: give its words with --instruction")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    from benchmarks.flight.mission import (STATE_ENDED, Mission, activate_calcity, converse,  # noqa: PLC0415
                                           flight_config, resolve_episode, summary_line)

    activate_calcity()
    from benchmarks.citynav.episode_workdir import open_episode_dir  # noqa: PLC0415
    from benchmarks.citynav.paths import TILES_DIR  # noqa: PLC0415
    from benchmarks.flight.live_source import open_live  # noqa: PLC0415
    from benchmarks.citynav.run_benchmark.episode_runner import start_altitude_m  # noqa: PLC0415
    from maps.perception.geometry.sensor_spec import resolve_sensor  # noqa: PLC0415

    cfg = flight_config(args.config)
    sensor = resolve_sensor(cfg.sensor)
    episode = resolve_episode(args.episode, args.split, args.start_xy)
    map_name = episode.map_name
    ortho = map_render.load_raster(args.ortho, map_name)
    if ortho is None:
        print(f"no {map_name} raster under {args.ortho}: a raster directory holds {map_render.raster_pair(map_name)}",
              file=sys.stderr)
        return 2
    start_alt, ground_z = start_altitude_m(cfg, map_name, episode.start_z)
    twin = Twin(tiles_dir=TILES_DIR, map_name=map_name, ortho=ortho, sensor=sensor, start_xy=episode.start_xy,
                agl_m=start_alt - ground_z, terrain=map_render.terrain(map_name, str(TILES_DIR)))
    live = open_live(vehicle=twin, tiles_dir=TILES_DIR, map_name=map_name, sensor=sensor,
                     boresight_path=args.boresight, register_raster=args.register)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sortie = args.out / stamp
    ep = open_episode_dir(sortie, episode_id=args.episode, run_no=1)
    mission = Mission(episode_id=args.episode, split=args.split, config=args.config, workdir=ep.dir, live=live,
                      instruction=args.instruction, recorder=ep.recorder, close_log=ep.close, run_id=stamp,
                      run_no=1, start_xy=args.start_xy, reasoning_effort=args.reasoning_effort,
                      agl_m=start_alt - ground_z)
    converse(mission, args.then)
    print(summary_line(mission))
    return 0 if mission.status == STATE_ENDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
