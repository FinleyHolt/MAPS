"""The station: one mission flown on the twin, and every finished run, served to the console.

    python -m station.serve --ortho <raster dir>     # fly Cal City missions on the twin, review them after
    python -m station.serve                          # review the runs under --out: no raster, model or GPU

One process holds the mission in the seat (``benchmarks.flight.mission.Mission``
over ``benchmarks.flight.twin.Twin``), publishes its state as an event stream
and serves the built console (``station/web/dist``) from the same port, so the
page is a browser tab and nothing else. A mission's reasoner and segmenter
come from the environment as the bench's do (``SEARCH_REASONER_URL``,
``SEARCH_SAM3_1_URL``; the top-level README's Endpoints section).

Each mission lands in ``<out>/<UTC stamp>/`` with the run records a flown run
carries; the console's review reads those folders, so a run flown by
``python -m benchmarks.flight.twin --out <out>`` or copied in from another
machine reviews the same way.

ponytail: one mission in the seat and one map per process; a second seat is a
second process on another port.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import logging
import math
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from benchmarks.citynav.analysis.rows import iter_json_rows
from benchmarks.citynav.episode_workdir import (CONVERSATION_JSONL, FRAMES_JSONL, TRACK_JSONL,
                                                open_episode_dir)
from benchmarks.citynav.episodes import EPISODES_JSONL, STATUS_INFRA
from benchmarks.flight import map_render
from benchmarks.flight.tile_georef import local_to_lonlat, lonlat_to_local, map_bounds

log = logging.getLogger("station")

WEB_DIST = Path(__file__).parent / "web" / "dist"
#: The zoom range the basemap is served over: the whole block down to about 0.3 m a pixel.
TILE_ZOOMS = (10, 19)
TILE_PX = 256
#: How often the event stream looks for a change, and how often it says it is alive.
STREAM_PERIOD_S, STREAM_KEEPALIVE_S = 0.5, 15.0
#: The live trail's length, in ownship reads (one a second).
TRAIL_POINTS = 3600
LIVE = ("flying", "awaiting_tasking")

_PATHS = frozenset({"footway", "cycleway", "path", "service", "track", "steps", "pedestrian", "bridleway"})
_PITCHES = frozenset({"pitch", "track", "stadium", "playground", "sports_centre", "golf_course"})


def prior_group(f: dict) -> str | None:
    """The legend group a bake feature draws in, or None for a row with no geometry."""
    contour = f.get("world_contour") or ()
    if not contour:
        return None
    if len(contour) == 1:
        return "node"
    highway = str(f.get("highway", "")).removesuffix("_link")
    if highway:
        return "path" if highway in _PATHS else "road"
    if f.get("building"):
        return "building"
    if f.get("leisure") in _PITCHES or f.get("sport"):
        return "pitch"
    if f.get("amenity") == "parking":
        return "parking"
    if f.get("barrier") or f.get("railway") or f.get("waterway"):
        return "barrier"
    if f.get("landuse") or f.get("natural") or f.get("leisure"):
        return "grass"
    return "other"


@functools.lru_cache(maxsize=2)
def prior_json(tiles_dir: str, map_name: str) -> bytes:
    """The bake's features as ``[[group, [[x, y], ...]], ...]`` in local metres, built once."""
    rows = json.loads((Path(tiles_dir) / f"{map_name}_features.json").read_text())["prior"]
    out = [[g, [[round(float(x), 1), round(float(y), 1)] for x, y in f["world_contour"]]]
           for f in rows if (g := prior_group(f)) is not None]
    return json.dumps(out, separators=(",", ":")).encode()


def tile_lonlat(z: int, x: float, y: float) -> tuple[float, float]:
    """``(lon, lat)`` of a point in XYZ tile space (web mercator)."""
    n = 2 ** z
    return x / n * 360.0 - 180.0, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


def tile_png(ortho, tiles_dir, map_name: str, z: int, x: int, y: int) -> bytes | None:
    """One 256 px web-mercator tile cut from the raster, transparent off its
    edge; None where the tile holds none of it. The tile's four corners are
    placed in the raster and the quad between them warped: over one tile the
    mercator grid and the bake's UTM grid differ by less than a pixel."""
    import cv2  # noqa: PLC0415

    rgb, bounds, ppm = ortho
    rows, cols = rgb.shape[:2]
    quad = []
    for cx, cy in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)):
        local = lonlat_to_local(tiles_dir, map_name, *tile_lonlat(z, cx, cy))
        if local is None:
            return None
        quad.append([(local[0] - bounds.x_min) * ppm, rows - (local[1] - bounds.y_min) * ppm])
    src = np.float32(quad)
    c0, r0 = (int(v) for v in np.floor(src.min(0)))
    c1, r1 = (int(v) for v in np.ceil(src.max(0)))
    # A tile far wider than 256 px of raster is read every ``step``-th pixel.
    step = max(1, min(c1 - c0, r1 - r0) // (2 * TILE_PX))
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, cols), min(r1, rows)
    if c1 <= c0 or r1 <= r0:
        return None
    crop = np.ascontiguousarray(rgb[r0:r1:step, c0:c1:step, :3])
    warp = cv2.getPerspectiveTransform((src - np.float32([c0, r0])) / step,
                                       np.float32([[0, 0], [TILE_PX, 0], [TILE_PX, TILE_PX], [0, TILE_PX]]))
    size = (TILE_PX, TILE_PX)
    out = cv2.warpPerspective(crop, warp, size, flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(np.full(crop.shape[:2], 255, np.uint8), warp, size, flags=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", np.dstack([out[..., ::-1], alpha]))
    return png.tobytes() if ok else None


class Station:
    """The seat, the catalog and the finished runs of one map."""

    def __init__(self, *, config: str, out: Path, ortho_dir: Path | None, register: Path | None) -> None:
        import benchmarks.calcity  # noqa: PLC0415

        benchmarks.calcity.use()
        from benchmarks.citynav.adapter import load_split, parse_episode  # noqa: PLC0415
        from benchmarks.citynav.paths import DEFAULT_SPLIT, TILES_DIR, split_path  # noqa: PLC0415

        self.config, self.out, self.register = config, Path(out), register
        self.split, self.tiles_dir = DEFAULT_SPLIT, Path(TILES_DIR)
        self.catalog = [parse_episode(row, row_index=i) for i, row in enumerate(load_split(split_path(self.split)))]
        self.by_id = {e.episode_id: e for e in self.catalog}
        self.map_name = self.catalog[0].map_name
        #: Whether the bake is on disk: without it the map has no georef and the page draws no basemap.
        self.baked = (self.tiles_dir / f"{self.map_name}_meta.json").is_file()
        self.ortho = map_render.load_raster(ortho_dir, self.map_name) if ortho_dir is not None else None
        if ortho_dir is not None and self.ortho is None:
            raise SystemExit(f"no {self.map_name} raster under {ortho_dir}: a raster directory holds "
                             f"{map_render.raster_pair(self.map_name)}")
        self._lock = threading.Lock()
        self.mission: Any = None
        self._sortie: Path | None = None
        self._trail: list[list[float]] = []

    # -- placing things on the basemap --------------------------------------

    def lonlat(self, xy) -> tuple[float, float] | None:
        return local_to_lonlat(self.tiles_dir, self.map_name, float(xy[0]), float(xy[1])) if self.baked else None

    def _placed(self, doc: dict[str, Any], xy) -> dict[str, Any]:
        """``doc`` with ``lon``/``lat`` of local ``xy``, where the bake carries a georef."""
        ll = self.lonlat(xy) if xy is not None else None
        if ll is not None:
            doc["lon"], doc["lat"] = ll
        return doc

    def target_lonlat(self, episode_id: str) -> list[list[float]] | None:
        mask = getattr(self.by_id.get(episode_id), "target_mask", None)
        ring = [self.lonlat(p) for p in mask or ()]
        return [list(p) for p in ring] if ring and all(ring) else None

    def entry(self, e) -> dict[str, Any]:
        return {"episode_id": e.episode_id, "instruction": e.description, "map": e.map_name,
                "target_lonlat": self.target_lonlat(e.episode_id)}

    # -- a run as the console's state document ------------------------------

    def _observations(self, workdir: Path, targets: list[str | None]) -> list[dict[str, Any]]:
        out = []
        for f in iter_json_rows(workdir / FRAMES_JSONL):
            if not f.get("observation"):
                continue
            n = int(f.get("flight") or 0)
            mosaic, window = f.get("mosaic") or {}, f.get("world_window")
            out.append({"sha": f["observation"], "xy": f.get("pose_xy") or [0.0, 0.0], "flight": n,
                        "target": (targets[n] if n < len(targets) else None) or f"flight {n + 1}",
                        "looks": int(mosaic.get("asked") or 1), "placed": int(mosaic.get("placed") or 0),
                        "ground_m2": (window[2] - window[0]) * (window[3] - window[1]) if window else 0.0,
                        "world_window": window, "shape": f.get("shape"), "t_s": f.get("t_s") or 0.0})
        return out

    def _flights(self, points, targets: list[str | None]) -> list[dict[str, Any]]:
        return [self._placed({"step_idx": i, "target": targets[i] if i < len(targets) else None,
                              "resolved_xy": [float(xy[0]), float(xy[1])]}, xy) for i, xy in enumerate(points)]

    def run_state(self, sortie: Path, row: dict[str, Any]) -> dict[str, Any]:
        """A ledger row and its run folder as the console's ``MissionState``."""
        workdir = sortie / row["workdir"]
        prov = row.get("provenance") or {}
        transcript = list(iter_json_rows(workdir / CONVERSATION_JSONL))
        return {
            "version": 0, "path": f"{sortie.name}/{row['workdir']}", "episode_id": row.get("episode_id"),
            "split": row.get("split"), "kind": "replay", "run_no": row.get("run_no"),
            "status": "failed" if row.get("status") == STATUS_INFRA else "ended",
            "instruction": row.get("description") or "", "ended_by": row.get("ended_by"), "error": row.get("error"),
            "reply_text": None, "turns_used": row.get("turns_used") or 0,
            "flights_flown": row.get("flights_flown") or 0, "started_at": row.get("ts"),
            "git_commit": prov.get("git_commit"), "git_dirty": prov.get("git_dirty"),
            "map_name": row.get("map_name"), "target_lonlat": self.target_lonlat(row.get("episode_id") or ""),
            "transcript": transcript, "flights": self._flights((row.get("visited_xy") or [])[1:], []),
            "observations": self._observations(workdir, []), "leg": None, "score": row.get("metrics"),
            "duration_s": row.get("elapsed_s"),
        }

    def seat_state(self) -> dict[str, Any] | None:
        """The mission in the seat as the console's ``MissionState``, or None with the seat empty."""
        m = self.mission
        if m is None:
            return None
        flights = list(m.flights)
        targets = [str(f.get("target") or f.get("target_fid") or "") or None for f in flights]
        transcript = list(m.transcript)
        observations = self._observations(m.workdir, targets)
        return {
            "version": hash((m.status, len(transcript), len(flights), len(observations), m.error,
                             json.dumps(m.waiting, default=str))),
            "path": f"{m.workdir.parent.name}/{m.workdir.name}", "episode_id": m.episode_id, "split": m.split,
            "kind": "twin", "run_no": m.run_no, "status": "ready" if m.status == "idle" else m.status,
            "instruction": m.instruction, "ended_by": m.ended_by, "error": m.error, "reply_text": None,
            "turns_used": m.turns(), "flights_flown": len(flights), "started_at": m.started_at,
            "git_commit": m.git_commit, "git_dirty": m.git_dirty, "map_name": m.map_name or self.map_name,
            "target_lonlat": self.target_lonlat(m.episode_id), "transcript": transcript,
            "flights": self._flights([f["resolved_xy"] for f in flights if f.get("resolved_xy")], targets),
            "observations": observations, "score": None,
            "leg": ({"closest_m": 0.0, "short_passes": 0, "packageVersion": "", **m.waiting}
                    if m.waiting is not None else None),
        }

    def runs(self) -> list[dict[str, Any]]:
        """Every finished run under the output root, newest first; the one in the seat is the seat's."""
        seat = self._sortie if self.mission is not None and self.mission.status in LIVE else None
        out = []
        for ledger in sorted(self.out.glob(f"*/{EPISODES_JSONL}"), reverse=True):
            if ledger.parent != seat:
                out += [self.run_state(ledger.parent, r) for r in reversed(list(iter_json_rows(ledger)))
                        if r.get("workdir")]
        return out

    def run_at(self, path: str) -> tuple[Path, dict[str, Any]]:
        """The sortie folder and ledger row a ``<sortie>/<workdir>`` path names."""
        sortie_name, _, workdir = path.partition("/")
        sortie = self.out / sortie_name
        if not sortie_name or ".." in Path(path).parts or not (sortie / EPISODES_JSONL).is_file():
            raise HTTPException(404, f"no run at {path!r}")
        rows = [r for r in iter_json_rows(sortie / EPISODES_JSONL) if r.get("workdir")]
        row = next((r for r in reversed(rows) if r["workdir"] == workdir), rows[-1] if rows and not workdir else None)
        if row is None:
            raise HTTPException(404, f"no run at {path!r}")
        return sortie, row

    def track(self, sortie: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
        """The run's flown path off the sortie's ``flight_track.jsonl``, mission seconds from its start."""
        t0, length = row.get("started_at_utc"), row.get("elapsed_s")
        if t0 is None:
            return []
        out = []
        for fix in iter_json_rows(sortie / TRACK_JSONL):
            t = float(fix["t_unix"]) - float(t0)
            if -1e-6 <= t <= (float(length) if length is not None else math.inf) + 1.0:
                out.append(self._placed({"t_s": round(t, 2), "x": fix["x"], "y": fix["y"],
                                         "track_deg": fix.get("track_deg") or 0.0,
                                         "agl_m": fix.get("alt_agl_m") or 0.0,
                                         "speed_mps": fix.get("ground_speed_mps") or 0.0}, (fix["x"], fix["y"])))
        return out

    # -- the aircraft --------------------------------------------------------

    def ownship(self, path: str | None, t: float) -> dict[str, Any] | None:
        """Where the aircraft is: the twin's pose now, or a reviewed run's at mission second ``t``."""
        if path:
            sortie, row = self.run_at(path)
            track = self.track(sortie, row)
            if not track:
                return None
            at = min(track, key=lambda p: abs(p["t_s"] - t))
            return {**at, "map": row.get("map_name"), "duration_s": row.get("elapsed_s"), "live": False}
        m = self.mission
        if m is None or m.status not in LIVE:
            return None
        pose, own = m.live.pose_now(), m.live.ownship()
        if pose is None or own is None:
            return None
        doc = self._placed({"x": own["x"], "y": own["y"], "track_deg": pose.track_deg, "agl_m": own["agl_m"],
                            "speed_mps": pose.speed_mps, "map": self.map_name, "t_s": m.elapsed_s(),
                            "live": True}, (own["x"], own["y"]))
        if "lon" in doc and (not self._trail or self._trail[-1] != [doc["lon"], doc["lat"]]):
            self._trail = (self._trail + [[doc["lon"], doc["lat"]]])[-TRAIL_POINTS:]
        return {**doc, "trail": list(self._trail)}

    # -- the seat ------------------------------------------------------------

    def start(self, episode_id: str | None, instruction: str, effort: str | None) -> dict[str, Any]:
        """Fly a catalog row, or the operator's own words from the map's centre, on a twin of its own."""
        from benchmarks.citynav.run_benchmark.episode_runner import start_altitude_m  # noqa: PLC0415
        from benchmarks.flight.live_source import open_live  # noqa: PLC0415
        from benchmarks.flight.mission import Mission, flight_config, resolve_episode  # noqa: PLC0415
        from benchmarks.flight.twin import Twin  # noqa: PLC0415
        from maps.perception.geometry.sensor_spec import resolve_sensor  # noqa: PLC0415

        if self.ortho is None:
            raise HTTPException(409, "this station reviews runs only: start it with --ortho <raster dir> to fly")
        with self._lock:
            if self.mission is not None and self.mission.status in LIVE:
                raise HTTPException(409, f"mission {self.mission.episode_id} is flying; stop it first")
            if episode_id is not None and episode_id not in self.by_id:
                raise HTTPException(404, f"{self.split} holds no episode {episode_id}")
            start_xy = None
            if episode_id is None:
                b = map_bounds(self.tiles_dir, self.map_name)
                start_xy = ((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2)
                episode_id = f"{self.map_name}::operator-{time.strftime('%H%M%S', time.gmtime())}"
            cfg = flight_config(self.config)
            sensor = resolve_sensor(cfg.sensor)
            episode = resolve_episode(episode_id, self.split, start_xy)
            start_alt, ground = start_altitude_m(cfg, self.map_name, episode.start_z)
            twin = Twin(tiles_dir=self.tiles_dir, map_name=self.map_name, ortho=self.ortho, sensor=sensor,
                        start_xy=episode.start_xy, agl_m=start_alt - ground)
            live = open_live(vehicle=twin, tiles_dir=self.tiles_dir, map_name=self.map_name, sensor=sensor,
                             register_raster=self.register)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._sortie = self.out / stamp
            ep = open_episode_dir(self._sortie, episode_id=episode_id, run_no=1)
            mission = Mission(episode_id=episode_id, split=self.split, config=self.config, workdir=ep.dir, live=live,
                              instruction=instruction.strip() or None, recorder=ep.recorder, close_log=ep.close,
                              run_id=stamp, run_no=1, start_xy=start_xy, reasoning_effort=effort,
                              agl_m=start_alt - ground)
            self.mission, self._trail = mission, []
            mission.start()
            # The live source closes when the mission's thread ends, however it ends.
            threading.Thread(target=lambda: (mission.wait(), live.close()), name="station-close", daemon=True).start()
        return self.seat_state()

    def clear(self) -> None:
        """Stop the mission in the seat and empty it; its run stays under the output root."""
        with self._lock:
            mission, self.mission, self._trail = self.mission, None, []
        if mission is not None:
            mission.stop()


class _Next(BaseModel):
    episode_id: str | None = None
    instruction: str = ""
    reasoning_effort: str | None = None


class _Tasking(BaseModel):
    text: str


def build_app(station: Station, web: Path | None = WEB_DIST) -> FastAPI:
    app = FastAPI(title="MAPS station")

    @app.get("/missions")
    def missions(q: str | None = None, limit: int = 200) -> dict[str, Any]:
        words = (q or "").lower().split()
        hits = [e for e in station.catalog
                if all(w in e.description.lower() or w in e.episode_id.lower() for w in words)]
        return {"missions": [station.entry(e) for e in hits[:max(0, limit)]], "total": len(station.catalog)}

    @app.get("/reasoner")
    def reasoner() -> dict[str, Any]:
        """The served model and the reasoning efforts its card takes; ``model`` is None while no reasoner answers."""
        from maps.llm import endpoint_spec, model_cards  # noqa: PLC0415

        try:
            model = (endpoint_spec.spec() if endpoint_spec.spec_installed()
                     else endpoint_spec.discover_from_env()).model_id
        except Exception as exc:  # noqa: BLE001 -- a station with no reasoner still reviews
            return {"model": None, "efforts": [], "error": str(exc)}
        return {"model": model, "efforts": list(model_cards.card_for(model).get("efforts", ())), "error": None}

    @app.get("/missions/runs")
    def missions_runs() -> dict[str, Any]:
        return {"runs": station.runs()}

    @app.get("/mission/events")
    def mission_events() -> StreamingResponse:
        """The seat's state document, whole, each time it changes."""
        def stream():
            sent, said = object(), 0.0
            while True:
                state = station.seat_state()
                version = None if state is None else state["version"]
                if version != sent:
                    sent, said = version, time.monotonic()
                    yield f"data: {json.dumps(state, default=str)}\n\n"
                elif time.monotonic() - said > STREAM_KEEPALIVE_S:
                    said = time.monotonic()
                    yield ": alive\n\n"
                time.sleep(STREAM_PERIOD_S)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.post("/mission/next")
    def mission_next(req: _Next) -> dict[str, Any]:
        if not req.instruction.strip() and req.episode_id is None:
            raise HTTPException(400, "a mission needs a catalog row or the operator's words")
        return station.start(req.episode_id, req.instruction, req.reasoning_effort)

    @app.post("/mission/tasking")
    def mission_tasking(req: _Tasking) -> dict[str, Any]:
        m = station.mission
        if m is None or m.status not in LIVE:
            raise HTTPException(409, "no mission is flying")
        m.tasking(req.text)
        return station.seat_state()

    @app.post("/mission/clear")
    def mission_clear() -> dict[str, Any]:
        station.clear()
        return {}

    @app.get("/replay")
    def replay(path: str) -> dict[str, Any]:
        sortie, row = station.run_at(path)
        return {**station.run_state(sortie, row), "track": station.track(sortie, row)}

    @app.get("/mission/ownship")
    def mission_ownship(path: str | None = None, t: float = 0.0) -> dict[str, Any] | None:
        return station.ownship(path, t)

    @app.get("/mission/prior.json")
    def mission_prior(map: str | None = None) -> Response:
        name = map or station.map_name
        if not (station.tiles_dir / f"{name}_features.json").is_file():
            raise HTTPException(404, f"no features for {name!r}")
        return Response(prior_json(str(station.tiles_dir), name), media_type="application/json",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/area")
    def area() -> dict[str, Any]:
        if not station.baked:
            raise HTTPException(404, "no bake: the map has no georef")
        b = map_bounds(station.tiles_dir, station.map_name)
        (w, s), (e, n) = station.lonlat((b.x_min, b.y_min)), station.lonlat((b.x_max, b.y_max))
        return {"map": station.map_name, "bbox": [w, s, e, n]}

    @app.get("/location")
    def location() -> dict[str, Any]:
        """The basemap: the raster's tiles over the raster's extent; 404 on a station with no raster."""
        if station.ortho is None:
            raise HTTPException(404, "no raster: the map draws no basemap")
        b = station.ortho[1]
        (w, s), (e, n) = station.lonlat((b.x_min, b.y_min)), station.lonlat((b.x_max, b.y_max))
        return {"name": station.map_name, "label": station.map_name, "bounds": [w, s, e, n],
                "center": {"lat": (s + n) / 2, "lon": (w + e) / 2}, "tiles": "tiles/{z}/{x}/{y}.png",
                "minzoom": TILE_ZOOMS[0], "maxzoom": TILE_ZOOMS[1], "attribution": ""}

    tile = functools.lru_cache(maxsize=1024)(
        lambda z, x, y: tile_png(station.ortho, station.tiles_dir, station.map_name, z, x, y))

    @app.get("/tiles/{z}/{x}/{y}.png")
    def tiles(z: int, x: int, y: int) -> Response:
        png = tile(z, x, y) if station.ortho is not None and TILE_ZOOMS[0] <= z <= TILE_ZOOMS[1] else None
        if png is None:
            raise HTTPException(404)
        return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/run/file")
    def run_file(path: str) -> FileResponse:
        """One file of a run, by ``<sortie>/<workdir>/<name>``; nothing outside the output root."""
        root = station.out.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target, media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    if web is not None and (web / "index.html").is_file():
        root = web.resolve()

        @app.get("/{path:path}")
        def console(path: str) -> FileResponse:
            target = (root / path).resolve()
            if path and target.is_relative_to(root) and target.is_file():
                return FileResponse(target)
            # The page names its hashed assets, so it is asked for again on every load.
            return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/calcity.yaml")
    ap.add_argument("--ortho", type=Path, default=None,
                    help="the raster directory the twin's frames and the basemap are cut from; unset reviews runs only")
    ap.add_argument("--register", type=Path, default=None,
                    help="the raster directory each frame is registered against; unset registers nothing")
    ap.add_argument("--out", type=Path, default=Path("output/station"), help="where runs land and are reviewed from")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    import uvicorn  # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)
    station = Station(config=args.config, out=args.out, ortho_dir=args.ortho, register=args.register)
    if not (WEB_DIST / "index.html").is_file():
        log.warning("no built console at %s: run `npm ci && npm run build` in station/web", WEB_DIST)
    uvicorn.run(build_app(station), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
