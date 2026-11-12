"""The station: a bench's finished runs, and one mission flown from the seat, served to the console.

    python -m station.serve --out <bench out-root>                    # review CityNav runs
    python -m station.serve                                            # fly CityNav episodes, review them after
    python -m station.serve --bench calcity --ortho <raster dir>       # fly Cal City missions on the twin

One process holds a bench value (``benchmarks.bench.Bench``, built from the
environment once), the mission in the seat, and the runs under ``--out``;
it publishes the seat's state as an event stream and serves the built console
(``station/web/dist``) from the same port, so the page is a browser tab and
nothing else. A mission's reasoner and segmenter come from the environment as
the bench's do (``SEARCH_REASONER_URL``, ``SAM_SERVER_URL``).

On CityNav the seat flies a row of the split through the bench's own runner
(``station.bench_seat``); on Cal City it flies the kinematic twin
(``benchmarks.flight``). Either lands in ``<out>/<UTC stamp>/`` as a run root:
a ledger ``episodes.jsonl`` and one attempt folder. The review reads run
roots, so a root written by ``python -m benchmarks.citynav.bench_fullset``,
by ``python -m benchmarks.flight.twin``, or copied in from another machine
reviews the same way. ``--out`` names one root or a directory of them.

A bench has many maps (CityNav: one per block). Every placement, the basemap
and the prior follow the map a request names (``station.placement``).

ponytail: one mission in the seat per process; a second seat is a second
process on another port.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib
import json
import logging
import math
import mimetypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from benchmarks.bench import Bench, missing_bake
from benchmarks.flight import map_render
from benchmarks.flight.live_source import open_live
from benchmarks.flight.mission import (
    STATE_AWAITING,
    STATE_ENDED,
    STATE_FAILED,
    STATE_FLYING,
    STATE_IDLE,
    Mission,
    resolve_episode,
)
from benchmarks.flight.tile_georef import map_bounds
from benchmarks.flight.twin import Twin
from benchmarks.run_record import rows
from benchmarks.run_record.config import load_config
from benchmarks.run_record.episode_workdir import (
    CONVERSATION_JSONL,
    FRAMES_JSONL,
    TRACK_JSONL,
    open_episode_dir,
)
from benchmarks.run_record.episodes import EPISODES_JSONL, STATUS_INFRA, TOUR_CURSOR
from benchmarks.run_record.rows import iter_json_rows
from maps.llm import endpoint_spec, model_cards
from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.settings import from_environ
from station.bench_seat import END_WAIT_S, BenchMission, with_words
from station.placement import OSM_CACHE_DIR, Placement, placement

log = logging.getLogger("station")

WEB_DIST = Path(__file__).parent / "web" / "dist"
#: The zoom range the basemap is served over: the whole block down to about 0.1 m a pixel.
TILE_ZOOMS = (10, 20)
TILE_PX = 256
#: How often the event stream looks for a change, and how often it says it is alive.
STREAM_PERIOD_S, STREAM_KEEPALIVE_S = 0.5, 15.0
#: The live trail's length, in ownship reads (one a second).
TRAIL_POINTS = 3600
#: The statuses in which a mission holds the seat, the twin's and the bench's alike.
LIVE = (STATE_FLYING, STATE_AWAITING)
#: What marks the id of a mission flown under the operator's own words, with no row in the split.
OPERATOR = "::operator-"
#: What a map's name is made of. A name is a file stem under the bake and the raster directory, so nothing else is one.
MAP_NAME = re.compile(r"[A-Za-z0-9_-]+")
#: How many maps' rasters the station holds open at once, and how many cut tiles it keeps.
BASEMAPS_HELD, TILES_HELD = 3, 2048
BENCHES = ("citynav", "calcity")


def bench_named(name: str, environ) -> Bench:
    """The bench ``name`` names, as ``environ`` describes it. By name: the station holds a bench as a value."""
    return getattr(importlib.import_module(f"benchmarks.{name}.paths"), f"{name}_bench")(environ)


_PATHS = frozenset(
    {"footway", "cycleway", "path", "service", "track", "steps", "pedestrian", "bridleway"}
)
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
    features = json.loads((Path(tiles_dir) / f"{map_name}_features.json").read_text())["prior"]
    out = [
        [g, [[round(float(x), 1), round(float(y), 1)] for x, y in f["world_contour"]]]
        for f in features
        if (g := prior_group(f)) is not None
    ]
    return json.dumps(out, separators=(",", ":")).encode()


def tile_lonlat(z: int, x: float, y: float) -> tuple[float, float]:
    """``(lon, lat)`` of a point in XYZ tile space (web mercator)."""
    n = 2**z
    return x / n * 360.0 - 180.0, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


def tile_png(ortho, to_local, z: int, x: int, y: int) -> bytes | None:
    """One 256 px web-mercator tile cut from the raster, transparent off its
    edge; None where the tile holds none of it. The tile's four corners are
    placed in the raster (``to_local``, the map's placement) and the quad
    between them warped: over one tile the mercator grid and the map's own
    grid differ by less than a pixel."""
    import cv2

    rgb, bounds, ppm = ortho
    rows_, cols = rgb.shape[:2]
    quad = []
    for cx, cy in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)):
        local = to_local(*tile_lonlat(z, cx, cy))
        if local is None:
            return None
        quad.append([(local[0] - bounds.x_min) * ppm, rows_ - (local[1] - bounds.y_min) * ppm])
    src = np.float32(quad)
    c0, r0 = (int(v) for v in np.floor(src.min(0)))
    c1, r1 = (int(v) for v in np.ceil(src.max(0)))
    # A tile far wider than 256 px of raster is read every ``step``-th pixel.
    step = max(1, min(c1 - c0, r1 - r0) // (2 * TILE_PX))
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, cols), min(r1, rows_)
    if c1 <= c0 or r1 <= r0:
        return None
    crop = np.ascontiguousarray(rgb[r0:r1:step, c0:c1:step, :3])
    warp = cv2.getPerspectiveTransform(
        (src - np.float32([c0, r0])) / step,
        np.float32([[0, 0], [TILE_PX, 0], [TILE_PX, TILE_PX], [0, TILE_PX]]),
    )
    size = (TILE_PX, TILE_PX)
    out = cv2.warpPerspective(crop, warp, size, flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(
        np.full(crop.shape[:2], 255, np.uint8), warp, size, flags=cv2.INTER_NEAREST
    )
    ok, png = cv2.imencode(".png", np.dstack([out[..., ::-1], alpha]))
    return png.tobytes() if ok else None


def _started(row: dict) -> float | None:
    """When a run started, unix seconds: the row's own ``ts``, else the stamp on its attempt folder."""
    if row.get("ts") is not None:
        return row["ts"]
    stamp = rows.attempt_stamp(row)
    return dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp() if stamp else None


class Station:
    """One bench's catalog, its maps, the seat, and the finished runs under the output root."""

    def __init__(
        self,
        *,
        bench: Bench,
        config: str,
        out: Path,
        ortho_dir: Path | None = None,
        register: Path | None = None,
        basemap: Path | None = None,
        osm_cache_dir: Path = OSM_CACHE_DIR,
    ) -> None:
        #: The bench whose catalog the station lists and whose engine its missions fly.
        self.bench = bench
        self.config, self.out, self.register = config, Path(out), register
        self.split, self.tiles_dir, self.osm_cache_dir = (
            bench.default_split,
            Path(bench.tiles_dir),
            Path(osm_cache_dir),
        )
        #: The twin's raster directory (Cal City); None on a station that flies no twin.
        self.ortho_dir = ortho_dir
        #: Where each map's raster pair lies: the twin's directory, else the bench's own.
        self.raster_dir = Path(ortho_dir) if ortho_dir is not None else Path(bench.rgbd_dir)
        #: ``--basemap``: the default map's north-up picture, or a directory of ``<map>.jpg|.jpeg|.png``.
        self._basemap_pictures = basemap
        self._catalogs: dict[str, tuple[list, dict]] = {}
        self._targets: dict[tuple[str, str], list | None] = {}
        self._ledgers: dict[Path, tuple[tuple, list]] = {}
        #: The map a request that names none is answered for: the default split's first row's.
        catalog = self.catalog(self.split)[0]
        self.map_name: str | None = catalog[0].map_name if catalog else None
        if (
            ortho_dir is not None
            and self.map_name
            and map_render.load_raster(ortho_dir, self.map_name) is None
        ):
            raise SystemExit(
                f"no {self.map_name} raster under {ortho_dir}: a raster directory holds "
                f"{map_render.raster_pair(self.map_name)}"
            )
        #: Each map's basemap, loaded when first asked for; the least recently used is let go.
        self.basemap = functools.lru_cache(maxsize=BASEMAPS_HELD)(self._basemap)
        self.tile = functools.lru_cache(maxsize=TILES_HELD)(self._tile)
        self._lock = threading.Lock()
        self.mission: Any = None
        self._sortie: Path | None = None
        self._trail: list[list[float]] = []

    # -- the bench's catalog ---------------------------------------------------

    def catalog(self, split: str) -> tuple[list, dict]:
        """``(episodes, by id)`` of a split, read once; empty where the split's file is not on disk."""
        if split not in self.bench.split_files:
            raise HTTPException(404, f"{self.bench.name} has no split {split!r}")
        if split not in self._catalogs:
            engine, path, episodes = self.bench.engine, self.bench.split_path(split), []
            for i, row in enumerate(engine.load_split(path) if path.is_file() else []):
                try:
                    episodes.append(engine.parse_episode(row, row_index=i))
                except (KeyError, ValueError, TypeError) as exc:
                    log.warning("%s row %d does not parse: %s", split, i, exc)
            self._catalogs[split] = (episodes, {e.episode_id: e for e in episodes})
        return self._catalogs[split]

    @functools.cached_property
    def objects(self) -> dict:
        """The bench's object catalog by map, or nothing where it is not on disk."""
        path = self.bench.objects_json
        return self.bench.engine.load_objects(path) if path.is_file() else {}

    def why_not_flying(self) -> str | None:
        """Why this station flies nothing, or None when it can."""
        if self.bench.name == "calcity":
            return (
                None
                if self.ortho_dir is not None
                else "this station reviews runs only: start it with --ortho <raster dir> to fly the twin"
            )
        try:
            self.bench.assert_available()
        except (FileNotFoundError, RuntimeError) as exc:
            return f"this station reviews runs only: {str(exc).splitlines()[0]}"
        return None

    def shown_split(self) -> str:
        """The split the mission list opens on: the newest run's under the output root, so its
        runs are in view, else the bench's default."""
        return next((r["split"] for r in self.runs() if r.get("split")), self.split)

    def about(self) -> dict[str, Any]:
        """What the console asks once: the bench, its splits, and what the seat can do."""
        held = [s for s in self.bench.split_files if self.bench.split_path(s).is_file()]
        flown = sorted({r["split"] for r in self.runs() if r.get("split")} - set(held))
        why = self.why_not_flying()
        return {
            "bench": self.bench.name,
            "split": self.shown_split(),
            "splits": held + flown,
            "map": self.map_name,
            "flies": why is None,
            "why": why,
            # The twin flies the operator's own words from the map's centre; a bench episode is a row of its split.
            "own_words": self.bench.name == "calcity",
        }

    # -- placing things on a map ---------------------------------------------

    def place(self, map_name: str | None) -> Placement | None:
        if not map_name or not MAP_NAME.fullmatch(map_name):
            return None
        return placement(str(self.tiles_dir), str(self.osm_cache_dir), map_name)

    def lonlat(self, map_name: str | None, xy) -> list[float] | None:
        at = self.place(map_name)
        return (
            list(at.to_lonlat(float(xy[0]), float(xy[1])))
            if at is not None and xy is not None
            else None
        )

    def _placed(self, map_name: str | None, doc: dict[str, Any], xy) -> dict[str, Any]:
        """``doc`` with ``lon``/``lat`` of local ``xy``, where the map is placed."""
        ll = self.lonlat(map_name, xy)
        if ll is not None:
            doc["lon"], doc["lat"] = ll
        return doc

    def target(self, split: str | None, episode_id: str | None) -> tuple[list | None, list | None]:
        """``(footprint ring, centre point)`` of an episode's target in WGS84, each None where it draws none.
        The footprint is the row's own mask, else its linked catalog object's contour."""
        e = self.catalog(split)[1].get(episode_id) if split in self.bench.split_files else None
        if e is None:
            return None, None
        key = (split, episode_id)
        if key not in self._targets:
            ids = set(e.target_object_ids)
            shape = e.target_mask or next(
                (o.contour for o in self.objects.get(e.map_name) or [] if o.id in ids), None
            )
            ring = [self.lonlat(e.map_name, p) for p in shape or ()]
            self._targets[key] = ring if ring and all(ring) else None
        return self._targets[key], self.lonlat(e.map_name, e.target_xy)

    def entry(self, split: str, e) -> dict[str, Any]:
        return {
            "episode_id": e.episode_id,
            "instruction": e.description,
            "map": e.map_name,
            "target_lonlat": self.target(split, e.episode_id)[0],
        }

    def _picture(self, map_name: str):
        """``--basemap``'s picture of ``map_name`` as ``(rgb, bounds, px_per_m)`` over the bake's bounds, or None."""
        from PIL import Image

        arg = self._basemap_pictures
        if arg is None or not (self.tiles_dir / f"{map_name}_meta.json").is_file():
            return None
        if Path(arg).is_dir():
            path = next(
                (
                    p
                    for ext in (".jpg", ".jpeg", ".png")
                    if (p := Path(arg) / f"{map_name}{ext}").is_file()
                ),
                None,
            )
        else:
            path = Path(arg) if map_name == self.map_name else None
        if path is None:
            return None
        Image.MAX_IMAGE_PIXELS = None  # a basemap of a whole block is large by design
        rgb = np.asarray(Image.open(path).convert("RGB"))
        bounds = map_bounds(self.tiles_dir, map_name)
        return rgb, bounds, rgb.shape[1] / (bounds.x_max - bounds.x_min)

    def _basemap(self, map_name: str):
        """What a map's tiles are cut from, ``(rgb, bounds, px_per_m)``: its raster pair, else
        ``--basemap``'s north-up picture of the bake's whole extent; None for a map that is not
        placed or has neither."""
        if self.place(map_name) is None:
            return None
        try:
            return map_render.load_raster(self.raster_dir, map_name) or self._picture(map_name)
        except Exception:
            log.exception("no basemap for %s", map_name)
            return None

    def _tile(self, map_name: str, z: int, x: int, y: int) -> bytes | None:
        ortho, at = self.basemap(map_name), self.place(map_name)
        return (
            tile_png(ortho, at.to_local, z, x, y) if ortho is not None and at is not None else None
        )

    # -- a run as the console's state document ------------------------------

    def _observations(self, workdir: Path, targets: list[str | None]) -> list[dict[str, Any]]:
        out = []
        for f in iter_json_rows(workdir / FRAMES_JSONL):
            if not f.get("observation"):
                continue
            n = int(f.get("flight") or 0)
            mosaic, window = f.get("mosaic") or {}, f.get("world_window")
            out.append(
                {
                    "sha": f["observation"],
                    "xy": f.get("pose_xy") or [0.0, 0.0],
                    "flight": n,
                    "target": (targets[n] if n < len(targets) else None) or f"flight {n + 1}",
                    "looks": int(mosaic.get("asked") or 1),
                    "placed": int(mosaic.get("placed") or 0),
                    "ground_m2": (window[2] - window[0]) * (window[3] - window[1])
                    if window
                    else 0.0,
                    "world_window": window,
                    "shape": f.get("shape"),
                    "t_s": f.get("t_s") or 0.0,
                }
            )
        return out

    @staticmethod
    def _reported(workdir: Path, transcript: list[dict[str, Any]]) -> str | None:
        """The model's reports, read off the wire record (``flight/llm/transcript.jsonl``: each
        answer with no tool call) onto the transcript: every report but the last rides its
        ``mission_end`` row as ``said``, and the last is returned as the mission's report."""
        replies = []
        for row in iter_json_rows(workdir / "flight" / "llm" / "transcript.jsonl"):
            message = ((row.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
            if (
                row.get("kind") == "response"
                and not message.get("tool_calls")
                and message.get("content")
            ):
                replies.append(str(message["content"]))
        ends = [
            i
            for i, r in enumerate(transcript)
            if r.get("kind") == "mission_end" and r.get("ended_by") == "reply"
        ]
        if len(ends) != len(replies) or not replies:
            return None  # a turn in flight, or a record that does not pair: no report is claimed
        for i, text in zip(ends[:-1], replies[:-1]):
            transcript[i] = {**transcript[i], "said": text}
        return replies[-1]

    def _flights(
        self, map_name: str | None, points, targets: list[str | None]
    ) -> list[dict[str, Any]]:
        return [
            self._placed(
                map_name,
                {
                    "step_idx": i,
                    "target": targets[i] if i < len(targets) else None,
                    "resolved_xy": [float(xy[0]), float(xy[1])],
                },
                xy,
            )
            for i, xy in enumerate(points)
        ]

    @staticmethod
    def _summary(root_name: str, row: dict[str, Any]) -> dict[str, Any]:
        """A ledger row as the run list shows it: the console's ``MissionState`` off the row alone, no file read."""
        prov, m = row.get("provenance") or {}, rows.metrics(row)
        eid = row.get("episode_id") or ""
        return {
            "version": 0,
            "path": f"{root_name}/{row['workdir']}",
            "episode_id": eid,
            "split": row.get("split") or prov.get("split"),
            "kind": "replay",
            "run_no": row.get("run_no"),
            "status": STATE_FAILED if row.get("status") == STATUS_INFRA else STATE_ENDED,
            "instruction": row.get("description") or "",
            "ended_by": rows.ended_by(row),
            "error": row.get("error"),
            "reply_text": None,
            "turns_used": row.get("turns_used") or (m.get("llm") or {}).get("turns") or 0,
            # A bench row does not count its flights: the review counts the stops it flew.
            "flights_flown": row.get("flights_flown"),
            "started_at": _started(row),
            "git_commit": prov.get("git_commit"),
            "git_dirty": prov.get("git_dirty"),
            "map_name": row.get("map_name") or eid.partition("::")[0] or None,
            "target_lonlat": None,
            "target_point": None,
            "commit_lonlat": None,
            "transcript": [],
            "flights": [],
            "observations": [],
            "leg": None,
            "score": row.get("metrics"),
            "duration_s": None,
        }

    def run_state(self, root: Path, row: dict[str, Any]) -> dict[str, Any]:
        """A ledger row and its attempt folder as the console's ``MissionState``."""
        doc = self._summary("." if root == self.out else root.name, row)
        workdir, where = root / row["workdir"], doc["map_name"]
        transcript = list(iter_json_rows(workdir / CONVERSATION_JSONL))
        if "flights_flown" in row:
            # A flight mission's row: its start, then each flight it is scored on.
            flights = self._flights(where, (row.get("visited_xy") or [])[1:], [])
        else:
            # A bench row's visited_xy is the six-action replay; the stops it flew are the tour cursor's.
            try:
                stops = json.loads((workdir / TOUR_CURSOR).read_text())["stops_visited"]
            except (OSError, ValueError, KeyError):
                stops = []
            flights = self._flights(
                where, [s["xy"] for s in stops], [s.get("stop_id") for s in stops]
            )
        ring, point = self.target(doc["split"], doc["episode_id"])
        return {
            **doc,
            "reply_text": self._reported(workdir, transcript),
            "transcript": transcript,
            "flights": flights,
            "flights_flown": row.get("flights_flown", len(flights)),
            "observations": self._observations(workdir, []),
            "target_lonlat": ring,
            "target_point": point or self.lonlat(where, row.get("target_xy")),
            "commit_lonlat": self.lonlat(where, row.get("commit_xy")),
        }

    def seat_state(self) -> dict[str, Any] | None:
        """The mission in the seat as the console's ``MissionState``, or None with the seat empty."""
        m = self.mission
        if m is None:
            return None
        kind = getattr(m, "kind", "twin")
        row = m.row if kind == "bench" else None  # the scored row of a bench episode that has ended
        where = m.map_name or self.map_name
        flights = list(m.flights)
        targets = [str(f.get("target") or f.get("target_fid") or "") or None for f in flights]
        transcript = list(m.transcript)
        observations = self._observations(m.workdir, targets)
        report = (
            self._reported(m.workdir, transcript)
            if m.status in (STATE_AWAITING, STATE_ENDED)
            else None
        )
        ring, point = self.target(m.split, m.episode_id)
        return {
            "version": hash(
                (
                    m.status,
                    len(transcript),
                    len(flights),
                    len(observations),
                    m.error,
                    row is not None,
                    json.dumps(m.waiting, default=str),
                )
            ),
            "path": f"{m.workdir.parent.name}/{m.workdir.name}",
            "episode_id": m.episode_id,
            "split": m.split,
            "kind": kind,
            "run_no": m.run_no,
            "status": "ready" if m.status == STATE_IDLE else m.status,
            "instruction": m.instruction,
            "ended_by": m.ended_by,
            "error": m.error,
            "reply_text": report,
            "turns_used": m.turns(),
            "flights_flown": len(flights),
            "started_at": m.started_at,
            "git_commit": m.git_commit,
            "git_dirty": m.git_dirty,
            "map_name": where,
            "target_lonlat": ring,
            "target_point": point,
            "commit_lonlat": self.lonlat(where, (row or {}).get("commit_xy")),
            "transcript": transcript,
            "flights": self._flights(
                where, [f["resolved_xy"] for f in flights if f.get("resolved_xy")], targets
            ),
            "observations": observations,
            "score": getattr(m, "score", None),
            "leg": (
                {"closest_m": 0.0, "short_passes": 0, "packageVersion": "", **m.waiting}
                if m.waiting is not None
                else None
            ),
        }

    def roots(self) -> list[Path]:
        """The run roots under the output root, newest name first: each folder holding a ledger, the root itself included."""
        found = sorted((p.parent for p in self.out.glob(f"*/{EPISODES_JSONL}")), reverse=True)
        return found + ([self.out] if (self.out / EPISODES_JSONL).is_file() else [])

    def runs(self) -> list[dict[str, Any]]:
        """Every finished run under the output root, newest first; the one in the seat is the seat's.
        A ledger is read again only when it has changed."""
        seat = self._sortie if self.mission is not None and self.mission.status in LIVE else None
        out = []
        for root in self.roots():
            if root == seat:
                continue
            ledger = root / EPISODES_JSONL
            stat = ledger.stat()
            key = (stat.st_mtime_ns, stat.st_size)
            if self._ledgers.get(root, (None,))[0] != key:
                name = "." if root == self.out else root.name
                self._ledgers[root] = (
                    key,
                    [
                        self._summary(name, r)
                        for r in reversed(list(iter_json_rows(ledger)))
                        if r.get("workdir")
                    ],
                )
            out += self._ledgers[root][1]
        return out

    def run_at(self, path: str) -> tuple[Path, dict[str, Any]]:
        """The run root and ledger row a ``<root>/<workdir>`` path names (``.`` the output root
        itself); 404 for a path that names none, or that leaves the output root."""
        root_name, _, workdir = path.partition("/")
        root = self.out / root_name
        if (
            not root_name
            or ".." in Path(path).parts
            or not root.resolve().is_relative_to(self.out.resolve())
            or not (root / EPISODES_JSONL).is_file()
        ):
            raise HTTPException(404, f"no run at {path!r}")
        found = [r for r in iter_json_rows(root / EPISODES_JSONL) if r.get("workdir")]
        row = next(
            (r for r in reversed(found) if r["workdir"] == workdir),
            found[-1] if found and not workdir else None,
        )
        if row is None:
            raise HTTPException(404, f"no run at {path!r}")
        return self.out if root_name == "." else root, row

    def track(self, root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
        """The run's flown path off the root's ``flight_track.jsonl``, mission seconds from its start;
        empty for a run that recorded none (a bench run: its aircraft was placed, not flown)."""
        t0, length = row.get("started_at_utc"), row.get("elapsed_s")
        if t0 is None:
            return []
        where, out = row.get("map_name"), []
        for fix in iter_json_rows(root / TRACK_JSONL):
            t = float(fix["t_unix"]) - float(t0)
            if -1e-6 <= t <= (float(length) if length is not None else math.inf) + 1.0:
                out.append(
                    self._placed(
                        where,
                        {
                            "t_s": round(t, 2),
                            "x": fix["x"],
                            "y": fix["y"],
                            "track_deg": fix.get("track_deg") or 0.0,
                            "agl_m": fix.get("alt_agl_m") or 0.0,
                            "speed_mps": fix.get("ground_speed_mps") or 0.0,
                        },
                        (fix["x"], fix["y"]),
                    )
                )
        return out

    def replay(self, path: str) -> dict[str, Any]:
        """One run under review: its state document, the path it flew, and the scrubber's length.
        A run with a recorded track scrubs along it; one without draws ``visited_xy`` as its path,
        one sample a pose in order, and has nothing to scrub (``duration_s`` None)."""
        root, row = self.run_at(path)
        doc, track = self.run_state(root, row), self.track(root, row)
        if track:
            return {**doc, "track": track, "duration_s": row.get("elapsed_s")}
        where = doc["map_name"]
        return {
            **doc,
            "duration_s": None,
            "track": [
                self._placed(
                    where,
                    {
                        "t_s": float(i),
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "track_deg": 0.0,
                        "agl_m": 0.0,
                    },
                    p,
                )
                for i, p in enumerate(row.get("visited_xy") or [])
            ],
        }

    # -- the aircraft --------------------------------------------------------

    def ownship(self, path: str | None, t: float) -> dict[str, Any] | None:
        """Where the aircraft is: the seat's pose now, or a reviewed run's at mission second ``t``
        (None for a run with no recorded track)."""
        if path:
            root, row = self.run_at(path)
            track = self.track(root, row)
            if not track:
                return None
            at = min(track, key=lambda p: abs(p["t_s"] - t))
            return {
                **at,
                "map": row.get("map_name"),
                "duration_s": row.get("elapsed_s"),
                "live": False,
            }
        m = self.mission
        if m is None or m.status not in LIVE:
            return None
        if getattr(m, "kind", "twin") == "bench":
            # A bench flight is a placement: the aircraft stands at its last stop, its trail the stops in order.
            pose = m.pose()
            if pose is None:
                return None
            stops = [m.episode.start_xy, *(f["resolved_xy"] for f in m.flights)]
            return {
                **self._placed(
                    m.map_name,
                    {
                        "x": pose[0],
                        "y": pose[1],
                        "track_deg": 0.0,
                        "agl_m": pose[2],
                        "speed_mps": 0.0,
                        "map": m.map_name,
                        "t_s": m.elapsed_s(),
                        "live": True,
                    },
                    pose,
                ),
                "trail": [ll for p in stops if (ll := self.lonlat(m.map_name, p)) is not None],
            }
        pose, own = m.live.pose_now(), m.live.ownship()
        if pose is None or own is None:
            return None
        doc = self._placed(
            self.map_name,
            {
                "x": own["x"],
                "y": own["y"],
                "track_deg": pose.track_deg,
                "agl_m": own["agl_m"],
                "speed_mps": pose.speed_mps,
                "map": self.map_name,
                "t_s": m.elapsed_s(),
                "live": True,
            },
            (own["x"], own["y"]),
        )
        if "lon" in doc and (not self._trail or self._trail[-1] != [doc["lon"], doc["lat"]]):
            self._trail = (self._trail + [[doc["lon"], doc["lat"]]])[-TRAIL_POINTS:]
        return {**doc, "trail": list(self._trail)}

    # -- the seat ------------------------------------------------------------

    def start(
        self, episode_id: str | None, instruction: str, effort: str | None, split: str | None = None
    ) -> dict[str, Any]:
        """Fly a mission from the seat: on Cal City a catalog row, or the operator's own words from
        the map's centre, on a twin of its own; on CityNav a row of ``split`` through the bench's runner."""
        if (why := self.why_not_flying()) is not None:
            raise HTTPException(409, why)
        with self._lock:
            if self.mission is not None and self.mission.status in LIVE:
                raise HTTPException(
                    409, f"mission {self.mission.episode_id} is flying; stop it first"
                )
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            if self.bench.name == "calcity":
                mission = self._twin_mission(episode_id, instruction, effort, stamp)
            else:
                mission = self._bench_mission(
                    episode_id, instruction, effort, split or self.split, stamp
                )
            self.mission, self._trail = mission, []
            mission.start()
        return self.seat_state()

    def _bench_mission(
        self, episode_id: str | None, instruction: str, effort: str | None, split: str, stamp: str
    ) -> BenchMission:
        if episode_id is None:
            raise HTTPException(
                400, f"a {self.bench.name} mission is a row of its split: pick one, then send"
            )
        episode = self.catalog(split)[1].get(episode_id)
        if episode is None:
            raise HTTPException(404, f"{split} holds no episode {episode_id}")
        if (why := missing_bake(self.bench.tile_stem(episode.map_name))) is not None:
            raise HTTPException(409, why)
        if not map_render.holds_raster(self.bench.rgbd_dir, episode.map_name):
            raise HTTPException(
                409,
                f"no {episode.map_name} raster under {self.bench.rgbd_dir}: "
                f"the flights of this episode have nothing to see",
            )
        self._sortie = self.out / stamp
        ep = open_episode_dir(self._sortie, episode_id=episode_id, benchmark=self.bench.name)
        return BenchMission(
            bench=self.bench,
            episode=with_words(episode, instruction),
            split=split,
            config=self.config,
            objects_by_map=self.objects,
            sortie=self._sortie,
            ep=ep,
            reasoning_effort=effort,
        )

    def _twin_mission(
        self, episode_id: str | None, instruction: str, effort: str | None, stamp: str
    ) -> Mission:
        if episode_id is not None and episode_id not in self.catalog(self.split)[1]:
            if OPERATOR not in episode_id:
                raise HTTPException(404, f"{self.split} holds no episode {episode_id}")
            episode_id = None  # an operator's past tasking, picked again: its words fly as a mission of their own
        start_xy = None
        if episode_id is None:
            b = map_bounds(self.tiles_dir, self.map_name)
            start_xy = ((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2)
            episode_id = f"{self.map_name}{OPERATOR}{time.strftime('%H%M%S', time.gmtime())}"
        cfg = load_config(self.config)
        sensor = resolve_sensor(cfg.sensor)
        episode = resolve_episode(self.bench, episode_id, self.split, start_xy)
        start_alt, ground = self.bench.engine.start_altitude_m(cfg, self.map_name, episode.start_z)
        twin = Twin(
            tiles_dir=self.tiles_dir,
            map_name=self.map_name,
            ortho=map_render.load_raster(self.ortho_dir, self.map_name),
            sensor=sensor,
            start_xy=episode.start_xy,
            agl_m=start_alt - ground,
        )
        live = open_live(
            vehicle=twin,
            tiles_dir=self.tiles_dir,
            map_name=self.map_name,
            sensor=sensor,
            register_raster=self.register,
        )
        self._sortie = self.out / stamp
        ep = open_episode_dir(
            self._sortie, episode_id=episode_id, benchmark=self.bench.name, run_no=1
        )
        mission = Mission(
            bench=self.bench,
            episode_id=episode_id,
            split=self.split,
            config=self.config,
            workdir=ep.dir,
            live=live,
            instruction=instruction.strip() or None,
            recorder=ep.recorder,
            close_log=ep.close,
            run_id=stamp,
            run_no=1,
            start_xy=start_xy,
            reasoning_effort=effort,
            agl_m=start_alt - ground,
        )
        # The live source closes when the mission's thread ends, however it ends.
        threading.Thread(
            target=lambda: (mission.wait(), live.close()), name="station-close", daemon=True
        ).start()
        return mission

    def clear(self) -> None:
        """Stop the mission in the seat and empty it; its run stays under the output root. A bench
        episode is scored as it ends, so its row is waited for."""
        with self._lock:
            mission, self.mission, self._trail = self.mission, None, []
        if mission is not None:
            mission.stop()
            if isinstance(mission, BenchMission):
                mission.wait(END_WAIT_S)


class _Next(BaseModel):
    episode_id: str | None = None
    instruction: str = ""
    reasoning_effort: str | None = None
    split: str | None = None


class _Tasking(BaseModel):
    text: str


def build_app(station: Station, web: Path | None = WEB_DIST) -> FastAPI:
    app = FastAPI(title="MAPS station")

    def the_map(name: str | None) -> str:
        """The map a request names (the station's default when it names none); 404 for a name that is no map's."""
        name = name or station.map_name
        if not name or not MAP_NAME.fullmatch(name):
            raise HTTPException(404, f"no map {name!r}")
        return name

    @app.get("/station")
    def about() -> dict[str, Any]:
        return station.about()

    @app.get("/missions")
    def missions(
        q: str | None = None, limit: int = 200, split: str | None = None
    ) -> dict[str, Any]:
        split = split or station.shown_split()
        catalog, by_id = station.catalog(split)
        words = (q or "").lower().split()
        hits = [
            e
            for e in catalog
            if all(w in e.description.lower() or w in e.episode_id.lower() for w in words)
        ]
        # A run with no row in the split shown (flown under the operator's own words, or reviewed on a
        # station without the split's file) lists first, by its words.
        own = {
            r["episode_id"]: {
                "episode_id": r["episode_id"],
                "instruction": r["instruction"],
                "map": r["map_name"],
                "target_lonlat": None,
            }
            for r in reversed(station.runs())
            if r["episode_id"] not in by_id and r["split"] in (None, split)
        }
        mine = [
            m
            for m in own.values()
            if all(w in m["instruction"].lower() or w in m["episode_id"].lower() for w in words)
        ]
        # A row with a run under the output root lists ahead of the rest, so a reviewing station opens on its runs.
        flown = {r["episode_id"] for r in station.runs()}
        hits.sort(key=lambda e: e.episode_id not in flown)
        return {
            "split": split,
            "total": len(catalog) + len(own),
            "missions": (mine + [station.entry(split, e) for e in hits])[: max(0, limit)],
        }

    @app.get("/reasoner")
    def reasoner() -> dict[str, Any]:
        """The served model and the reasoning efforts its card takes: the seat's serve, else
        the one discovery finds; ``model`` is None while no reasoner answers. A station that
        flies nothing answers at once: the console asks on every load."""
        seat = station.mission
        if seat is None and (why := station.why_not_flying()) is not None:
            return {"model": None, "efforts": [], "error": why}
        try:
            model = (
                (seat.bench.endpoint_spec if seat is not None else None)
                or endpoint_spec.discover_from_settings(
                    from_environ(station.bench.environ), probe_sam=False
                )
            ).model_id
        except Exception as exc:
            return {"model": None, "efforts": [], "error": str(exc)}
        return {
            "model": model,
            "efforts": list(model_cards.card_for(model).get("efforts", ())),
            "error": None,
        }

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

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    @app.post("/mission/next")
    def mission_next(req: _Next) -> dict[str, Any]:
        if not req.instruction.strip() and req.episode_id is None:
            raise HTTPException(400, "a mission needs a catalog row or the operator's words")
        return station.start(req.episode_id, req.instruction, req.reasoning_effort, req.split)

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
        return station.replay(path)

    @app.get("/mission/ownship")
    def mission_ownship(path: str | None = None, t: float = 0.0) -> dict[str, Any] | None:
        return station.ownship(path, t)

    @app.get("/mission/prior.json")
    def mission_prior(map: str | None = None) -> Response:
        name = the_map(map)
        if not (station.tiles_dir / f"{name}_features.json").is_file():
            raise HTTPException(404, f"no features for {name!r}")
        return Response(
            prior_json(str(station.tiles_dir), name),
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    def bbox(name: str, b) -> list[float]:
        (w, s), (e, n) = (
            station.lonlat(name, (b.x_min, b.y_min)),
            station.lonlat(name, (b.x_max, b.y_max)),
        )
        return [w, s, e, n]

    @app.get("/area")
    def area(map: str | None = None) -> dict[str, Any]:
        """A map's extent, off its bake; 404 for a map with no bake or no placement."""
        name = the_map(map)
        if station.place(name) is None or not (station.tiles_dir / f"{name}_meta.json").is_file():
            raise HTTPException(
                404, f"{name} has no bake, or no georeference: it is placed nowhere"
            )
        return {"map": name, "bbox": bbox(name, map_bounds(station.tiles_dir, name))}

    @app.get("/location")
    def location(map: str | None = None) -> dict[str, Any]:
        """A map's basemap: tiles over its raster's (or its picture's) extent; 404 for a map with neither."""
        name = the_map(map)
        basemap = station.basemap(name)
        if basemap is None:
            raise HTTPException(
                404, f"no raster or picture of {name}, or no georeference: it draws no basemap"
            )
        w, s, e, n = bounds = bbox(name, basemap[1])
        return {
            "name": name,
            "label": name,
            "bounds": bounds,
            "center": {"lat": (s + n) / 2, "lon": (w + e) / 2},
            "tiles": f"tiles/{name}/{{z}}/{{x}}/{{y}}.png",
            "minzoom": TILE_ZOOMS[0],
            "maxzoom": TILE_ZOOMS[1],
            "attribution": "",
        }

    @app.get("/tiles/{map}/{z}/{x}/{y}.png")
    def tiles(map: str, z: int, x: int, y: int) -> Response:
        png = station.tile(the_map(map), z, x, y) if TILE_ZOOMS[0] <= z <= TILE_ZOOMS[1] else None
        if png is None:
            raise HTTPException(404)
        return Response(
            png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
        )

    @app.get("/run/file")
    def run_file(path: str) -> FileResponse:
        """One file of a run, by ``<root>/<workdir>/<name>``; nothing outside the output root."""
        root = station.out.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404)
        return FileResponse(
            target, media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )

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
    ap.add_argument(
        "--bench",
        choices=BENCHES,
        default="citynav",
        help="the bench whose catalog, maps and runs the station serves (default citynav)",
    )
    ap.add_argument("--config", default=None, help="the run config (default configs/<bench>.yaml)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/station"),
        help="where runs land and are reviewed from: one run root, or a directory of them",
    )
    ap.add_argument(
        "--ortho",
        type=Path,
        default=None,
        help="Cal City: the raster directory the twin's frames and the basemap are cut from; "
        "unset reviews runs only",
    )
    ap.add_argument(
        "--register",
        type=Path,
        default=None,
        help="Cal City: the raster directory each frame is registered against; unset registers nothing",
    )
    ap.add_argument(
        "--basemap",
        type=Path,
        default=None,
        help="for a map with no raster: a north-up picture of its bake's whole extent, or a "
        "directory of them named <map>.jpg or <map>.png",
    )
    ap.add_argument(
        "--osm-cache",
        type=Path,
        default=OSM_CACHE_DIR,
        help="the OSM caches a CityNav block is placed by (default tiles/osm_cache)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s %(message)s"
    )

    import uvicorn

    args.out.mkdir(parents=True, exist_ok=True)
    station = Station(
        bench=bench_named(args.bench, os.environ),
        config=args.config or f"configs/{args.bench}.yaml",
        out=args.out,
        ortho_dir=args.ortho,
        register=args.register,
        basemap=args.basemap,
        osm_cache_dir=args.osm_cache,
    )
    if (why := station.why_not_flying()) is not None:
        log.info("%s", why)
    if not (WEB_DIST / "index.html").is_file():
        log.warning(
            "no built console at %s: run `npm ci && npm run build` in station/web", WEB_DIST
        )
    try:
        uvicorn.run(build_app(station), host=args.host, port=args.port, log_level="warning")
    finally:
        station.clear()  # a bench episode holding at its report is scored and its row landed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
