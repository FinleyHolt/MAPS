"""One flying mission, held open across operator turns.

The method half of the station that flew the C-12. It hydrates the search
stack for a mission exactly as the bench does, flies the conversation on a
worker thread, and keeps the session alive afterwards so the operator can
re-task what the model just reported.

Two things separate this from the bench, and both are the point:

* **The conversation does not end at the reply.** A bench episode's plain
  reply is its ending. In flight the reply is a REPORT, so the session parks
  in ``awaiting_tasking`` and the operator's answer goes to
  ``continue_mission`` on the SAME reasoner. Everything the earlier turns
  folded into the world stays.
* **The operator speaks whenever.** Taskings land in a queue the loop drains
  at the top of each turn (``retask``), so a bad survey is redirected without
  waiting for the model to finish.

A flight is a leg the aircraft flies (:meth:`Mission._fly_live`, the wrap of
``MissionSpec.fly``): it is flown for an AREA and has arrived when the
recording holds it (``LiveSource.coverage``), and its observation is the
frames that held it, fused (``LiveSource.observe_area``). Nothing else in the
engine's loop changes. Nothing here scores the mission while it flies: the
row is scored after the reply, against the split's target when the episode
has one.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.flight.live_source import OBSERVATION_JPEG_QUALITY, LiveSource
from benchmarks.flight.run_imagery import RunImagery
from benchmarks.flight.tile_georef import local_to_lonlat
from benchmarks.flight.vehicle import Leg
from maps.search.parse_compile.conversation import CONVERSATION_JSONL, ConversationLog

log = logging.getLogger("flight.mission")

#: How long a leg may fly before its arrival is given up, counted from the
#: aircraft leaving the ground. A C-12 pass is a 9 to 12 km lap at a 1.5 km
#: turn radius and the crew go round again on a miss; the cap ends a leg the
#: crew never fly.
ARRIVAL_TIMEOUT_S = 1800.0
#: A leg has arrived when the recording holds this share of the area it was
#: flown for. ponytail: a share rather than all of it, because a footprint's
#: rim cells sit under a frame's edge at the pass's longest lever and nine
#: tenths of a house is the house; the leg row states the share.
COVERED_FRACTION = 0.9
#: How many of a footprint's cells sample its area for the arrival rule.
AREA_POINTS = 48
#: The radius of the disc about a point with no footprint when the run's own
#: observation span is not known.
AREA_DISC_M = 60.0
#: How many times the recording is asked whether it holds the area before a
#: leg is posted anyway, and the pause between asks: a read that never
#: answered is not ground the pass never held. Bounded, because the aircraft
#: can always fly the leg.
COVERAGE_TRIES = 4
COVERAGE_RETRY_S = 3.0
#: A pass is SHORT when the aircraft came within the leg's reach of its point,
#: has since drawn this far past its closest approach and stayed there this
#: long, and the recording holds part of the area but less than
#: ``COVERED_FRACTION``: the crew flew the waypoint and the frame missed a side
#: of the ground. It is said in the transcript and asked again with the
#: waypoint moved toward the missed ground, ``SHORT_PASS_REQUESTS`` times
#: before the leg falls to the arrival timeout.
SHORT_PASS_LEAVE_M = 150.0
SHORT_PASS_LEAVE_S = 5.0
SHORT_PASS_REQUESTS = 2
#: Below this above the map's floor the aircraft is on the ground, and an
#: arrival wait does not spend its timeout.
AIRBORNE_AGL_M = 10.0
#: The most a leg waits for its point to leave the frame after arriving.
FRAME_WAIT_S = 30.0

#: One row per observation a mission perceived, in the episode's directory:
#: the key from the mission's reasoning back into the recording.
FRAMES_JSONL = "frames.jsonl"
#: Where each waypoint's observation goes: the pass fused aboard as one
#: north-up image, ``<sha>.jpg``. Never ``frames/``, which is what the model saw.
OBSERVATIONS_DIR = "observations"

STATE_IDLE = "idle"
STATE_FLYING = "flying"
STATE_AWAITING = "awaiting_tasking"
STATE_ENDED = "ended"
STATE_FAILED = "failed"


def authored_episode(episode_id: str, start_xy: tuple[float, float]):
    """The episode of a mission flown under the operator's own words: the map
    its id names, the start given, and no target, so nothing scores it."""
    from benchmarks.citynav.adapter import CityNavEpisode  # noqa: PLC0415

    return CityNavEpisode(
        map_name=episode_id.split("::")[0], episode_id=episode_id, description="",
        start_xy=(float(start_xy[0]), float(start_xy[1])), target_xy=None,
        target_object_ids=[], target_mask=None, marker_positions=[], raw={},
    )


@dataclass
class Hydrated:
    """The search stack for one mission, built the way the bench builds it."""

    cfg: Any
    state: Any
    assets: Any
    episode: Any
    sensor: Any
    tile_frame: Any
    perception: Any
    search_loop: Any
    compiler: Any


def resolve_episode(episode_id: str, split: str, start_xy: tuple[float, float] | None = None):
    """The split's row named ``episode_id``, or with ``start_xy`` an
    operator-authored mission with no row (:func:`authored_episode`)."""
    from benchmarks.citynav.adapter import load_split, parse_episode  # noqa: PLC0415
    from benchmarks.citynav.paths import split_path  # noqa: PLC0415

    if start_xy is not None:
        return authored_episode(episode_id, start_xy)
    rows = load_split(split_path(split))
    episode = next((ep for ep in (parse_episode(r, row_index=i) for i, r in enumerate(rows))
                    if ep.episode_id == episode_id), None)
    if episode is None:
        raise ValueError(f"{split} holds no episode {episode_id}")
    return episode


def hydrate(*, episode_id: str, split: str, config: str, workdir: Path,
            start_xy: tuple[float, float] | None = None) -> Hydrated:
    """Assets, perception and the belief engine for one mission, exactly as the bench builds them."""
    from benchmarks.citynav.episode import EpisodeState  # noqa: PLC0415
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR, tile_stem  # noqa: PLC0415
    from benchmarks.citynav.run_benchmark import load_config, load_episode_assets, load_episode_graph  # noqa: PLC0415
    from benchmarks.citynav.run_benchmark.config import assert_flight_config  # noqa: PLC0415
    from benchmarks.citynav.run_benchmark.episode_runner import start_altitude_m  # noqa: PLC0415
    from benchmarks.citynav.run_benchmark.perception import make_perception  # noqa: PLC0415
    from maps.perception.geometry.geolocate import TileFrame  # noqa: PLC0415
    from maps.perception.geometry.sensor_spec import resolve_sensor, served_description  # noqa: PLC0415
    from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider  # noqa: PLC0415
    from maps.search import MapsCompiler, SearchLoop, SearchLoopConfig  # noqa: PLC0415
    from maps.sws import GridDomain, WorldState  # noqa: PLC0415

    cfg = load_config(str(config))
    assert_flight_config(cfg)
    episode = resolve_episode(episode_id, split, start_xy)
    assets = load_episode_assets(episode.map_name, cfg.cell_size_m)

    perception = make_perception(cfg)
    if perception is not None and hasattr(perception, "reset"):
        perception.reset()
    sam_concept_setter = None
    cp = getattr(perception, "concept_provider", None)
    if isinstance(cp, ReasonerUnionConceptProvider):
        sam_concept_setter = cp.set_concepts

    try:
        graph = load_episode_graph(tile_stem(episode.map_name))
    except FileNotFoundError:
        from maps.sws.graph import build_object_graph  # noqa: PLC0415

        graph = build_object_graph(assets.base_features, cell_size_m=float(assets.cell_size_m))

    grid_h, grid_w = assets.grid_shape
    world_state = WorldState(
        grid_domain=GridDomain(height=grid_h, width=grid_w, cell_size_m=float(assets.cell_size_m)),
        object_graph=graph,
    )
    sensor = resolve_sensor(cfg.sensor) if cfg.sensor else None
    if sensor is None:
        raise RuntimeError(f"cfg.sensor did not resolve a SensorSpec: {cfg.sensor!r}")
    compiler = MapsCompiler(sensor_description=served_description(sensor, cfg.start_altitude_agl_m),
                            workdir=workdir)

    start_alt_m, ground = start_altitude_m(cfg, episode.map_name, episode.episode_id)
    search_loop = SearchLoop(
        SearchLoopConfig.from_env(),
        world_state,
        tile_origin_xy=(assets.bounds.x_min, assets.bounds.y_min),
        tile_name=f"citynav::{assets.map_name}",
        base_features=assets.base_features,
        compiler=compiler,
        sam_concept_setter=sam_concept_setter,
        # The block's ground datum: the mission's observation span is
        # ``2 * (alt - ground)`` off it, and without it every explore answers
        # "no footprint to cover".
        ortho_ground_z_m=ground,
        # The look tool's crop renderer is wired off this; in flight every
        # crop is cut from the run's own imagery instead (``look_imagery``).
        rgbd_dir=str(CITYNAV_NATIVE_RGBD_DIR),
    )
    search_loop.start(prebake_graph=graph, runtime_graph=graph)

    state = EpisodeState(pose_xy=episode.start_xy, altitude_m=start_alt_m, graph=graph)
    state.record_visit()

    tile_frame = TileFrame(x_min=float(assets.bounds.x_min), y_min=float(assets.bounds.y_min),
                           cell_size_m=float(assets.cell_size_m), grid_shape=tuple(assets.grid_shape))
    return Hydrated(cfg=cfg, state=state, assets=assets, episode=episode, sensor=sensor, tile_frame=tile_frame,
                    perception=perception, search_loop=search_loop, compiler=compiler)


def leg_span_m(h: Hydrated) -> float:
    """The run's observation span in metres: what one frame holds along its
    long side at the mission's height, or twice ``AREA_DISC_M`` when the height
    is not known."""
    from maps.perception.observation import observation_extent_m  # noqa: PLC0415

    alt = getattr(h.state, "altitude_m", None)
    ground = getattr(h.search_loop, "_ortho_ground_z_m", None)
    if alt is None or ground is None:
        return 2.0 * AREA_DISC_M
    return float(observation_extent_m(float(alt), float(ground), h.sensor))


def leg_area(h: Hydrated, token: str, xy) -> list[tuple[float, float]]:
    """The ground a leg is flown for, as points in tile metres: the target's
    own cells when the token names a row the bake gave a footprint (at most
    ``AREA_POINTS`` of them, evenly), else a disc of half the observation span
    about the point: a centre, six points at a quarter span, twelve at half."""
    cells = h.search_loop.node_cells_xy(token) if token else None
    # Three cells at least: fewer is a point or a line, which no coverage
    # fraction answers. A row that thin is flown as a cover stop.
    if cells and len(cells) >= 3:
        step = max(1, -(-len(cells) // AREA_POINTS))
        return [(float(c[0]), float(c[1])) for c in cells[::step]][:AREA_POINTS]
    r = leg_span_m(h) / 2.0
    x, y = float(xy[0]), float(xy[1])
    points = [(x, y)]
    for n, radius in ((6, r / 2.0), (12, r)):
        points += [(x + radius * math.cos(2.0 * math.pi * k / n), y + radius * math.sin(2.0 * math.pi * k / n))
                   for k in range(n)]
    return points


class _StampedLog(ConversationLog):
    """The conversation log, each row stamped with ``t_s``, seconds since the mission started."""

    def __init__(self, path: Path, elapsed) -> None:
        super().__init__(path)
        self._elapsed = elapsed

    def append(self, item) -> None:
        if isinstance(item, dict) and "t_s" not in item:
            item = {**item, "t_s": self._elapsed()}
        super().append(item)


class Mission:
    """A mission in the seat: its thread, its tasking queue, its record."""

    def __init__(self, *, episode_id: str, split: str, config: str, workdir: Path, live: LiveSource,
                 instruction: str | None = None, recorder: Any = None, close_log=None,
                 run_id: str | None = None, run_no: int | None = None,
                 start_xy: tuple[float, float] | None = None, reasoning_effort: str | None = None) -> None:
        self.episode_id = episode_id
        self.split = split
        self.config = config
        self.workdir = Path(workdir)
        self.instruction_override = instruction
        #: A string the serve accepts; None flies the arm's.
        self.reasoning_effort = reasoning_effort
        self.run_id = run_id
        self.run_no = run_no
        self._start_xy = start_xy
        self._live = live
        self._vehicle = live.vehicle
        self._recorder = recorder
        self.close_log = close_log or (lambda: None)

        self._lock = threading.Lock()
        self._taskings: queue.Queue[str] = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

        self.status = STATE_IDLE
        self.instruction = instruction or ""
        self.ended_by: str | None = None
        #: When the conversation ended (the reply), so the row's ``elapsed_s``
        #: is the mission's and not the park's.
        self.ended_at: float | None = None
        self.error: str | None = None
        self.reply_text: str | None = None
        #: Model turns of the conversations that have ENDED; :meth:`_turns` adds the one in flight.
        self.turns_used = 0
        #: How many reports the model has made.
        self.replies = 0
        self._turn_diag: dict[str, Any] = {}
        self.started_at: float | None = None
        #: The same instant on the receiver's clock (UTC, unix seconds), what a replay anchors on.
        self.started_at_utc: float | None = None
        #: Every row lands in ``<workdir>/conversation.jsonl`` as it happens.
        self.transcript: list[dict] = _StampedLog(self.workdir / CONVERSATION_JSONL, self.elapsed_s)
        self.flights: list[dict] = []
        self.map_name: str | None = None
        self._tiles_dir: Path | None = None
        self._hydrated: Hydrated | None = None
        #: What this run has captured, the imagery ``look`` crops.
        self._imagery = RunImagery()
        #: The fused pass the flight in progress fetched when it held its area.
        self._held: tuple[Any, dict[str, Any]] | None = None
        #: The area the flight in progress was flown for, about its point: what
        #: the observation is asked over again when the fused pass is spent.
        self._held_area: tuple[list[tuple[float, float]], Any] | None = None
        #: The nearest the pose came to the last leg's point.
        self._closest_m = float("inf")
        #: The recording's last answer over the leg's area: the unheld points a
        #: short pass is re-asked toward.
        self._last_coverage: dict[str, Any] | None = None

        from benchmarks import _git_provenance  # noqa: PLC0415

        prov = _git_provenance()
        self.git_commit: str | None = prov.get("git_commit")
        self.git_dirty: bool | None = prov.get("git_dirty")

    # -- the operator's channel -------------------------------------------

    def tasking(self, text: str) -> None:
        """Queue an operator tasking; never blocks, never rejects. Typed
        mid-turn it lands at the top of the next one; typed after the model
        reported it wakes the parked session. The row is written when the
        operator speaks."""
        text = (text or "").strip()
        if not text:
            return
        self.transcript.append({"kind": "retask", "said": text})
        self._taskings.put(text)

    def _next_tasking(self) -> str | None:
        """The ``retask`` callable handed to the loop."""
        try:
            return self._taskings.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """End the mission. The loop's cancel event does the work; a mission
        parked on its report ended at the reply, and keeps its ``ended_by``."""
        self._cancel.set()
        with self._lock:
            if self.status in (STATE_FLYING, STATE_AWAITING):
                if self.status == STATE_FLYING:
                    self.ended_by = "stopped"
                self.status = STATE_ENDED

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the leg ------------------------------------------------------------

    def _area_lonlat(self, area: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
        """``area`` in WGS84, the form the recording is asked in; None when the map georeferences nothing."""
        if not self._tiles_dir or not self.map_name:
            return None
        out: list[tuple[float, float]] = []
        for x, y in area:
            lonlat = local_to_lonlat(self._tiles_dir, self.map_name, float(x), float(y))
            if lonlat is None:
                return None
            out.append((float(lonlat[0]), float(lonlat[1])))
        return out

    def _fly_live(self, inner, h: Hydrated, item: dict[str, Any]) -> dict[str, Any]:
        """A flight on the aircraft, under the one arrival rule: a leg is flown
        for an AREA (:func:`leg_area`) and has arrived when the recording holds
        it, whichever pass held it. The observation is the frames that held
        it, fused, the ONE thing a waypoint serves. An area the recording
        holds already is answered without a leg, and only a ring that ANSWERED
        decides that (:meth:`_coverage_before_leg`). Otherwise the leg is
        posted, the recording is asked as it grows until it holds the area,
        then until the point falls out of the frame (:meth:`_wait_departure`),
        and the frames that held the area are fetched. A leg that never
        arrives within ``ARRIVAL_TIMEOUT_S`` serves the part of the area the
        pass did hold, and nothing when none was; the ``leg`` row says which.
        The model is served the stop as the target it named either way: the
        pass geometry is the crew's to fix, not the reasoner's."""
        from benchmarks.citynav.run_benchmark.mission_engine import resolve_flight_target  # noqa: PLC0415

        token = str(item.get("target_fid") or "")

        def _node_xy(fid: str):
            xy = h.search_loop.world_xy_for_node(fid)
            return ((float(xy[0]), float(xy[1])), "prior") if xy is not None else None

        markers = list(getattr(h.episode, "marker_positions", None) or [])
        marker_xy = ((float(sum(p[0] for p in markers) / len(markers)),
                      float(sum(p[1] for p in markers) / len(markers))) if markers else None)
        resolved = resolve_flight_target(token, node_xy=_node_xy, pose_xy=h.state.pose_xy, marker_xy=marker_xy)
        if "error" not in resolved:
            xy = [float(resolved["xy"][0]), float(resolved["xy"][1])]
            area = leg_area(h, token, xy)
            area_ll = self._area_lonlat(area)
            step = len(self.flights)
            cov, unread = self._coverage_before_leg(area_ll)
            held = self._held_frame(area, xy) if cov and cov["fraction"] >= COVERED_FRACTION else None
            if held is not None:
                self.transcript.append({"kind": "leg", "said": (
                    f"{token} at ({xy[0]:.0f}, {xy[1]:.0f}) is held by the pass already: "
                    f"{self._held_words(held[1])}; no leg is flown")})
            else:
                if unread is not None:
                    log.warning("the leg to %s at (%.0f, %.0f) is posted unread: the recording did not answer "
                                "in %d reads: %s", token, xy[0], xy[1], COVERAGE_TRIES, unread)
                    self.transcript.append({"kind": "leg", "said": (
                        f"{token} at ({xy[0]:.0f}, {xy[1]:.0f}) may be held by the pass already, but the "
                        f"recording could not be asked in {COVERAGE_TRIES} reads ({unread}); the leg is flown")})
                elif cov is not None and cov["fraction"] < COVERED_FRACTION:
                    log.info("the pass holds %.0f%% of %s at (%.0f, %.0f) (%d of %d points, %d frames), short "
                             "of %.0f%%; the leg is flown", 100.0 * cov["fraction"], token, xy[0], xy[1],
                             cov["held"], cov["points"], cov["frames"], 100.0 * COVERED_FRACTION)
                elif cov is not None:
                    self.transcript.append({"kind": "leg", "said": (
                        f"{token} at ({xy[0]:.0f}, {xy[1]:.0f}) is held by the pass ({cov['fraction']:.0%}), "
                        f"but no frame of it could be served "
                        f"({self._live.last_refusal or 'no reason recorded'}); the leg is flown")})
                ahead = [(float(x), float(y)) for x, y in (item.get("pass_ahead") or [])]
                leg = Leg(target=token, point=(xy[0], xy[1]), step=step, area=area_ll,
                          coverage_floor=COVERED_FRACTION if area_ll is not None else None, ahead=ahead)
                self._vehicle.post(leg)
                # The leg's reach: how far from the point the aircraft is still
                # over the area. A pass that came inside it and left with the
                # area part held is short.
                reach = max(math.dist(xy, p) for p in area) if area else None
                shorts = 0
                self._closest_m = float("inf")
                while True:
                    outcome = self._wait_arrival(xy, area=area_ll, timeout_s=ARRIVAL_TIMEOUT_S,
                                                 reach_m=reach if shorts < SHORT_PASS_REQUESTS else None)
                    if outcome != "short":
                        break
                    shorts += 1
                    moved, note = self._short_pass(token, xy, area, shorts)
                    self._vehicle.post(Leg(target=token, point=(moved[0], moved[1]), step=step, area=area_ll,
                                       coverage_floor=leg.coverage_floor, ahead=ahead, revision=shorts, note=note))
                if outcome == "arrived":
                    waited = self._wait_departure(xy)
                    held = self._held_frame(area, xy)
                    if held is not None:
                        last = held[1]
                        reg = last.get("registration") or {}
                        by = ", ".join(f"{n} {k}" for k, n in (reg.get("sources") or {}).items())
                        posed = (f"the map registered it{f' ({by})' if by else ''}" if reg.get("applied") else
                                 f"the map did not register it: {reg.get('why') or 'the corrector is not running'}")
                        self.transcript.append({"kind": "leg", "said": (
                            f"the leg to {token} at ({xy[0]:.0f}, {xy[1]:.0f}) arrived; {waited:.1f} s later "
                            f"{self._held_words(last)}; {posed}")})
                    else:
                        self.transcript.append({"kind": "leg", "said": (
                            f"the leg to {token} at ({xy[0]:.0f}, {xy[1]:.0f}) arrived, but no frame of the pass "
                            f"could be served ({self._live.last_refusal or 'no reason recorded'}); "
                            f"the flight has no observation")})
                elif outcome == "lost":
                    held = self._held_frame(area, xy)
                    if held is not None:
                        self.transcript.append({"kind": "leg", "said": (
                            f"the leg to {token} at ({xy[0]:.0f}, {xy[1]:.0f}) never arrived in "
                            f"{ARRIVAL_TIMEOUT_S:.0f} s (closest {self._closest_m:.0f} m); "
                            f"{self._held_words(held[1])}, and those frames are the observation")})
                    else:
                        self.transcript.append({"kind": "leg", "said": (
                            f"the leg to {token} at ({xy[0]:.0f}, {xy[1]:.0f}) never arrived in "
                            f"{ARRIVAL_TIMEOUT_S:.0f} s (closest {self._closest_m:.0f} m); no frame held any of "
                            f"it, and the flight has no observation")})
            self._held, self._held_area = held, (area, xy)
        try:
            return inner(item)
        finally:
            # The fused pass is spent with the flight; the area stays, so a
            # later observation in the turn is the pass over it again.
            self._held = None

    def _wait_departure(self, xy) -> float:
        """Hold until the point leaves the camera's frame, and answer with the
        seconds it took. The arrival rule turns true as the point crosses INTO
        the frame; waiting for it to fall out the other side puts the whole
        pass in the ring. Bounded by the transit the frame's geometry allows
        and by :data:`FRAME_WAIT_S`."""
        bound = self._live.frame_transit_s()
        if bound is None:
            return 0.0
        bound = min(float(bound), FRAME_WAIT_S)
        t0 = self._vehicle.now()
        while not self._cancel.is_set() and self._vehicle.now() - t0 < bound:
            if not self._live.in_frame(float(xy[0]), float(xy[1])):
                break
            self._vehicle.sleep(0.25)
        return self._vehicle.now() - t0

    def _coverage(self, area_lonlat: list[tuple[float, float]] | None) -> dict[str, Any] | None:
        """How much of the area the recording holds now, poses only; None when
        it cannot be asked, or there is no area."""
        if not area_lonlat:
            return None
        cov = self._live.coverage(area_lonlat)
        if cov is not None:
            self._last_coverage = cov
        return cov

    def _coverage_before_leg(self, area_lonlat) -> tuple[dict[str, Any] | None, str | None]:
        """:meth:`_coverage` asked up to ``COVERAGE_TRIES`` times,
        ``COVERAGE_RETRY_S`` apart, because only a read that answers may decide
        whether a leg is flown. ``(coverage, refusal)``: the refusal is the
        live source's when the ring never answered."""
        if not area_lonlat:
            return None, None
        refusal = None
        for k in range(1, COVERAGE_TRIES + 1):
            cov = self._coverage(area_lonlat)
            if cov is not None:
                return cov, None
            refusal = self._live.last_refusal or "no reason recorded"
            log.warning("read %d of %d: the recording could not be asked whether it holds the area: %s",
                        k, COVERAGE_TRIES, refusal)
            if k < COVERAGE_TRIES:
                self._vehicle.sleep(COVERAGE_RETRY_S)
                if self._cancel.is_set():
                    break
        return None, refusal

    @staticmethod
    def _coverage_block(cov: dict[str, Any] | None) -> dict[str, Any] | None:
        """The recording's answer as a record: the share, which of the area's
        points some frame held, and how many frames held any."""
        if cov is None:
            return None
        best = cov.get("best") or []
        return {"fraction": round(float(cov.get("fraction") or 0.0), 3), "held": [b is not None for b in best],
                "frames": len({b[1] for b in best if b is not None})}

    def _held_now(self, h: Hydrated, stops: list) -> list[bool]:
        """Which of a pass's stops the recording holds already, for the loop's
        greedy order (``MissionSpec.held_now``): each stop's ground as its leg
        would be flown for it, off one index read."""
        out = [False] * len(stops)
        if not stops:
            return out
        areas, idx = [], []
        for i, (token, xy) in enumerate(stops):
            if xy is None:
                continue
            ll = self._area_lonlat(leg_area(h, str(token), [float(xy[0]), float(xy[1])]))
            if ll:
                areas.append(ll)
                idx.append(i)
        covs = self._live.coverage_of_each(areas) if areas else None
        for i, cov in zip(idx, covs or []):
            out[i] = bool(cov and float(cov.get("fraction") or 0.0) >= COVERED_FRACTION)
        return out

    def _short_pass(self, token: str, xy, area: list[tuple[float, float]], n: int) -> tuple[list[float], str]:
        """Where a short pass is asked again, and the line for why: the area's
        centroid moved toward the centroid of the points no frame held, by the
        missing ground's extent along that line, no further than the leg's
        reach. Said in the transcript with the points held, the frames, and the
        side that is missing."""
        cov = self._last_coverage or {}
        best = cov.get("best") or []
        unheld = [p for p, b in zip(area, best) if b is None] or list(area)
        cx, cy = (sum(p[0] for p in area) / len(area), sum(p[1] for p in area) / len(area))
        ux, uy = (sum(p[0] for p in unheld) / len(unheld), sum(p[1] for p in unheld) / len(unheld))
        dx, dy = ux - cx, uy - cy
        span = math.hypot(dx, dy)
        reach = max(math.dist((cx, cy), p) for p in area)
        if span < 1.0:
            shift, moved = 0.0, [cx, cy]
        else:
            extent = max(((p[0] - cx) * dx + (p[1] - cy) * dy) / span for p in unheld)
            shift = min(max(extent, span), reach)
            moved = [cx + dx / span * shift, cy + dy / span * shift]
        side = "N NE E SE S SW W NW".split()[round(math.degrees(math.atan2(dx, dy)) / 45.0) % 8] if span >= 1.0 else "-"
        block = self._coverage_block(cov) or {"frames": 0}
        held = sum(1 for b in best if b is not None)
        said = (f"the pass over {token} at ({xy[0]:.0f}, {xy[1]:.0f}) held {held}/{len(area)} points of the area in "
                f"{block['frames']} frame(s), the {side} side missing; "
                f"the leg is asked again {shift:.0f} m {side} of the point (re-fly {n} of {SHORT_PASS_REQUESTS})")
        log.info(said)
        self.transcript.append({"kind": "leg", "said": said})
        return moved, f"Re-fly {token}: the pass held {held}/{len(area)} points, the {side} side missing"

    @staticmethod
    def _held_words(last: dict[str, Any]) -> str:
        """The leg row's account of what held the area: the share, the frames, the nearest to nadir."""
        cov = last.get("coverage") or {}
        seqs = cov.get("seqs") or ([last["seq"]] if last.get("seq") is not None else [])
        short = f", {cov['fetch_failed']} of them not fetched" if cov.get("fetch_failed") else ""
        return (f"the pass held {float(cov.get('fraction', 1.0)):.0%} of it in {len(seqs)} frame(s){short}, "
                f"the nearest {last.get('seq')} at {last.get('off_nadir_deg')} deg off nadir")

    def _held_frame(self, area: list[tuple[float, float]], xy) -> tuple[Any, dict[str, Any]] | None:
        """The ground the pass held over ``area`` (tile metres, about ``xy``) as
        the observation, with the live source's record of it; None when no
        frame held any of it."""
        points = self._area_lonlat(area)
        if points is None:
            return None
        obs = self._live.observe_area(points)
        last, refusal = dict(self._live.last or {}), self._live.last_refusal
        if obs is None:
            log.info("no frame in the recording held the area about (%.0f, %.0f): %s",
                     float(xy[0]), float(xy[1]), refusal)
            return None
        last["observation"] = self._downlink(obs)
        return obs, last

    def _downlink(self, obs) -> str | None:
        """The observation as one north-up image under ``observations/``, the
        fused pixels the model is about to be given, by its sha256; None when
        it could not be written, which is a log line, never a lost stop."""
        from PIL import Image  # noqa: PLC0415

        out = BytesIO()
        Image.fromarray(np.ascontiguousarray(obs.rgb)).save(out, format="JPEG", quality=OBSERVATION_JPEG_QUALITY)
        jpeg = out.getvalue()
        sha = hashlib.sha256(jpeg).hexdigest()
        try:
            folder = self.workdir / OBSERVATIONS_DIR
            folder.mkdir(exist_ok=True)
            (folder / f"{sha}.jpg").write_bytes(jpeg)
        except OSError:
            log.exception("observation not written under %s", self.workdir)
            return None
        return sha

    def _wait_arrival(self, xy: Any, *, area: list[tuple[float, float]] | None, timeout_s: float,
                      reach_m: float | None = None) -> str:
        """``arrived`` when the recording holds ``area`` at ``COVERED_FRACTION``
        (asked as it grows, poses only); ``short`` when ``reach_m`` is given and
        the pass came inside it, left (``SHORT_PASS_LEAVE_M`` for
        ``SHORT_PASS_LEAVE_S``) with the area part held; ``lost`` when
        ``timeout_s`` ran out airborne; ``cancel`` when the mission was stopped."""
        best = math.inf
        receding_since: float | None = None
        # The clock runs on the FLIGHT: it does not spend its timeout while the
        # aircraft is still on the ground.
        t0 = self._vehicle.now()
        while not self._cancel.is_set():
            pose = self._pose()
            if pose is not None:
                d = math.hypot(pose["x"] - float(xy[0]), pose["y"] - float(xy[1]))
                best = min(best, d)
                self._closest_m = best
                cov = self._coverage(area)
                holds = bool(cov and float(cov.get("fraction") or 0.0) >= COVERED_FRACTION)
                part = bool(cov) and 0.0 < float(cov.get("fraction") or 0.0) < COVERED_FRACTION
                # Over the area and gone, with part of it held: a short pass.
                if reach_m is not None and part and best <= reach_m and d > best + SHORT_PASS_LEAVE_M:
                    receding_since = receding_since or self._vehicle.now()
                    if self._vehicle.now() - receding_since >= SHORT_PASS_LEAVE_S:
                        log.info("short pass at (%.0f, %.0f): closest %.0f m, now %.0f m off, %.0f%% held",
                                 float(xy[0]), float(xy[1]), best, d, 100.0 * float(cov.get("fraction") or 0.0))
                        return "short"
                else:
                    receding_since = None
                if holds:
                    return "arrived"
                if pose["agl_m"] < AIRBORNE_AGL_M:
                    t0 = self._vehicle.now()
            if self._vehicle.now() - t0 > timeout_s:
                log.warning("no arrival at (%.0f, %.0f) in %.0f s, closest %.0f m: the leg is lost",
                            float(xy[0]), float(xy[1]), timeout_s, best)
                return "lost"
            self._vehicle.sleep(1.0)
        return "cancel"

    def _pose(self) -> dict[str, float] | None:
        """Where the aircraft is: tile-local x/y and the height above the map's floor; None without a fix."""
        own = self._live.ownship()
        if not own or own.get("x") is None:
            return None
        return {"x": float(own["x"]), "y": float(own["y"]), "agl_m": float(own.get("agl_m") or 0.0)}

    def _note_frame(self, last: dict[str, Any], obs) -> None:
        """One row of ``frames.jsonl``: the observation this mission just
        perceived, keyed to the recording by the newest frame's ``log_time``
        and ``seq``, with every frame of the fuse in ``mosaic``."""
        row = {"log_time": last.get("log_time"), "seq": last.get("seq"), "t_host": last.get("t_host"),
               "utc": last.get("utc"), "age_s": last.get("age_s"), "attitude": last.get("attitude"),
               "pose_xy": [float(obs.pose_xy[0]), float(obs.pose_xy[1])], "altitude_m": float(obs.altitude_m),
               "corners_xy": last.get("corners_xy"), "footprint_xy": last.get("footprint_xy"),
               "boresight": last.get("boresight"), "off_nadir_deg": last.get("off_nadir_deg"),
               "registration": last.get("registration"), "mosaic": last.get("mosaic"),
               "observation": last.get("observation"), "flight": len(self.flights)}
        try:
            with (self.workdir / FRAMES_JSONL).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            log.exception("frames.jsonl not written under %s", self.workdir)

    # -- lifecycle ---------------------------------------------------------

    def elapsed_s(self) -> float | None:
        """Seconds since the mission started, or None before: the axis a transcript row is stamped on."""
        return None if self.started_at is None else time.time() - self.started_at

    def _utc_now(self) -> float | None:
        """The receiver's UTC now, unix seconds, or None without a fix."""
        stamp = (self._live.ownship() or {}).get("utc")
        if not stamp:
            return None
        try:
            return dt.datetime.fromisoformat(str(stamp)).timestamp()
        except ValueError:
            return None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("mission already started")
        self.started_at = time.time()
        self.started_at_utc = self._utc_now()
        self._thread = threading.Thread(target=self._run, name=f"mission-{self.episode_id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from benchmarks.citynav.paths import TILES_DIR  # noqa: PLC0415
        from benchmarks.citynav.run_benchmark.mission_engine import build_mission_session  # noqa: PLC0415
        from maps.llm.endpoint_spec import discover_from_env, install_spec  # noqa: PLC0415
        from maps.llm.http import CallCanceled  # noqa: PLC0415
        from maps.search.parse_compile.reasoner_loop import continue_mission, run_mission  # noqa: PLC0415

        try:
            with self._lock:
                self.status = STATE_FLYING
            # One probe per process, before any inference call: a dead reasoner
            # fails this mission, honestly.
            install_spec(discover_from_env())
            self._write_manifests()

            h = hydrate(episode_id=self.episode_id, split=self.split, config=self.config, workdir=self.workdir,
                        start_xy=self._start_xy)
            self._hydrated = h
            instruction = self.instruction_override or h.episode.description
            # The override IS the operator's instruction, so the episode
            # carries no second copy: fly_one echoes the description back as
            # the flight report's cue and hints the detector with it, and the
            # split's own sentence would serve an attribute the operator
            # withheld. ponytail: stamp the episode rather than thread the
            # instruction through fly_one.
            h.episode.description = instruction
            with self._lock:
                self.instruction = instruction
                self.map_name = h.episode.map_name
            self._tiles_dir = TILES_DIR

            # The live frames project onto this mission's tile grid; look crops
            # what this run captured, never the bake's raster.
            self._live.tile = h.tile_frame
            h.search_loop.look_imagery = self._imagery

            def observe():
                # The pass fused over the flight's area, fetched by the flight
                # when it held it, else asked over again now; a waypoint's
                # observation is that or nothing.
                held, self._held = self._held, None
                if held is None and self._held_area is not None:
                    held = self._held_frame(*self._held_area)
                if held is None:
                    log.warning("live observation refused: %s", self._live.last_refusal)
                    return None
                obs, last = held
                log.info("live frame seq %s log_time %s age %.2fs attitude %s fix gap %.2fs",
                         last.get("seq"), last.get("log_time"), last.get("age_s", 0.0),
                         last.get("attitude"), last.get("gap_s", 0.0))
                self._note_frame(last, obs)
                self._imagery.add(obs)
                return obs

            session = build_mission_session(
                cfg=h.cfg, state=h.state, assets=h.assets, episode=h.episode, sensor=h.sensor,
                tile_frame=h.tile_frame, episode_ply_path=None, perception=h.perception,
                search_loop=h.search_loop, recorder=self._recorder, osm_ctx=None, t_start=time.monotonic(),
                instruction=instruction, observe=observe,
            )
            self.flights = session.flights
            # On the aircraft the leg is flown BEFORE the frame is read, and a
            # pass takes the stop the recording holds first.
            session.mission_spec.fly = functools.partial(self._fly_live, session.mission_spec.fly, h)
            session.mission_spec.held_now = functools.partial(self._held_now, h)

            common = dict(
                ctx=session.ctx,
                mission_spec=session.mission_spec,
                look_renderer=(h.search_loop._render_candidate_crops if h.search_loop._look_tool_enabled else None),
                look_audit_dir=h.search_loop._look_audit_dir(),
                obs_span_m=session.obs_span_m,
                obs_minor_m=session.obs_minor_m,
                watchlist_apply=h.search_loop._watchlist_apply_cb,
                boost_queue=session.boost,
                transcript=self.transcript,
                retask=self._next_tasking,
            )
            seed = h.compiler._resolve_seed_base() & 0x7FFFFFFF
            reply, reasoner = run_mission(
                instruction=instruction,
                reasoner_url=h.compiler.reasoner_url,
                endpoint_spec=h.compiler.endpoint_spec,
                api_key=h.compiler.api_key,
                seed=seed,
                sensor_description=h.compiler.sensor_description,
                cancel_event=self._cancel,
                look_enabled=h.search_loop._look_tool_enabled,
                diag=self._turn_diag,
                reasoning_effort=self.reasoning_effort,
                **common,
            )
            self._record_reply(reply)

            # THE REPLY IS A REPORT, NOT AN ENDING: park and wait for the
            # operator; each tasking drives another turn over the same conversation.
            turn = 1
            while not self._cancel.is_set():
                nxt = self._await_tasking()
                if nxt is None:
                    break
                with self._lock:
                    self.status = STATE_FLYING
                    self.ended_by = None
                    self.ended_at = None
                turn += 1
                reply = continue_mission(reasoner=reasoner, instruction=nxt, diag=self._turn_diag,
                                         rid_tag=f"m{turn}", **common)
                self._record_reply(reply)
            with self._lock:
                self.status = STATE_ENDED
        except CallCanceled:
            # Stop, pressed while the reasoner call was in flight.
            with self._lock:
                self.status = STATE_ENDED
                self.ended_by = "stopped"
        except Exception as exc:  # noqa: BLE001 -- the operator must see why
            if self._cancel.is_set():
                log.info("mission %s: %s after the stop", self.episode_id, type(exc).__name__)
                with self._lock:
                    self.status, self.ended_by = STATE_ENDED, "stopped"
            else:
                log.exception("mission %s failed", self.episode_id)
                with self._lock:
                    self.status = STATE_FAILED
                    self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                summary = {"stage": "mission", "kind": "summary", "status": self.status,
                           "ended_by": self.ended_by, "turns_used": self._turns(),
                           "flights_flown": len(self.flights), "error": self.error}
            self.transcript.append(summary)
            self.bank()
            self.close_log()

    def _write_manifests(self) -> None:
        """The sortie's ``run_manifest.json``, written by the first mission flown
        in it, and this mission's own in its workdir with the mission's grain:
        which mission, and the reasoning effort its requests carry. Non-fatal."""
        from benchmarks.citynav.run_manifest import MANIFEST_NAME, build_manifest, emit_or_verify  # noqa: PLC0415

        sortie = self.workdir.parent
        common = dict(config_path=Path(self.config), split=self.split, sample_seed=None, sample_offset=None,
                      limit=None, workers=None, allowlist_path=None)
        try:
            if not (sortie / MANIFEST_NAME).exists():
                emit_or_verify(sortie, build_manifest(out_root=sortie, episodes=[], **common), log=log.info)
            emit_or_verify(self.workdir, build_manifest(out_root=self.workdir, episodes=[self.episode_id],
                                                        mission=self._mission_stamp(), **common), log=log.info)
        except Exception:  # noqa: BLE001 -- a mission without a manifest still flies
            log.exception("run_manifest not written for %s", self.workdir)

    def _mission_stamp(self) -> dict[str, Any]:
        """The mission's own grain of the run manifest: the reasoning effort its
        requests carry, the mission's pick, else the arm's environment value,
        else the serve's own default (None), and which."""
        arm = os.environ.get("SEARCH_SAMPLING_REASONING_EFFORT") or None
        if self.reasoning_effort is not None:
            effort, source = self.reasoning_effort, "mission"
        elif arm is not None:
            effort, source = arm, "arm"
        else:
            effort, source = None, "serve"
        return {"episode_id": self.episode_id, "run_id": self.run_id, "run_no": self.run_no,
                "reasoning_effort": effort, "reasoning_effort_source": source}

    def _turns(self) -> int:
        """Model turns taken: the conversations that have ended plus the turn the one in flight is on."""
        return self.turns_used + int(self._turn_diag.get("turns_used") or 0)

    def _record_reply(self, reply) -> None:
        with self._lock:
            self.ended_by = reply.ended_by
            self.ended_at = time.time()
            self.reply_text = getattr(reply, "reply_text", None)
            self.turns_used += int(getattr(reply, "turns_used", 0) or 0)
            self._turn_diag.clear()
            self.replies += 1
            self.status = STATE_AWAITING
        self.bank()

    def _await_tasking(self) -> str | None:
        """Block until the operator speaks or the mission is stopped."""
        while not self._cancel.is_set():
            try:
                return self._taskings.get(timeout=0.5)
            except queue.Empty:
                continue
        return None

    # -- the record ----------------------------------------------------------

    def score(self, visited: list[list[float]]) -> dict[str, Any] | None:
        """SR/OSR/NE/SPL for the mission through the bench's own scorer, the
        flown points standing for the trajectory and the last of them for the
        commit; None for an episode with no target."""
        h = self._hydrated
        if h is None or h.episode.target_xy is None or len(visited) < 1:
            return None
        from benchmarks.citynav import eval as bench_eval  # noqa: PLC0415
        from benchmarks.citynav.adapter import load_objects  # noqa: PLC0415
        from benchmarks.citynav.paths import OBJECTS_JSON  # noqa: PLC0415
        from benchmarks.citynav.run_benchmark.episode_runner import _success_mask  # noqa: PLC0415

        trajectory = [(float(p[0]), float(p[1])) for p in visited]
        criterion, mask = _success_mask(h.cfg, h.episode, load_objects(OBJECTS_JSON))
        scores = bench_eval.score_arrival(
            criterion=criterion, final_pos_xy=trajectory[-1], pred_xy=trajectory[-1],
            target_xy=(float(h.episode.target_xy[0]), float(h.episode.target_xy[1])),
            pos_trajectory=trajectory, pred_trajectory=[trajectory[-1]],
            success_dist_m=float(h.cfg.success_radius_m), flew=len(trajectory) > 1, **mask)
        scores["success_basis"] = "final_pos_xy"
        return scores

    def row(self) -> dict[str, Any]:
        """The mission as a ledger row, the keys a bench row carries, so
        ``benchmarks.citynav.analysis`` reads a sortie the way it reads a run.
        The commit is the last flight, verbatim; zero flights is an honest zero."""
        from benchmarks.citynav import paths  # noqa: PLC0415
        from benchmarks.citynav.episodes import STATUS_INFRA, STATUS_OK  # noqa: PLC0415
        from benchmarks.citynav.run_manifest import arm_env_stamp  # noqa: PLC0415
        from maps.llm.endpoint_spec import spec, spec_installed  # noqa: PLC0415

        episode = self._hydrated.episode if self._hydrated else None
        with self._lock:
            visited = [[float(episode.start_xy[0]), float(episode.start_xy[1])]] if episode else []
            visited += [[float(f["resolved_xy"][0]), float(f["resolved_xy"][1])] for f in self.flights]
            status, ended_by, error, turns = self.status, self.ended_by, self.error, self._turns()
            flown = len(self.flights)
        last = visited[-1] if visited else None
        try:
            metrics = self.score(visited)
        except Exception:  # noqa: BLE001 -- an unscored row still lands
            log.exception("scoring failed for %s", self.episode_id)
            metrics = None
        return {
            "episode_id": self.episode_id, "run_id": self.run_id, "run_no": self.run_no, "split": self.split,
            "map_name": self.map_name, "description": self.instruction, "reasoning_effort": self.reasoning_effort,
            "start_xy": visited[0] if episode else None,
            "target_xy": ([float(episode.target_xy[0]), float(episode.target_xy[1])]
                          if episode is not None and episode.target_xy is not None else None),
            "pred_xy": last, "final_pos_xy": last, "success_basis": "final_pos_xy", "visited_xy": visited,
            # A mission PARKED on its report flew: ok with ``ended_by: reply``. Only a fault is infra.
            "status": STATUS_OK if status in (STATE_ENDED, STATE_AWAITING) else STATUS_INFRA,
            "error": error, "ended_by": ended_by, "turns_used": turns, "flights_flown": flown,
            "metrics": metrics,
            "provenance": {"git_commit": self.git_commit, "git_dirty": self.git_dirty,
                           "served_model": spec().model_id if spec_installed() else None,
                           "bake_sha256": paths.bake_sha256()},
            "search_env_flags": arm_env_stamp(),
            "ts": self.started_at,
            "started_at_utc": self.started_at_utc,
            "elapsed_s": ((self.ended_at or time.time()) - self.started_at) if self.started_at else None,
        }

    def bank(self) -> None:
        """The run's row on disk: ``episode.json`` in the mission's directory
        and ONE line in the sortie's ``episodes.jsonl``, keyed by the run, so
        a parked run that flies again replaces its row. Banked at every report
        and at the end; a row that will not land is logged, never raised."""
        from benchmarks.citynav.episodes import upsert_row, write_episode_json  # noqa: PLC0415

        try:
            row = self.row()
            sortie = self.workdir.parent
            write_episode_json(self.workdir, row)
            upsert_row(sortie, {**row, "workdir": str(self.workdir.relative_to(sortie))})
        except Exception:  # noqa: BLE001 -- a mission is not failed by its paperwork
            log.exception("the row for %s did not land", self.episode_id)
