"""One flying mission, held open across operator turns: the method half of the station that flew the C-12.

The search stack is hydrated as the bench hydrates it and the conversation
flies on a worker thread. Two things differ from the bench. The reply is a
REPORT: the session parks in ``awaiting_tasking`` and the operator's answer
goes to ``continue_mission`` on the same mission. The operator speaks
whenever: taskings queue and the loop drains them at the top of each turn.
A flight is a leg flown for an AREA (``legs.Legs.fly``); it arrives when the
recording holds the area, and its observation is the frames that held it,
fused. The row is banked at each report and scored by ``measures``.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import logging
import queue
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import git_provenance
from benchmarks.bench import Bench
from benchmarks.flight.legs import Hydrated, Legs
from benchmarks.flight.live_source import LiveSource
from benchmarks.flight.run_imagery import RunImagery
from benchmarks.flight.tile_georef import local_to_lonlat, lonlat_to_local
from benchmarks.run_record.config import load_config
from benchmarks.run_record.episode_workdir import (
    CONVERSATION_JSONL,
    FRAMES_JSONL,
    OBSERVATIONS_DIR,
    TRACK_JSONL,
    ConversationLog,
)
from benchmarks.run_record.episodes import (
    MANIFEST_NAME,
    STATUS_INFRA,
    STATUS_OK,
    upsert_row,
    write_episode_json,
)
from benchmarks.run_record.run_manifest import (
    arm_env_stamp,
    build_manifest,
    emit_or_verify,
    settle_effort,
)
from maps.llm import endpoint_spec
from maps.llm.http import CallCanceled
from maps.llm.model_cards import check_effort, effort
from maps.settings import from_environ

log = logging.getLogger("flight.mission")

#: The quality a waypoint's observation is written at (``observations/<sha>.jpg``).
OBSERVATION_JPEG_QUALITY = 90
#: The ownship track's period (``TRACK_JSONL``), seconds on the vehicle's clock.
TRACK_PERIOD_S = 1.0

STATE_IDLE = "idle"
STATE_FLYING = "flying"
STATE_AWAITING = "awaiting_tasking"
STATE_ENDED = "ended"
STATE_FAILED = "failed"


def resolve_episode(
    bench: Bench, episode_id: str, split: str, start_xy: tuple[float, float] | None = None
):
    """The row of ``bench``'s split named ``episode_id``; with ``start_xy``, a
    mission flown under the operator's own words: the map its id names, the
    start given, and no target."""
    engine = bench.engine
    if start_xy is not None:
        return engine.authored_episode(episode_id, start_xy)
    found = engine.find_row(engine.load_split(bench.split_path(split)), episode_id)
    if found is None:
        raise ValueError(f"{split} holds no episode {episode_id}")
    return engine.parse_episode(found[1], row_index=found[0])


def hydrate(
    *,
    bench: Bench,
    episode_id: str,
    split: str,
    config: str,
    agl_m: float,
    start_xy: tuple[float, float] | None = None,
    workdir: Path | None = None,
) -> Hydrated:
    """The hydrated search stack for one mission, built by the bench's engine
    (``EpisodeEngine.build_episode``), at the operator's start and at
    ``agl_m``, the crew's commanded height above ground: the one height the
    served sensor block and every leg read. The mission records under
    ``workdir`` and its requests ride its episode id as their routing key."""
    engine = bench.engine
    cfg = load_config(config)
    episode = resolve_episode(bench, episode_id, split, start_xy)
    h = engine.build_episode(
        cfg,
        episode,
        engine.load_assets(bench, episode.map_name),
        bench,
        settings=from_environ(bench.environ),
        workdir=workdir,
        routing_key=episode_id,
    )
    h.state.altitude_m = h.ground_z_m + float(agl_m)
    return h


class _StampedLog(ConversationLog):
    """The conversation log, each row stamped with ``t_s``, vehicle seconds since the mission started."""

    def __init__(self, path: Path, elapsed) -> None:
        super().__init__(path)
        self._elapsed = elapsed

    def append(self, item: dict) -> None:
        super().append({**item, "t_s": self._elapsed()})


class Mission:
    """A mission in the seat: its thread, its tasking queue, its record."""

    def __init__(
        self,
        *,
        bench: Bench,
        episode_id: str,
        split: str,
        config: str,
        workdir: Path,
        live: LiveSource,
        recorder: Any,
        close_log,
        agl_m: float,
        instruction: str | None = None,
        run_id: str | None = None,
        run_no: int | None = None,
        start_xy: tuple[float, float] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        #: The bench the mission flies: its split, its bake, its engine.
        self.bench = bench
        self.episode_id = episode_id
        self.split = split
        self.config = config
        self.workdir = Path(workdir)
        #: A string the serve accepts; None (or blank) flies the arm's.
        self.reasoning_effort = (reasoning_effort or "").strip() or None
        #: The height above ground the crew is commanded to fly.
        self.agl_m = float(agl_m)
        self.run_id = run_id
        self.run_no = run_no
        self._start_xy = start_xy
        self.live = live
        self._vehicle = live.vehicle
        self._recorder = recorder
        self.close_log = close_log

        self._lock = threading.Lock()
        self._taskings: queue.Queue[str] = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

        self.status = STATE_IDLE
        self.instruction = instruction or ""
        self.ended_by: str | None = None
        #: Vehicle seconds from the start to the reply, so the row's
        #: ``elapsed_s`` is the mission's and not the park's.
        self.ended_s: float | None = None
        self.error: str | None = None
        #: The mission failed on a serve fault (the bench's infra rule), not its own.
        self.infra = False
        #: Model turns of the conversations that have ENDED; :meth:`turns` adds the one in flight.
        self.turns_used = 0
        #: How many conversations have ended and parked the mission.
        self.replies = 0
        #: How many flights the row scores and counts: those flown by the last report; None, every flight flown.
        self._scored_flights: int | None = None
        self._turn_diag: dict[str, Any] = {}
        #: The host's wall clock at the start (the row's ``ts``).
        self.started_at: float | None = None
        #: The vehicle's clock at the start: every time the mission records
        #: (``t_s``, ``elapsed_s``, the track) is vehicle seconds from here.
        self._t0: float | None = None
        #: The same instant on the receiver's clock (UTC, unix seconds).
        self.started_at_utc: float | None = None
        #: How many track periods have been read; one with no pose writes no line.
        self._tracked = 0
        #: Every row lands in ``<workdir>/conversation.jsonl`` as it happens.
        self.transcript: list[dict] = _StampedLog(self.workdir / CONVERSATION_JSONL, self.elapsed_s)
        self.flights: list[dict] = []
        self.map_name: str | None = None
        self._tiles_dir: Path | None = None
        self.hydrated: Hydrated | None = None
        #: What this run has captured, the imagery ``look`` crops.
        self._imagery = RunImagery()
        #: Each flight's leg on the aircraft, and the pass the flight in progress held.
        self.legs = Legs(
            live=live,
            transcript=self.transcript,
            cancel=self._cancel,
            agl_m=self.agl_m,
            to_lonlat=self._area_lonlat,
            keep_observation=self._downlink,
            flights_flown=lambda: len(self.flights),
        )

        prov = git_provenance()
        self.git_commit: str | None = prov.get("git_commit")
        self.git_dirty: bool | None = prov.get("git_dirty")

    @property
    def waiting(self) -> dict[str, Any] | None:
        """The leg being waited on, None between legs: what the station draws,
        read off the mission as it is off a bench seat."""
        return self.legs.waiting

    # -- the operator's channel -------------------------------------------

    def tasking(self, text: str) -> None:
        """Queue an operator tasking (never blocks): mid-turn it lands at the
        next turn, after a report it wakes the parked session."""
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
        that has reported stands at its report (:meth:`_record_reply`)."""
        self._cancel.set()
        with self._lock:
            if self.status in (STATE_FLYING, STATE_AWAITING):
                self._stopped()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the leg ------------------------------------------------------------

    def _area_lonlat(self, area: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """``area`` in WGS84, the form the recording is asked in."""
        return [
            local_to_lonlat(self._tiles_dir, self.map_name, float(x), float(y)) for x, y in area
        ]

    def _downlink(self, obs) -> str | None:
        """The observation as ``observations/<sha256>.jpg``, its sha; None (logged) when it could not be written."""
        from PIL import Image

        out = BytesIO()
        Image.fromarray(np.ascontiguousarray(obs.rgb)).save(
            out, format="JPEG", quality=OBSERVATION_JPEG_QUALITY
        )
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

    def _note_frame(self, last: dict[str, Any], obs) -> None:
        """One ``frames.jsonl`` row: the observation, keyed to the recording by its newest frame."""
        row = {
            "log_time": last.get("log_time"),
            "seq": last.get("seq"),
            "t_host": last.get("t_host"),
            "utc": last.get("utc"),
            "age_s": last.get("age_s"),
            "attitude": last.get("attitude"),
            "pose_xy": [float(obs.pose_xy[0]), float(obs.pose_xy[1])],
            "altitude_m": float(obs.altitude_m),
            "corners_xy": last.get("corners_xy"),
            "boresight": last.get("boresight"),
            "off_nadir_deg": last.get("off_nadir_deg"),
            "registration": last.get("registration"),
            "mosaic": last.get("mosaic"),
            "observation": last.get("observation"),
            "flight": len(self.flights),
            "world_window": [float(v) for v in obs.world_window],
            "shape": [int(v) for v in obs.rgb.shape[:2]],
            "t_s": self.elapsed_s(),
        }
        try:
            with (self.workdir / FRAMES_JSONL).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            log.exception("frames.jsonl not written under %s", self.workdir)

    # -- lifecycle ---------------------------------------------------------

    def elapsed_s(self) -> float | None:
        """Vehicle seconds since the mission started, or None before: the axis a transcript row is stamped on."""
        return None if self._t0 is None else self._vehicle.now() - self._t0

    def _utc_now(self) -> float | None:
        """The receiver's UTC now, unix seconds, or None without a fix."""
        stamp = (self.live.ownship() or {}).get("utc")
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
        self._t0 = self._vehicle.now()
        self.started_at_utc = self._utc_now()
        self._thread = threading.Thread(
            target=self._run, name=f"mission-{self.episode_id}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        from maps.mission.loop import continue_mission

        engine = self.bench.engine
        try:
            self._set(STATE_FLYING)
            endpoint_spec.install_spec(
                endpoint_spec.discover_from_settings(from_environ(self.bench.environ))
            )
            self.bench = settle_effort(self.bench)
            if self.reasoning_effort:
                check_effort(endpoint_spec.spec().model_id, self.reasoning_effort)
            self._write_manifests()

            h = hydrate(
                bench=self.bench,
                episode_id=self.episode_id,
                split=self.split,
                config=self.config,
                start_xy=self._start_xy,
                agl_m=self.agl_m,
                workdir=self.workdir,
            )
            self.hydrated = h
            instruction = self.instruction or h.episode.description
            # The override IS the operator's instruction, so the episode
            # carries no second copy: a flight echoes the description back as
            # the flight report's cue and hints the detector with it, and the
            # split's own sentence would serve an attribute the operator
            # left out. ponytail: stamp the episode rather than thread the
            # instruction through the flight.
            h.episode.description = instruction
            with self._lock:
                self.instruction = instruction
                self.map_name = h.episode.map_name
            self._tiles_dir = self.bench.tiles_dir

            # look crops what this run captured, never the bake's raster.
            h.world.look_imagery = self._imagery

            def observe():
                # The pass fused over the flight's area, fetched by the flight
                # when it held it; a waypoint's observation is that or nothing.
                held, self.legs.held = self.legs.held, None
                if held is None:
                    log.warning("live observation refused: %s", self.live.last_refusal)
                    return None
                obs, last = held
                log.info(
                    "live frame seq %s log_time %s age %.2fs attitude %s",
                    last.get("seq"),
                    last.get("log_time"),
                    last.get("age_s", 0.0),
                    last.get("attitude"),
                )
                self._note_frame(last, obs)
                self._imagery.add(obs)
                return obs

            session = engine.open_session(h, recorder=self._recorder, observe=observe)
            self.flights = session.flights
            # On the aircraft the leg is flown BEFORE the frame is read, and a
            # pass takes the stop the recording holds first.
            session.mission_spec.fly = functools.partial(self.legs.fly, session)
            session.mission_spec.held_now = functools.partial(self.legs.held_now, h)
            reply, conversation = session.run(
                instruction,
                cancel_event=self._cancel,
                diag=self._turn_diag,
                reasoning_effort=self.reasoning_effort,
                transcript=self.transcript,
                retask=self._next_tasking,
            )
            self._record_reply(reply)

            # THE REPLY IS A REPORT, NOT AN ENDING: park and wait for the
            # operator; each tasking drives another turn over the same conversation.
            turn = 1
            while not self._cancel.is_set():
                nxt = self._await_tasking()
                if nxt is None:
                    break
                self._set(STATE_FLYING)
                turn += 1
                reply = continue_mission(conversation, nxt, rid_tag=f"m{turn}")
                self._record_reply(reply)
            self._set(STATE_ENDED)
        except CallCanceled:
            # Stop, pressed while the reasoner call was in flight.
            with self._lock:
                self._stopped()
        except Exception as exc:
            if self._cancel.is_set():
                log.info("mission %s: %s after the stop", self.episode_id, type(exc).__name__)
                with self._lock:
                    self._stopped()
            else:
                log.exception("mission %s failed", self.episode_id)
                error = f"{type(exc).__name__}: {exc}"
                # A run that has reported stands at its report, whatever failed after it.
                self._set(
                    STATE_FAILED, error=error, infra=not self.replies and engine.is_infra_fault(exc)
                )
        finally:
            self.bank()
            self.close_log()

    def _set(self, status: str, **fields) -> None:
        """``status`` and any other of the mission's fields, set under its lock."""
        with self._lock:
            self.status = status
            for name, value in fields.items():
                setattr(self, name, value)

    def _write_manifests(self) -> None:
        """The sortie's ``run_manifest.json`` (by its first mission) and this
        mission's own in its workdir; a failure is logged."""
        sortie = self.workdir.parent
        common = dict(
            bench=self.bench,
            config_path=Path(self.config),
            split=self.split,
            sample_seed=None,
            limit=None,
            allowlist_path=None,
        )
        try:
            if not (sortie / MANIFEST_NAME).exists():
                emit_or_verify(
                    sortie, build_manifest(out_root=sortie, episodes=[], **common), log=log.info
                )
            emit_or_verify(
                self.workdir,
                build_manifest(
                    out_root=self.workdir,
                    episodes=[self.episode_id],
                    mission=self._mission_stamp(),
                    **common,
                ),
                log=log.info,
            )
        except Exception:
            log.exception("run_manifest not written for %s", self.workdir)

    def _mission_stamp(self) -> dict[str, Any]:
        """The mission's grain of the run manifest: which mission, and the
        reasoning effort its requests carry."""
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "run_no": self.run_no,
            "reasoning_effort": effort(
                self.reasoning_effort
                if self.reasoning_effort is not None
                else self.bench.environ.get("SEARCH_SAMPLING_REASONING_EFFORT", "")
            ),
        }

    def turns(self) -> int:
        """Model turns taken: the conversations that have ended plus the turn the one in flight is on."""
        return self.turns_used + int(self._turn_diag.get("turns_used") or 0)

    def _stopped(self) -> None:
        """Ended by the crew's Stop, under the lock: ``stopped``, unless the run stands at a report."""
        self.status = STATE_ENDED
        if not self.replies:
            self.ended_by = "stopped"

    def _record_reply(self, reply) -> None:
        """A conversation's end parks the mission. The first end, and each
        plain reply after it, is the run's REPORT: the row takes its
        ``ended_by``, its time, and the flights flown by then. Every end after
        a report without a new one (a harness ending, the crew's Stop, a
        failure) leaves the row standing at the last report; its turns keep
        counting, as the run's cost."""
        with self._lock:
            if not self.replies or reply.ended_by == "reply":
                self.ended_by = reply.ended_by
                self.ended_s = self.elapsed_s()
                self._scored_flights = len(self.flights)
            self.turns_used += int(reply.turns_used)
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

    def row(self) -> dict[str, Any]:
        """The mission as a ledger row with a bench row's keys; the commit is the last of ``visited_xy``."""
        episode = self.hydrated.episode if self.hydrated else None
        with self._lock:
            visited = [[float(episode.start_xy[0]), float(episode.start_xy[1])]] if episode else []
            scored = self.flights[: self._scored_flights]
            visited += [[float(f["resolved_xy"][0]), float(f["resolved_xy"][1])] for f in scored]
            infra, ended_by, error, turns = self.infra, self.ended_by, self.error, self.turns()
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "run_no": self.run_no,
            "split": self.split,
            "map_name": self.map_name,
            "description": self.instruction,
            "reasoning_effort": self.reasoning_effort,
            "start_xy": visited[0] if episode else None,
            "visited_xy": visited,
            # The bench's infra rule: only a serve fault is infra; any other failure is a scored miss.
            "status": STATUS_INFRA if infra else STATUS_OK,
            "error": error,
            "ended_by": ended_by,
            "turns_used": turns,
            "flights_flown": len(scored),
            "provenance": {
                "git_commit": self.git_commit,
                "git_dirty": self.git_dirty,
                "served_model": (
                    endpoint_spec.spec().model_id if endpoint_spec.spec_installed() else None
                ),
                "bake_sha256": self.bench.bake_sha256(),
            },
            "search_env_flags": arm_env_stamp(self.bench),
            "ts": self.started_at,
            "started_at_utc": self.started_at_utc,
            "elapsed_s": self.ended_s if self.ended_s is not None else self.elapsed_s(),
        }

    def bank(self) -> None:
        """The run's row in ``episode.json`` and as ONE line of the sortie's
        ``episodes.jsonl``, replaced at every report, then the track flown
        since; a row that will not land is logged, never raised. A start
        without a fix takes its UTC at the first bank with one."""
        try:
            if (
                self.started_at_utc is None
                and self._t0 is not None
                and (utc := self._utc_now()) is not None
            ):
                # No fix at the start: the start's UTC is the first fix's, back through the seconds since.
                self.started_at_utc = utc - self.elapsed_s()
            row = self.row()
            sortie = self.workdir.parent
            write_episode_json(self.workdir, row)
            upsert_row(sortie, {**row, "workdir": str(self.workdir.relative_to(sortie))})
            self._record_track()
        except Exception:
            log.exception("the row for %s did not land", self.episode_id)

    def _record_track(self) -> None:
        """The ownship's fixes since the last one written, one a second on the
        vehicle's clock read through the timing offset as ``started_at_utc``
        is, appended to the sortie's ``flight_track.jsonl``:
        ``t_unix`` is ``started_at_utc`` plus vehicle seconds from the start,
        the axis ``started_at_utc + t_s`` reads every transcript row on. No
        UTC start, no track. ponytail: written at each report and at the end,
        so a process killed mid-turn loses the fixes since the last report;
        a writer on its own thread is the upgrade if that ever matters."""
        if self._t0 is None or self.started_at_utc is None or self._tiles_dir is None:
            return
        lines = []
        while (t := self._t0 + self._tracked * TRACK_PERIOD_S) <= self._vehicle.now() + 1e-9:
            self._tracked += 1
            pose = self.live.pose_now(t)
            xy = (
                None
                if pose is None
                else lonlat_to_local(self._tiles_dir, self.map_name, pose.lon, pose.lat)
            )
            if xy is not None:
                lines.append(
                    json.dumps(
                        {
                            "t_unix": round(self.started_at_utc + (t - self._t0), 3),
                            "x": round(xy[0], 2),
                            "y": round(xy[1], 2),
                            "alt_msl_m": round(pose.alt_msl_m, 2),
                            "alt_agl_m": round(pose.alt_msl_m - self.live.ground_msl(xy), 2),
                            "track_deg": round(pose.track_deg, 2),
                            "ground_speed_mps": round(pose.speed_mps, 3),
                        }
                    )
                )
        if lines:
            with (self.workdir.parent / TRACK_JSONL).open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")


def converse(mission: Mission, taskings: list[str]) -> None:
    """Fly ``mission`` to its end, speaking the next of ``taskings`` at each
    report and stopping when none is left, then close the live source."""
    mission.start()
    pending, heard = list(taskings), 0
    while mission.status in (STATE_IDLE, STATE_FLYING, STATE_AWAITING):
        if mission.status == STATE_AWAITING and mission.replies > heard:
            heard = mission.replies
            if pending:
                mission.tasking(pending.pop(0))
            else:
                mission.stop()
                break
        time.sleep(0.2)
    mission.wait()
    mission.live.close()


def summary_line(mission: Mission) -> str:
    """How a mission ended, in one line."""
    return (
        f"{mission.workdir}: {mission.status}, ended by {mission.ended_by}, {len(mission.flights)} flights, "
        f"{mission.turns()} turns" + (f", error {mission.error}" if mission.error else "")
    )
