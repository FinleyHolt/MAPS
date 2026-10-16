"""A flown run flown again on the release harness, answered from its own recording.

    python -m benchmarks.flight.replay <sortie>/<run dir> [--bank week] [--out DIR]
    python -m benchmarks.flight.replay <sortie>/<run dir> --forced
    python -m benchmarks.flight.replay <sortie>/<run dir> --frames <sensors.mcap> --boresight <chain.json>
        --raster <raster dir> [--nav <sensors_nocamera.mcap>]

The harness's own :class:`~benchmarks.flight.mission.Mission` flies under the
recorded operator's words and start; what answers a flight is the observation
bank (default: the recorded observation holding most of the leg's area, your
reasoner and segmenter answering), the recording itself (``--forced``: every
request diffed into ``replay_diff.json``, a divergence exits non-zero, no
endpoint contacted), or the camera recording re-fused as the aircraft's ring
(``--frames``). The bank answers at the live source's seam. Every replay
writes a flown run's records, its row naming the run it replays
(``replay_of``).
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import difflib
import functools
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import benchmarks.calcity
from benchmarks.citynav.analysis.rows import iter_json_rows
from benchmarks.citynav.episode_workdir import (CONVERSATION_JSONL, DETECTIONS_JSONL, FRAMES_JSONL, OBSERVATIONS_DIR,
                                                TRACK_JSONL)
from benchmarks.citynav.episodes import EPISODES_JSONL, RUN_LOG, TOUR_CURSOR
from benchmarks.flight import map_render
from benchmarks.flight.live_source import check_map, coverage_record, held_record, open_live, ortho_observation
from benchmarks.flight.measures import frame_agl, run_bases, sortie_of
from benchmarks.flight.mission import Mission, converse, flight_config, summary_line
from benchmarks.flight.tile_georef import FLOOR_MSL_M, lonlat_to_local
from maps.llm.http import TRANSCRIPT_JSONL, CallCanceled

log = logging.getLogger("flight.replay")

#: A pixel of a recorded observation darker than this in every band is
#: ground no frame held: the fuse leaves it black, the q90 JPEG lifts it a few
#: counts. ponytail: a level, not the fuse's own alpha, which the recording
#: does not keep.
BLACK_LEVEL = 12
#: The window a forced or frame replay emulates when the recorded log does not name it.
DEFAULT_WINDOW = 262144
#: Recorded environment flags that name paths on the aircraft, not the arm.
_PATH_FLAG = re.compile(r"_(TILES|RGBD_DIR)$")


# -- the recording -----------------------------------------------------------


def tour_stops(run_dir: Path) -> list[tuple[str, tuple[float, float]]]:
    """The flights a run flew, in order: ``(target, xy)`` off its ``flight/tour_cursor.json``."""
    try:
        cursor = json.loads((run_dir / TOUR_CURSOR).read_text())
    except (OSError, ValueError):
        return []
    return [(str(s.get("stop_id") or ""), (float(s["xy"][0]), float(s["xy"][1])))
            for s in cursor.get("stops_visited") or [] if s.get("xy")]


@dataclass
class Recording:
    """One flown run as its records hold it."""

    sortie: Path
    run_dir: Path
    row: dict[str, Any]
    params: dict[str, Any]
    #: Every reasoner request as the full message list it sent, in order.
    requests: list[list[dict[str, Any]]]
    #: The response to each request, None where none landed.
    responses: list[dict[str, Any] | None]
    conversation: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    window: int = DEFAULT_WINDOW
    detections: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.sortie.name}/run{int(self.row.get('run_no') or 0):02d}"

    @property
    def map_name(self) -> str:
        return str(self.row.get("map_name") or "calcity")

    @property
    def taskings(self) -> list[str]:
        return [str(r["said"]) for r in self.conversation if r.get("kind") == "retask"]


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every row of a recorded ``.jsonl``, none when the file is absent; a line
    that is not a JSON object is refused by file and line, since a replay
    answers from every row."""
    if not path.is_file():
        return []
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: a torn recording line ({exc})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{n}: a recording line that is not a JSON object")
        rows.append(row)
    return rows


def _messages(new: Any) -> list[dict[str, Any]]:
    # Some flown transcripts wrote the list's repr rather than the list.
    return ast.literal_eval(new) if isinstance(new, str) else list(new)


def read_transcript(path: Path) -> tuple[dict[str, Any], list[list[dict]], list[dict | None]]:
    """``(params, requests, responses)`` of a ``flight/llm/transcript.jsonl``,
    one per turn: each request rebuilt whole as ``history[:n_prefix] +
    new_messages``, a transport retry folded into the request it repeats."""
    params: dict[str, Any] = {}
    requests: list[list[dict]] = []
    ids: list[str] = []
    answers: dict[str, dict] = {}
    for rec in read_records(path):
        if rec.get("kind") == "params":
            params = rec.get("params") or {}
        elif rec.get("kind") == "request":
            prev = requests[-1] if requests else []
            requests.append(prev[: int(rec.get("n_prefix") or 0)] + _messages(rec.get("new_messages") or []))
            ids.append(str(rec.get("request_id")))
        elif rec.get("kind") == "response":
            answers[str(rec.get("request_id"))] = rec.get("response") or {}
    responses = [answers.get(i) for i in ids]
    # A request left unanswered and sent again whole is the transport's retry, one turn.
    keep = [i for i in range(len(requests))
            if not (responses[i] is None and i + 1 < len(requests) and requests[i + 1] == requests[i])]
    return params, [requests[i] for i in keep], [responses[i] for i in keep]


def open_recording(run_dir: Path) -> Recording:
    """The recording of the run in ``run_dir``, a folder under a sortie (or
    under its ``_grnd`` copy) that the sortie's ``episodes.jsonl`` names."""
    run_dir = Path(run_dir).resolve()
    sortie = sortie_of(run_dir)
    row = next((r for r in read_records(sortie / EPISODES_JSONL) if r.get("workdir") == run_dir.name), None)
    if row is None:
        raise ValueError(f"{sortie / EPISODES_JSONL} names no run {run_dir.name}")
    params, requests, responses = read_transcript(run_dir / "flight" / "llm" / TRANSCRIPT_JSONL)
    detections: dict[int, list[dict]] = {}
    for d in read_records(run_dir / DETECTIONS_JSONL):
        detections.setdefault(int(d.get("step", -1)), []).append(d)
    run_log = run_dir / RUN_LOG
    logged = re.search(r"endpoint spec installed: model_id=\S+ max_model_len=(\d+)",
                       run_log.read_text(errors="replace") if run_log.is_file() else "")
    window = int(logged.group(1)) if logged else DEFAULT_WINDOW
    return Recording(sortie=sortie, run_dir=run_dir, row=row, params=params, requests=requests,
                     responses=responses, conversation=read_records(run_dir / CONVERSATION_JSONL),
                     frames=read_records(run_dir / FRAMES_JSONL), window=window, detections=detections)


def flown_runs(sorties: Path) -> list[Path]:
    """Every run under a sorties root that holds a reasoner transcript, in the
    order flown: the aircraft's copy of its folder, else the ground station's.
    A torn ledger line skips that run, as the week bank's other records do."""
    out = []
    for ledger in sorted(Path(sorties).glob(f"*Z/{EPISODES_JSONL}")):
        for row in iter_json_rows(ledger):
            found = next((b for b in run_bases(ledger.parent, row)
                          if (b / "flight" / "llm" / TRANSCRIPT_JSONL).is_file()), None)
            if found is not None:
                out.append(found)
    return out


# -- the observation bank ------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _pixels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A recorded observation's ``(rgb, alpha)``, the alpha the pixels no frame
    held left black. ponytail: sixteen decoded at once, enough for a week's bank
    asked one area at a time."""
    from PIL import Image  # noqa: PLC0415

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return rgb, np.where(rgb.max(axis=2) > BLACK_LEVEL, 255, 0).astype(np.uint8)


@dataclass(frozen=True)
class Held:
    """One recorded observation: its pixels, its window on the map grid, and the row the run noted it with."""

    path: Path
    extent: tuple[float, float, float, float]
    camera_z: float
    frame: dict[str, Any] = field(compare=False, hash=False)

    @classmethod
    def of(cls, run_dir: Path, frame: dict[str, Any]) -> "Held | None":
        sha, corners = frame.get("observation"), frame.get("corners_xy")
        path = run_dir / OBSERVATIONS_DIR / f"{sha}.jpg" if sha else None
        if path is None or not path.is_file() or not corners:
            return None
        xs, ys = [float(c[0]) for c in corners], [float(c[1]) for c in corners]
        return cls(path=path, extent=(min(xs), min(ys), max(xs), max(ys)), camera_z=float(frame["altitude_m"]),
                   frame=frame)

    def pixels(self) -> tuple[np.ndarray, np.ndarray]:
        return _pixels(self.path)

    def holds(self, xy: tuple[float, float]) -> bool:
        x0, y0, x1, y1 = self.extent
        if not (x0 <= xy[0] < x1 and y0 <= xy[1] < y1):
            return False
        _rgb, alpha = self.pixels()
        h, w = alpha.shape
        col = min(w - 1, int((xy[0] - x0) / (x1 - x0) * w))
        row = min(h - 1, int((y1 - xy[1]) / (y1 - y0) * h))
        return bool(alpha[row, col])


class Clock:
    """The vehicle a recording stands behind: it flies nothing, and its clock moves only while the harness waits."""

    def __init__(self) -> None:
        self.t = 0.0

    def post(self, leg) -> None:
        pass

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class BankSource:
    """Recorded observations at the live source's seam. With ``forced`` (the
    mission's flight count, read live) each flight is answered by what the
    recording served it, in order. The bank keeps no clock: a read ``after``
    a time reads the whole bank."""

    def __init__(self, bank: list[Held], *, tiles_dir, map_name: str, tile_ground_z: float, forced=None,
                 by_flight: dict[int, list[Held]] | None = None) -> None:
        self.bank = bank
        self.tiles_dir, self.map_name = Path(tiles_dir), map_name
        self.tile_ground_z = float(tile_ground_z)
        self.vehicle = Clock()
        self.last: dict[str, Any] = {}
        self.last_refusal = ""
        self._forced = forced
        self._by_flight = by_flight or {}
        self._served: dict[int, int] = {}
        #: :meth:`_best` per area: the bank never changes, and arrival asks the same area every second.
        self._best_of: dict[tuple, tuple[int | None, list[bool]]] = {}

    def close(self) -> None:
        pass

    def ownship(self):
        return None

    def frame_transit_s(self):
        return None

    def _local(self, points) -> list[tuple[float, float]]:
        return [tuple(lonlat_to_local(self.tiles_dir, self.map_name, lon, lat)) for lon, lat in points]

    def _best(self, points) -> tuple[int | None, list[bool]]:
        """The bank's observation holding the most of ``points``, and which it holds."""
        if self._forced is not None:
            k = int(self._forced())
            held = bool(self._by_flight.get(k))
            return (k if held else None), [held] * len(points)
        key = tuple(tuple(p) for p in points)
        if key in self._best_of:
            return self._best_of[key]
        xy = self._local(points)
        best, best_held = None, [False] * len(points)
        for i, h in enumerate(self.bank):
            x0, y0, x1, y1 = h.extent
            if not any(x0 <= x < x1 and y0 <= y < y1 for x, y in xy):
                continue
            held = [h.holds(p) for p in xy]
            if sum(held) > sum(best_held):
                best, best_held = i, held
        self._best_of[key] = best, best_held
        return best, best_held

    def coverage(self, points, after=None) -> dict[str, Any]:
        best, held = self._best(points)
        self.last_refusal = ""
        return coverage_record([(0.0, best) if h else None for h in held], len(self.bank))

    def coverage_of_each(self, areas) -> list[dict[str, Any]]:
        return [self.coverage(a) for a in areas]

    def observe_area(self, points, after=None):
        best, held = self._best(points)
        if best is None or not any(held):
            self.last_refusal = f"no recorded observation holds any of the area's {len(points)} points"
            return None
        if self._forced is not None:
            served = self._by_flight[best]
            n = self._served.get(best, 0)
            self._served[best] = n + 1
            h = served[min(n, len(served) - 1)]
        else:
            h = self.bank[best]
        rgb, alpha = h.pixels()
        frame = h.frame
        boresight = frame.get("boresight")
        self.last = {**frame, "boresight": Path(str(boresight)).name if boresight else None,
                     "coverage": held_record(len(held), sum(held))}
        self.last_refusal = ""
        return ortho_observation(rgb, alpha, h.extent, h.camera_z, tile_ground_z=self.tile_ground_z)


def bank_of(run_dir: Path, frames) -> list[Held]:
    """The observations of the run in ``run_dir`` that ``frames`` (its ``frames.jsonl`` rows) name."""
    return [h for h in (Held.of(run_dir, f) for f in frames) if h is not None]


# -- forced: the recorded reasoner and segmenter ----------------------------


class Script:
    """The recorded reasoner: each request gets its recorded response, the
    next request's operator turns are spoken as it goes back, and past the
    last answered request the mission is stopped, as the crew stopped it."""

    def __init__(self, rec: Recording) -> None:
        self.rec = rec
        self.mission: Mission | None = None
        self.asked = 0
        self._taskings = list(rec.taskings)
        self.ended = False

    def _user_turns(self, k: int) -> int:
        if k <= 0 or k >= len(self.rec.requests):
            return 0
        # ponytail: the turns past the request before and its answer; a run the
        # compactor folded between them would miscount (none of the flown runs).
        before = len(self.rec.requests[k - 1]) + 1
        # An operator's turn is words alone; a picture rides a user turn as a
        # list, and a protocol nudge is the harness's own ``[Repair]`` turn.
        return sum(1 for m in self.rec.requests[k][before:]
                   if m.get("role") == "user" and isinstance(m.get("content"), str)
                   and not m["content"].startswith("[Repair]"))

    def __call__(self, url, headers, body, *, timeout, cancel_event):
        k, self.asked = self.asked, self.asked + 1
        answer = self.rec.responses[k] if k < len(self.rec.responses) else None
        if answer is None:
            self.ended = True
            self.mission.stop()
            raise CallCanceled("the recording holds no answer to this request")
        for _ in range(self._user_turns(k + 1)):
            if self._taskings:
                self.mission.tasking(self._taskings.pop(0))
        if k + 1 >= len(self.rec.requests):
            self.ended = True
        return copy.deepcopy(answer)


def recorded_segmenter(rec: Recording, state_of):
    """The segmenter's ``segment`` answered with the flight's recorded masks,
    through the client's own decode, split, projection and colour read."""

    def segment(client, obs, concepts):
        from maps.perception.sam.detect_client import record_region  # noqa: PLC0415

        rows = rec.detections.get(int(state_of().step_idx), [])
        hw = obs.rgb.shape[:2]
        return client.detections(obs, [
            (record_region(d, hw), d["label"], int((d.get("metadata") or {}).get("sam3_1_object_id", i)),
             float((d.get("metadata") or {}).get("sam3_1_score", d.get("confidence"))))
            for i, d in enumerate(rows) if d.get("mask_zpacked_b64")])

    return segment


class ReplayMission(Mission):
    """The harness's mission, its row naming the run it replays. Forced, a
    pass takes its stops in the order the recording flew them."""

    def __init__(self, *, replay_of: dict[str, Any], stops: list[tuple[str, tuple[float, float]]] | None = None,
                 **kw) -> None:
        super().__init__(**kw)
        self.replay_of = replay_of
        self._recorded_stops = stops

    def row(self) -> dict[str, Any]:
        """The mission's row naming the run it replays."""
        return {**super().row(), "replay_of": self.replay_of}

    def _utc_now(self) -> None:
        """No UTC start, and so no track: the replay's clock is not the flight's."""
        return None

    def _held_now(self, h, stops: list) -> list[bool]:
        if self._recorded_stops is None:
            return super()._held_now(h, stops)
        k = len(self.flights)
        if k >= len(self._recorded_stops):
            return [False] * len(stops)
        # A point stop within 1 m of a row was flown, and recorded, as that row's fid (the engine's snap).
        nxt, nxt_xy = self._recorded_stops[k]
        return [str(token) == nxt or (xy is not None and math.dist(xy, nxt_xy) <= 1.0) for token, xy in stops]

    def _await_tasking(self) -> str | None:
        """Forced, the recorded reasoner queues an operator's words before the
        report they answer, so a report with none queued ends the replay."""
        if self._recorded_stops is None:
            return super()._await_tasking()
        return self._next_tasking()


# -- the diff ------------------------------------------------------------------


def _text(message: dict[str, Any]) -> tuple[str, list[str]]:
    """A message as ``(role + text, image shas)``: the words exactly, each picture a mark."""
    content = message.get("content")
    shas: list[str] = []
    if isinstance(content, list):
        parts = []
        for p in content:
            if p.get("type") == "text":
                parts.append(str(p.get("text")))
            else:
                shas.append(str(p.get("sha256") or ""))
                parts.append("<image>")
        content = "\n".join(parts)
    calls = json.dumps(message.get("tool_calls"), sort_keys=True) if message.get("tool_calls") else ""
    return f"[{message.get('role')}] {content or ''}{calls}", shas


def diff(rec: Recording, replay_dir: Path) -> dict[str, Any]:
    """The replay against the recording: the params, every message of every
    request (``first`` the first that differs) and the flights, the last of
    them the commit."""
    params, requests, answers = read_transcript(replay_dir / "flight" / "llm" / TRANSCRIPT_JSONL)
    # A request past the recording's last, left unanswered, is where the crew's stop fell.
    extra = requests[len(rec.requests):]
    stopped_there = len(extra) <= 1 and all(a is None for a in answers[len(rec.requests):])
    out: dict[str, Any] = {"recorded": rec.run_id, "requests": [len(rec.requests), len(requests)],
                           "params": sorted(k for k in set(params) | set(rec.params)
                                            if params.get(k) != rec.params.get(k)),
                           "stopped_there": stopped_there,
                           "turns": [], "first": None, "pictures_differ": 0}
    for i, (a, b) in enumerate(zip(rec.requests, requests)):
        ta, tb = [_text(m) for m in a], [_text(m) for m in b]
        out["pictures_differ"] += sum(1 for (_x, sa), (_y, sb) in zip(ta, tb) if sa != sb)
        if [t for t, _s in ta] == [t for t, _s in tb]:
            out["turns"].append({"request": i, "same": True})
            continue
        j = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x[0] != y[0]), min(len(ta), len(tb)))
        x = ta[j][0] if j < len(ta) else ""
        y = tb[j][0] if j < len(tb) else ""
        lines = list(difflib.unified_diff(x.splitlines(), y.splitlines(), "recorded", "replay", lineterm="", n=1))
        turn = {"request": i, "same": False, "message": j, "role": (a[j] if j < len(a) else b[j]).get("role"),
                "lengths": [len(a), len(b)], "diff": lines[:60]}
        out["turns"].append(turn)
        if out["first"] is None:
            out["first"] = turn
    flown, replayed = tour_stops(rec.run_dir), tour_stops(replay_dir)
    same = len(flown) == len(replayed) and all(
        fa == fb and abs(xa[0] - xb[0]) < 0.5 and abs(xa[1] - xb[1]) < 0.5
        for (fa, xa), (fb, xb) in zip(flown, replayed))
    out["flights"] = {"same": same, "recorded": [f for f, _ in flown], "replay": [f for f, _ in replayed]}
    out["reproduced"] = (out["first"] is None and not out["params"] and out["flights"]["same"]
                         and len(requests) >= len(rec.requests) and stopped_there)
    return out


# -- the driver ----------------------------------------------------------------


#: The recorded reasoner's own settings: a forced replay speaks as it did, any
#: other replay uses the reasoner the caller brings.
_REASONER_FLAGS = ("SEARCH_COMPACT_TOKENIZER_NAME", "SEARCH_SAMPLING_REASONING_EFFORT")


def _arm_env(rec: Recording, *, forced: bool) -> None:
    """The recorded arm's environment flags, the aircraft's paths excepted."""
    for k, v in (rec.row.get("search_env_flags") or {}).items():
        if _PATH_FLAG.search(k) or v is None or (k in _REASONER_FLAGS and not forced):
            continue
        os.environ[k] = str(v)


def replay_agl_m(rec: Recording, *, forced: bool, config: str) -> float:
    """The height above ground a replay flies at, the one the served sensor
    block and every leg read. Forced, the config's start height, where the
    flown engine was hydrated; else the recorded run's: the median camera
    height of its observations, else of its GNSS track over the run."""
    if forced:
        from benchmarks.citynav.run_benchmark.config import load_config  # noqa: PLC0415

        return float(load_config(config).start_altitude_agl_m)
    map_name = rec.map_name
    alts = [frame_agl(f, map_name) for f in rec.frames if f.get("altitude_m") is not None]
    if not alts and rec.row.get("started_at_utc"):
        t0 = float(rec.row["started_at_utc"])
        t1 = t0 + float(rec.row.get("elapsed_s") or 0.0)
        alts = [float(f["alt_msl_m"]) - FLOOR_MSL_M[map_name] for f in read_records(rec.sortie / TRACK_JSONL)
                if f.get("alt_msl_m") is not None and t0 <= float(f["t_unix"]) <= t1]
    if not alts:
        raise ValueError(f"{rec.run_id}: the recording holds no height above ground: no observation and no GNSS "
                         f"fix over the run")
    return float(np.median(alts))


def fly(rec: Recording, out: Path, *, forced: bool = False, bank: list[Held] | None = None, live=None,
        config: str = "configs/calcity.yaml", taskings: list[str] | None = None,
        reasoning_effort: str | None = None) -> tuple[ReplayMission, Path]:
    """Fly ``rec`` again under ``out/<sortie>/``; ``live`` is a live source (the
    frame tier), else the bank answers. Returns the mission, ended, and its folder."""
    _arm_env(rec, forced=forced)
    benchmarks.calcity.use()
    flight_config(config)
    from benchmarks.citynav.episode_workdir import open_episode_dir  # noqa: PLC0415
    from benchmarks.citynav.paths import DEFAULT_SPLIT, TILES_DIR  # noqa: PLC0415
    from maps.perception.ortho.ground_level import GROUND_LEVEL  # noqa: PLC0415

    row = rec.row
    episode_id, map_name = str(row["episode_id"]), rec.map_name
    script = None
    if forced:
        from benchmarks.citynav.run_benchmark import episode_runner  # noqa: PLC0415
        from maps.llm import http  # noqa: PLC0415

        from huggingface_hub import constants as hub  # noqa: PLC0415

        # The tokenizer comes off the local cache: the hub read its flag at import.
        hub.HF_HUB_OFFLINE = True
        os.environ.update({"SEARCH_ENDPOINT_OFFLINE": "1",
                           "SEARCH_ENDPOINT_SPEC_OVERRIDE": f"{rec.params.get('model')}:{rec.window}"})
        if rec.params.get("seed") is not None:
            os.environ["SEARCH_SEED_BASE"] = str(rec.params["seed"])
        script = Script(rec)
        http.stream_chat_completions = script
        make = episode_runner.make_perception

        def recorded():
            adapter = make()
            adapter.segment = functools.partial(recorded_segmenter(rec, lambda: mission.hydrated.state), adapter)
            return adapter

        episode_runner.make_perception = recorded
    frames = live is not None
    if not frames:
        check_map(TILES_DIR, map_name)
        by_flight: dict[int, list[Held]] = {}
        if forced:
            for h in bank_of(rec.run_dir, rec.frames):
                by_flight.setdefault(int(h.frame.get("flight") or 0), []).append(h)
        live = BankSource(bank or [], tiles_dir=TILES_DIR, map_name=map_name,
                          tile_ground_z=float(GROUND_LEVEL[map_name]), by_flight=by_flight,
                          forced=(lambda: len(mission.flights)) if forced else None)
    sortie = Path(out) / rec.sortie.name
    run_no = int(row.get("run_no") or 1)
    ep = open_episode_dir(sortie, episode_id=episode_id, run_no=run_no)
    # The recorded effort stands only where the recorded reasoner answers;
    # the bank and frame tiers fly the caller's arm.
    effort = (rec.params.get("reasoning_effort") or row.get("reasoning_effort")) if forced else reasoning_effort
    mission = ReplayMission(
        replay_of={"sortie": rec.sortie.name, "run_no": run_no, "run_id": row.get("run_id"),
                   "workdir": rec.run_dir.name,
                   "mode": " ".join(m for m, on in (("forced", forced), ("frames", frames)) if on) or "bank"},
        stops=tour_stops(rec.run_dir) if forced else None, episode_id=episode_id, split=str(row.get("split") or DEFAULT_SPLIT),
        config=config, workdir=ep.dir, live=live, instruction=str(row.get("description") or ""),
        recorder=ep.recorder, close_log=ep.close, run_id=f"replay-{rec.sortie.name}", run_no=run_no,
        start_xy=tuple(row["start_xy"]), reasoning_effort=None if effort is None else str(effort),
        agl_m=replay_agl_m(rec, forced=forced, config=config))
    if script is not None:
        script.mission = mission
    converse(mission, taskings or [], ended=None if script is None else (lambda: script.ended))
    return mission, ep.dir


def _say(d: dict[str, Any]) -> str:
    if d["reproduced"]:
        return f"{d['recorded']}: reproduced turn for turn ({d['requests'][0]} requests)"
    first = d["first"]
    bits = [f"{d['recorded']}: DIVERGES"]
    if d["params"]:
        bits.append(f"params differ: {', '.join(d['params'])}")
    if first is not None:
        bits.append(f"first at request {first['request']} message {first['message']} ({first['role']})")
        bits.extend(first["diff"][:20])
    if d["requests"][0] != d["requests"][1]:
        bits.append(f"requests recorded {d['requests'][0]}, replayed {d['requests'][1]}")
    if not d["flights"]["same"]:
        bits.append(f"flights recorded {d['flights']['recorded']}, replayed {d['flights']['replay']}")
    return "\n".join(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="a flown run's folder under its sortie")
    ap.add_argument("--forced", action="store_true",
                    help="replay the recorded reasoner and segmenter and diff every request; with --frames, the "
                         "observations are re-fused from the camera recording")
    ap.add_argument("--bank", choices=("run", "week"), default="run",
                    help="the observations a flight is answered from: the run's own, or every run under the root")
    ap.add_argument("--frames", type=Path, default=None, help="the camera recording: re-fuse every waypoint from it")
    ap.add_argument("--nav", type=Path, nargs="*", default=None,
                    help="the recordings holding the fixes and attitude (default: the camera recording)")
    ap.add_argument("--boresight", type=Path, default=None)
    ap.add_argument("--raster", type=Path, default=None, help="the raster directory frames are registered against")
    ap.add_argument("--config", default="configs/calcity.yaml")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--out", type=Path, default=Path("output/replay"))
    args = ap.parse_args(argv)
    if args.bank == "week" and (args.forced or args.frames is not None):
        ap.error("--bank week chooses the bank tier's observations; --forced and --frames answer from the recording")
    if args.forced and args.reasoning_effort is not None:
        ap.error("--reasoning-effort is your reasoner's; --forced sends the recorded one")
    if args.frames is None and (args.raster or args.boresight or args.nav is not None):
        ap.error("--raster, --boresight and --nav belong to the frame tier: give --frames")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    rec = open_recording(args.run)
    if not rec.row.get("start_xy"):
        ap.error(f"{rec.run_id} records no start: the run never reached the reasoner")
    # The frame tier's inputs, checked before the camera recording is read.
    if args.frames is not None and not args.frames.is_file():
        ap.error(f"--frames {args.frames}: no such recording")
    if args.boresight is not None and not args.boresight.is_file():
        ap.error(f"--boresight {args.boresight}: no such file")
    if args.raster is not None and not map_render.holds_raster(args.raster, rec.map_name):
        ap.error(f"--raster {args.raster}: no {rec.map_name} raster there; a raster directory holds "
                 f"{map_render.raster_pair(rec.map_name)}")
    out = args.out / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    live = None
    bank: list[Held] | None = None
    if args.frames is not None:
        benchmarks.calcity.use()
        from benchmarks.citynav.paths import TILES_DIR  # noqa: PLC0415
        from benchmarks.flight.recorded_ring import RecordedRing, run_window  # noqa: PLC0415
        from benchmarks.citynav.run_benchmark.config import load_config  # noqa: PLC0415
        from maps.perception.geometry.sensor_spec import resolve_sensor  # noqa: PLC0415

        t0 = time.perf_counter()
        start_ns, end_ns = run_window(rec.row, rec.conversation)
        ring = RecordedRing(args.frames, nav=args.nav, start_ns=start_ns, end_ns=end_ns,
                            clock_ns=int(float(rec.row["started_at_utc"]) * 1e9))
        log.info("the ring over %s: %d frames kept, %.1f s to read", args.frames.name, len(ring.rows),
                 time.perf_counter() - t0)
        live = open_live(vehicle=ring, tiles_dir=TILES_DIR, map_name=rec.map_name,
                         sensor=resolve_sensor(load_config(args.config).sensor), boresight_path=args.boresight,
                         register_raster=args.raster)
    if args.forced:
        mission, run = fly(rec, out, forced=True, live=live, config=args.config)
        d = diff(rec, run)
        (run / "replay_diff.json").write_text(json.dumps(d, indent=1) + "\n")
        print(_say(d))
        return 0 if d["reproduced"] else 1
    if live is None:
        if args.bank == "run":
            runs, bank = 1, bank_of(rec.run_dir, rec.frames)
        else:
            # The other runs serve only their observations: a torn line there is skipped, not refused.
            week = flown_runs(rec.sortie.parent)
            runs, bank = len(week), [h for p in week for h in bank_of(p, iter_json_rows(p / FRAMES_JSONL))]
        log.info("observation bank: %d observations from %d run(s)", len(bank), runs)
    mission, run = fly(rec, out, bank=bank, live=live, config=args.config, taskings=rec.taskings,
                       reasoning_effort=args.reasoning_effort)
    print(summary_line(mission))
    return 0 if mission.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
