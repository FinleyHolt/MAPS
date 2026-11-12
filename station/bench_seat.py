"""One bench episode in the seat: flown by the bench's own runner, held open for the operator.

The episode is dispatched exactly as the bench's runner dispatches one
(``EpisodeEngine.fly_attempt``: the bench's episode build, its mission
session, its six-action replay, its scorer, its ``episode.json``), on a worker
thread. Flights are the bench's: the aircraft is placed at each stop and the
raster cropped there. One thing differs, through the runner's ``converse``
seam: the model's reply is a REPORT. The mission parks in
``awaiting_tasking`` and the operator's answer is one more turn of the same
conversation (``maps.mission.loop.continue_mission``); a tasking sent while
the model flies lands on its next turn. The episode is scored when the
operator ends the mission, on every flight flown by then, and its row lands
in ``<sortie>/episodes.jsonl`` as a bench run's does.

ponytail: the row lands when the mission ends, so a station killed while a
mission holds at its report leaves the attempt folder and no row. Scoring at
each report needs the scorer split out of ``run_episode``.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks import git_provenance
from benchmarks.bench import Bench
from benchmarks.run_record import rows
from benchmarks.run_record.config import load_config
from benchmarks.run_record.episode_workdir import (
    CONVERSATION_JSONL,
    ConversationLog,
    OpenEpisode,
)
from benchmarks.run_record.episodes import STATUS_INFRA, append_row
from benchmarks.run_record.run_manifest import (
    build_manifest,
    emit_or_verify,
    install_endpoint,
)
from maps.llm.http import CallCanceled
from maps.mission.spec import MissionReply
from maps.settings import from_environ

log = logging.getLogger("station.bench")

#: How long ending a mission waits for its row to land before answering anyway.
END_WAIT_S = 60.0


class BenchMission:
    """A bench episode in the seat: its thread, its tasking queue, and the
    fields the station's state document reads (the twin ``Mission``'s names)."""

    kind = "bench"
    run_no = None
    #: A bench flight is a teleport: no leg is ever waited on.
    waiting = None

    def __init__(
        self,
        *,
        bench: Bench,
        episode: Any,
        split: str,
        config: str,
        objects_by_map: dict,
        sortie: Path,
        ep: OpenEpisode,
        reasoning_effort: str | None = None,
    ) -> None:
        # The operator's effort is the arm's: the requests carry it and the row's flags record it.
        effort = (reasoning_effort or "").strip()
        self.bench = (
            bench.with_environ({"SEARCH_SAMPLING_REASONING_EFFORT": effort}) if effort else bench
        )
        self.episode, self.split, self.config = episode, split, config
        self.episode_id, self.map_name, self.instruction = (
            episode.episode_id,
            episode.map_name,
            episode.description,
        )
        self._objects = objects_by_map
        self.sortie, self.workdir, self._ep = Path(sortie), ep.dir, ep
        self.status = "flying"
        self.ended_by: str | None = None
        self.error: str | None = None
        #: The scored row, once the mission has ended and the bench has scored it.
        self.row: dict[str, Any] | None = None
        self.started_at = time.time()
        self.transcript: list[dict] = ConversationLog(self.workdir / CONVERSATION_JSONL)
        self._session: Any = None
        self._turns = 0
        self._turn_diag: dict[str, Any] = {}
        self._taskings: queue.Queue[str] = queue.Queue()
        self._cancel = threading.Event()
        prov = git_provenance()
        self.git_commit, self.git_dirty = prov.get("git_commit"), prov.get("git_dirty")
        self._thread = threading.Thread(
            target=self._run, name=f"bench-{self.episode_id}", daemon=True
        )

    # -- what the state document reads ---------------------------------------

    @property
    def flights(self) -> list[dict]:
        return self._session.flights if self._session is not None else []

    @property
    def score(self) -> dict | None:
        return rows.metrics(self.row) or None if self.row else None

    def turns(self) -> int:
        return self._turns + int(self._turn_diag.get("turns_used") or 0)

    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def pose(self) -> tuple[float, float, float] | None:
        """``(x, y, height above ground)`` of the aircraft now, or None before the episode is built."""
        if self._session is None:
            return None
        h = self._session.h
        return (
            float(h.state.pose_xy[0]),
            float(h.state.pose_xy[1]),
            float(h.state.altitude_m) - float(h.ground_z_m),
        )

    # -- the operator's channel ----------------------------------------------

    def tasking(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.transcript.append({"kind": "retask", "said": text})
            self._taskings.put(text)

    def _next_tasking(self) -> str | None:
        try:
            return self._taskings.get_nowait()
        except queue.Empty:
            return None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """End the mission: a turn in flight is cancelled, a mission at its report is released to be scored."""
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    # -- the episode ----------------------------------------------------------

    def _converse(self, session):
        """The conversation over the bench's session (``run_episode``'s
        ``converse``): each reply parks the mission until the operator speaks
        or ends it. Returns how the last conversation ended."""
        from maps.mission.loop import continue_mission  # noqa: PLC0415  (the mission stack, with its tokenizer)

        self._session = session
        try:
            reply, conversation = session.run(
                session.h.episode.description,
                transcript=self.transcript,
                cancel_event=self._cancel,
                retask=self._next_tasking,
                diag=self._turn_diag,
            )
            for turn in range(2, 1 << 30):
                self._turns += int(reply.turns_used)
                self._turn_diag.clear()
                self.status = "awaiting_tasking"
                while (nxt := self._next_tasking()) is None and not self._cancel.wait(0.2):
                    pass
                if nxt is None:
                    return reply
                self.status = "flying"
                reply = continue_mission(conversation, nxt, rid_tag=f"m{turn}")
        except CallCanceled:
            return MissionReply(
                ended_by="stopped", turns_used=int(self._turn_diag.get("turns_used") or 0)
            )

    def _run(self) -> None:
        try:
            cfg = load_config(self.config)
            settings = from_environ(self.bench.environ)
            bench = install_endpoint(self.bench, settings)
            if "SEARCH_BENCHMARK_NAME" not in bench.environ:
                bench = bench.with_environ({"SEARCH_BENCHMARK_NAME": bench.name})
            emit_or_verify(
                self.sortie,
                build_manifest(
                    bench=bench,
                    out_root=self.sortie,
                    config_path=Path(self.config),
                    split=self.split,
                    sample_seed=0,
                    limit=0,
                    allowlist_path=None,
                    episodes=[self.episode_id],
                ),
                log=log.info,
            )
            row = bench.engine.fly_attempt(
                episode=self.episode,
                objects_by_map=self._objects,
                cfg=cfg,
                bench=bench,
                settings=settings,
                split=self.split,
                recorder=self._ep.recorder,
                episode_dir=self.workdir,
                routing_key=self.episode_id,
                converse=self._converse,
            )
            row["workdir"] = str(self.workdir.relative_to(self.sortie))
            append_row(self.sortie, row)
            self.row, self.ended_by, self.error = row, rows.ended_by(row), row.get("error")
            self.status = (
                "failed" if row.get("status") == STATUS_INFRA or not rows.metrics(row) else "ended"
            )
        except Exception as exc:  # noqa: BLE001 -- the operator must see why
            log.exception("mission %s failed", self.episode_id)
            self.error, self.status = f"{type(exc).__name__}: {exc}", "failed"
        finally:
            self._ep.close()


def with_words(episode: Any, instruction: str) -> Any:
    """``episode`` flown under the operator's edit of its words: the row's start and target, scored as the row is."""
    words = (instruction or "").strip()
    return (
        dataclasses.replace(episode, description=words)
        if words and words != episode.description
        else episode
    )
