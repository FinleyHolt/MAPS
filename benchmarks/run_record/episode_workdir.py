"""An episode's directory: the function that opens it, its layout, and the two writers that
fill it as the episode runs.

    <root>/maps_<YYYYmmdd_HHMMSS>[_run<NN>]_<benchmark>_<slug>/  one ATTEMPT of one episode
        episode.json       the runner's row once the attempt lands: ok | infra
        run.log            this process's log while the episode runs
        conversation.jsonl the mission's event trace
        detections.jsonl   one line per detection, stamped with its step
        frames/            every jpeg the model was shown, by sha256
        flight/            the engine's working files: llm/transcript.jsonl (the wire
                           log), look/ (the look audit), tour_cursor.json
        frames.jsonl       a flight mission's: one row per observation it perceived
        observations/      a flight mission's: each observation's fused pass, <sha>.jpg
    <sortie>/flight_track.jsonl  a flight sortie's ownship track, one fix a second

The runner and the flight mission open their attempt directories through
``open_episode_dir``. The stamp is the attempt's start, which the readers
order attempts by, so a retry is a new directory.

``ConversationLog`` is the transcript list the loop appends to (a row per
turn and the reasons a turn was not a block); every ``append`` also lands the
row in ``conversation.jsonl``, opened per append, so a mission killed between
turns keeps every turn it flew. ``LiveRecorder`` appends one line per
detection the step's perception fold produced (``maps.mission.episode_world``'s
record, masks packed) plus ``step`` to ``detections.jsonl``, flushed per line
so a reader sees at most one torn final line; the file exists from the start.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from benchmarks.run_record.episodes import RUN_LOG, json_default

CONVERSATION_JSONL = "conversation.jsonl"
DETECTIONS_JSONL = "detections.jsonl"
#: One row per observation a flight mission perceived: the key from the
#: mission's reasoning back into the recording.
FRAMES_JSONL = "frames.jsonl"
#: Where each observation goes: the pass fused aboard as one north-up image,
#: ``<sha>.jpg``. Never ``frames/``, which is what the model saw.
OBSERVATIONS_DIR = "observations"
#: The ownship track, in the sortie's folder: the path
#: :mod:`benchmarks.flight.measures` reads OSR and SPL off.
TRACK_JSONL = "flight_track.jsonl"


#: The benchmarks a directory name may claim: the corpus the episode came from.
BENCHMARKS = ("citynav", "calcity")

#: Every attempt directory this tree mints starts here.
WORKDIR_PREFIX = "maps"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def mint_workdir(root: Path, episode_id: str, run_no: int | None = None, *, benchmark: str) -> Path:
    """``<root>/maps_<stamp>[_run<NN>]_<benchmark>_<slug>``, this attempt's
    directory (not created). ``benchmark`` is the bench's name
    (``Bench.name``), one of ``BENCHMARKS``; there is no default. A flight
    mission carries its run number in the sortie, the link to its ledger row;
    a bench attempt carries none.
    """
    if benchmark not in BENCHMARKS:
        raise RuntimeError(
            f"{benchmark!r} names no benchmark: an episode directory is named "
            f"for its benchmark, one of {', '.join(BENCHMARKS)}."
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"_run{run_no:02d}" if run_no is not None else ""
    return Path(root) / (f"{WORKDIR_PREFIX}_{stamp}{run}_{benchmark}_{_slug(episode_id)[:40]}")


class ConversationLog(list):
    """A transcript list whose every append also lands on disk; ``path=None``
    is a plain list."""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = Path(path) if path is not None else None

    def append(self, row: dict) -> None:  # type: ignore[override]
        super().append(row)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()


@dataclass
class LiveRecorder:
    """What the drone detected each step, one recorder per episode directory."""

    episode_dir: Path

    def __post_init__(self) -> None:
        self.episode_dir = Path(self.episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        # A mission that detects nothing has zero rows: an empty file says so.
        (self.episode_dir / DETECTIONS_JSONL).touch()

    def write_step(self, *, step: int, detections: list[dict[str, Any]]) -> None:
        for det in detections:
            line = json.dumps({**det, "step": int(step)}, default=json_default) + "\n"
            try:
                with (self.episode_dir / DETECTIONS_JSONL).open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
            except OSError:
                pass


@dataclass
class OpenEpisode:
    dir: Path
    recorder: LiveRecorder
    close: Callable[[], None]


def open_episode_dir(
    root: Path | str,
    *,
    episode_id: str,
    benchmark: str,
    run_no: int | None = None,
) -> OpenEpisode:
    """Mint the attempt's directory under ``root`` (:func:`mint_workdir`) and
    send this process's log into ``<dir>/run.log``. ``close()`` detaches the
    log handler, so a batch's next episode logs elsewhere. The caller hands
    ``dir`` to the engine as the mission's workdir: the wire log and the
    frames shown land under it.
    """
    d = mint_workdir(Path(root), episode_id, run_no, benchmark=benchmark)
    flight = d / "flight"
    flight.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(d / RUN_LOG, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)

    recorder = LiveRecorder(d)

    def close() -> None:
        logging.getLogger().removeHandler(handler)
        try:
            handler.close()
        except OSError:
            pass

    return OpenEpisode(dir=d, recorder=recorder, close=close)
