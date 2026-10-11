"""One function opens an episode's directory; everything the episode writes lands under it.

    <root>/maps_<YYYYmmdd_HHMMSS>[_run<NN>]_<benchmark>_<slug>/  one ATTEMPT of one episode
        episode.json       the runner's row once the attempt lands: ok | infra
        run.log            this process's log while the episode runs
        conversation.jsonl the mission's event trace
        detections.jsonl   one line per detection, stamped with its step
        frames/            every jpeg the model was shown, by sha256
        flight/            the engine's working files: llm/transcript.jsonl (the wire
                           log), look/ (the look audit), tour_cursor.json

The runner and the field station open their attempt directories through
``open_episode_dir``. The stamp is the attempt's start, which the readers
order attempts by, so a retry is a new directory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("run_benchmark")

RUN_LOG = "run.log"


#: The benchmarks a directory name may claim: the corpus the episode came from.
BENCHMARKS = ("citynav", "calcity")

#: Every attempt directory this tree mints starts here.
WORKDIR_PREFIX = "maps"


def benchmark() -> str:
    """Which benchmark this process flies: ``SEARCH_BENCHMARK_NAME``, one of
    ``BENCHMARKS``, read at call time. There is no default: the bench CLIs and
    the station declare it, and a process that has not raises.
    """
    name = os.environ.get("SEARCH_BENCHMARK_NAME", "").strip()
    if name not in BENCHMARKS:
        raise RuntimeError(
            f"SEARCH_BENCHMARK_NAME is {name or 'unset'!r}: an episode "
            f"directory is named for its benchmark, one of "
            f"{', '.join(BENCHMARKS)}. Declare it where the benchmark is "
            f"known -- the bench CLI's own package, or the station's site."
        )
    return name


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def mint_workdir(root: Path, episode_id: str, run_no: int | None = None) -> Path:
    """``<root>/maps_<stamp>[_run<NN>]_<benchmark>_<slug>``, this attempt's
    directory (not created). A station flight carries its run number in the
    sortie, the link to its ledger row; a bench attempt carries none.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"_run{run_no:02d}" if run_no is not None else ""
    return Path(root) / (
        f"{WORKDIR_PREFIX}_{stamp}{run}_{benchmark()}_{_slug(episode_id)[:40]}"
    )


@dataclass
class OpenEpisode:
    dir: Path
    recorder: Any
    close: Callable[[], None]


def open_episode_dir(
    root: Path | str,
    *,
    episode_id: str,
    run_no: int | None = None,
) -> OpenEpisode:
    """Mint the attempt's directory under ``root`` (:func:`mint_workdir`) and
    point this process at it, its log into ``<dir>/run.log``. ``close()``
    detaches the log handler, so a batch's next episode logs elsewhere.
    """
    # numpy at import; the driver imports this module and must stay light.
    from benchmarks.citynav.live_recorder import LiveRecorder

    d = mint_workdir(Path(root), episode_id, run_no)
    flight = d / "flight"
    flight.mkdir(parents=True, exist_ok=True)

    # Assigned, never setdefault: a batch's previous episode left a value,
    # and the wire log resolves its directory from it at every call.
    os.environ["SEARCH_COMPILE_WORKDIR"] = str(d)
    # Router-affinity key: a key-affine router keeps the episode on one worker.
    os.environ["SEARCH_LLM_ROUTING_KEY"] = episode_id

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
