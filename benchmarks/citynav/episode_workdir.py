"""One function opens an episode's directory; everything the episode writes lands under it.

    <root>/maps_<YYYYmmdd_HHMMSS>[_run<NN>]_<benchmark>_<slug>/  one ATTEMPT of one episode
        episode.json       the runner's row once the attempt lands: ok | infra
        run.log            this process's log while the episode runs
        map_prior.json     the survey features the episode was handed, byte-for-byte
        conversation.jsonl the mission's event trace
        steps.jsonl        one line per step (``live_recorder``)
        detections.jsonl   one line per detection, stamped with its step
        frames/            every jpeg the model was shown, by sha256
        flight/            the engine's working files: llm/transcript.jsonl (the wire
                           log), look/ (the look audit), net_meter.jsonl, tour_cursor.json

The launcher (``launch.spawn_runner``), the warm runner (``episode_server``)
and the batch loop (``cli.main``) each carried their own copy of this setup
and had drifted: one ``setdefault`` the env where another assigned it, one
attached a log handler where another attached two. One function, three
callers, and the field station is the fourth.

The stamp in the name is the attempt's start; the readers order attempts of
one episode by it, so a retry is a new directory, never an overwrite. A
station flight also carries its run number there, which is what its ledger
row and both cabin screens name it by.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("run_benchmark")

RUN_LOG = "run.log"
MAP_PRIOR_JSON = "map_prior.json"


#: The benchmarks a directory name may claim. One word each, and the word is
#: the corpus the episode came from -- not the engine that flew it: Stanford
#: flies the CityNav engine over its own bake and is its own test program.
BENCHMARKS = ("citynav", "calcity", "stanford")

#: Every attempt directory this tree mints starts here, whichever benchmark
#: flew it. The name says MAPS ran; the benchmark word says what it ran.
WORKDIR_PREFIX = "maps"


def benchmark() -> str:
    """Which benchmark this process flies, one of ``BENCHMARKS``.

    ``SEARCH_BENCHMARK_NAME``, read at CALL time so an entry point that
    declares it after this module is imported still wins -- the Cal City
    bridge and the field station both do.

    THERE IS NO DEFAULT. The declaration is made by whoever knows: the two
    bench CLIs from their own package, the station from the site the
    aircraft's fix resolved (a wrong site draws the wrong map and fails on
    the first frame, so it cannot go stale unnoticed). Defaulting to
    ``citynav`` is what named a Cal City sortie ``citynav_..._calcity__8_0``
    and put the wrong benchmark in front of an operator browsing the sortie
    tree; a process that mints an episode directory without saying what it
    is flying raises instead.
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
    directory (not created).

    A STATION flight carries the crew's run number. That is the link between
    a directory and the row that describes it: a sortie's ledger numbers its
    runs from 1 and both cabin screens say the number out loud, so ``ls |
    grep run03`` is how a person browsing the tree reaches the flight,
    instead of matching timestamps by eye. A bench attempt carries none --
    there the run IS the root the attempt sits in.

    The stamp stays first: the readers order attempts of one episode by it
    (``benchmarks.citynav.analysis.rows``), and so does ``ls``.
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
    map_name: str,
    workdir: Path | str | None = None,
    run_log: bool = True,
    run_no: int | None = None,
) -> OpenEpisode:
    """Create the attempt's directory and point this process at it.

    ``workdir`` names a directory a launcher already minted (it wrote the
    in-flight handle for it before spawning); otherwise one is minted under
    ``root``. ``run_log=False`` is for that launched child, whose stdout and
    stderr the launcher already redirects into ``<workdir>/run.log``: a
    second handler would print every line twice.

    ``run_no`` is the run's number in its sortie; it rides the directory name
    so a folder and a run row find each other (:func:`mint_workdir`). A bench
    attempt carries none.

    ``close()`` detaches the log handler, so a warm runner's next episode
    does not bleed into this one's file.
    """
    # numpy at import; the launcher imports this module and must stay light.
    from benchmarks.citynav.live_recorder import LiveRecorder

    d = Path(workdir) if workdir is not None else mint_workdir(Path(root), episode_id, run_no)
    flight = d / "flight"
    flight.mkdir(parents=True, exist_ok=True)

    # ASSIGNED, never setdefault: a warm runner's previous episode left a
    # value here, and the wire log resolves its directory from
    # SEARCH_COMPILE_WORKDIR at every call.
    os.environ["SEARCH_COMPILE_WORKDIR"] = str(d)
    os.environ["NET_METER_LOG"] = str(flight / "net_meter.jsonl")
    # Router-affinity key: every LLM call of this episode carries it, so a
    # key-affine router keeps the episode on one worker.
    os.environ["SEARCH_LLM_ROUTING_KEY"] = episode_id

    handler: logging.FileHandler | None = None
    if run_log:
        handler = logging.FileHandler(d / RUN_LOG, mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s %(name)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(handler)

    _copy_map_prior(d, map_name)

    recorder = LiveRecorder(d, episode_id=episode_id)

    def close() -> None:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            try:
                handler.close()
            except OSError:
                pass

    return OpenEpisode(dir=d, recorder=recorder, close=close)


def _copy_map_prior(d: Path, map_name: str) -> None:
    """``map_prior.json`` is the block's ``_features.json`` from the tile bake,
    copied byte-for-byte: what the episode was handed, hashing to the bake
    that ``run_manifest.perception.bake_sha256`` names."""
    # At call time: the Cal City bridge rebinds these names after import.
    from benchmarks.citynav import paths

    src = Path(f"{paths.tile_stem(map_name)}_features.json")
    try:
        shutil.copyfile(src, d / MAP_PRIOR_JSON)
    except OSError as exc:
        log.warning("%s not written from %s: %s", MAP_PRIOR_JSON, src, exc)
