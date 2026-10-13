"""main(): argparse, config/env/endpoint bootstrap, and the batch loop.

Example:
  python -m benchmarks.citynav.run_benchmark \
      --split val_seen --config configs/citynav.yaml --limit 5
"""

from __future__ import annotations

from benchmarks.citynav.adapter import episode_ids, find_row, load_objects, load_split, parse_episode
from benchmarks.citynav.paths import DEFAULT_SPLIT, OBJECTS_JSON, assert_available, split_path
from benchmarks.citynav.run_benchmark.config import (
    export_withhold,
    load_config,
)
from datetime import datetime
from pathlib import Path
from maps.llm import endpoint_spec as endpoint_spec_module
import argparse
import ctypes
import os
import signal
from benchmarks.citynav.run_benchmark.dispatch import (
    _dispatch_one,
)
from benchmarks.citynav import episodes
from benchmarks.citynav import run_manifest as _rm
from benchmarks.citynav.episode_workdir import (
    benchmark,
    open_episode_dir,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_benchmark")

# The driver's pid, which bench_fullset sets when it spawns an attempt.
PARENT_PID_ENV = "CITYNAV_BENCH_PARENT_PID"
_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Have the kernel SIGKILL this attempt when the driver that spawned it
    dies: the attempt sits in its own session, so nothing else reaps it. A
    driver already gone before the request is armed is caught by the
    parent-pid check."""
    ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    parent = os.environ.get(PARENT_PID_ENV)
    if parent and os.getppid() != int(parent):
        os._exit(1)


def _emit_run_manifest(args, *, out_root: Path, indexed_rows: list) -> None:
    """Write this run's receipt at the run root.

    THE BAKE STAMP IS WHY THIS EXISTS. Two tile bakes of the same blocks
    live at the same path under the same name and pass every check that
    compares NAMES; a run flown against the wrong one scores a different
    SR with nothing failing. ``perception.bake_sha256`` is the content hash that tells
    them apart after the fact, and ``run_report`` prints
    ``bake: UNVERIFIED`` for any run that carries none.

    A direct run's root is one run, so the write-once identity check
    refuses a root reused under another config. The fleet driver emits at
    the fleet root; its attempt children return before reaching here.
    """
    _rm.emit_or_verify(
        out_root,
        _rm.build_manifest(
            out_root=out_root,
            config_path=Path(args.config),
            split=args.split,
            # A direct run takes the split's head, not a seeded sample:
            # bench_fullset owns shuffling, and reporting a seed this run
            # never applied would be a fabricated field.
            sample_seed=0,
            limit=args.limit or 0,
            allowlist_path=None,
            episodes=episode_ids(indexed_rows),
        ),
        log=log.info,
    )


def main() -> int:
    """The runner's CLI, for the benchmark ``paths.BENCHMARK`` names."""
    # benchmarks.calcity.use() declares Cal City before calling in;
    # episode_workdir.benchmark() refuses an undeclared process.
    os.environ.setdefault("SEARCH_BENCHMARK_NAME", "citynav")

    parser = argparse.ArgumentParser(
        description=f"Run the {benchmark()} navigation benchmark."
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--config", default=None, help="Path to YAML under configs/")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "The run root: run_manifest.json, episodes.jsonl and one "
            "maps_<stamp>_<benchmark>_<slug>/ directory per episode land "
            "here. Default output/<benchmark>/<benchmark>_<split>_<date>_<time>, "
            "a fresh root per invocation."
        ),
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Run only the episode with this id.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Deterministic per-turn seed base (wins over SEARCH_SEED_BASE).",
    )
    parser.add_argument(
        "--attempt-root",
        default=None,
        help=(
            "bench_fullset's per-worker directory: fly only --episode-id into "
            "a fresh attempt directory under it, print that directory as the "
            "first stdout line, and write its episode.json. The driver owns "
            "the ledger and the manifest."
        ),
    )
    args = parser.parse_args()
    if args.attempt_root:
        if not args.episode_id:
            parser.error("--attempt-root flies one episode: name it with --episode-id")
        _die_with_parent()
    if args.seed_base is not None:
        os.environ["SEARCH_SEED_BASE"] = str(args.seed_base)
        log.info("Deterministic seed base: SEARCH_SEED_BASE=%d", args.seed_base)

    assert_available()
    cfg = load_config(args.config)
    export_withhold(cfg)
    # Endpoint discovery, once per process before any inference or
    # perception call: a fact it cannot discover stops the run here.
    try:
        spec = endpoint_spec_module.discover_from_env()
    except endpoint_spec_module.EndpointDiscoveryError as exc:
        log.error("endpoint discovery failed: %s", exc)
        return 2
    endpoint_spec_module.install_spec(spec)
    _rm.settle_effort(spec.model_id)
    log.info(
        "MISSION: one reasoner conversation per episode -- requested "
        "viewpoints fly the moment they land, the plain reply ends the "
        "mission, and the commit is the last flight's target. A "
        "deterministic discrete 6-action replay over the flown stops "
        "produces the submittable CityNav trajectory (OSR/SPL/NE)."
    )
    raw_rows = load_split(split_path(args.split))
    # Pair each row with its absolute split index so the row_index is
    # stable across filter / limit and parse_episode produces the same
    # disambiguated episode_id whatever the filter.
    indexed_rows: list[tuple[int, dict]] = list(enumerate(raw_rows))
    if args.episode_id:
        found = find_row(raw_rows, args.episode_id)
        if found is None:
            log.error("No episode with id=%s in split=%s", args.episode_id, args.split)
            return 2
        indexed_rows = [found]
    if args.limit > 0:
        indexed_rows = indexed_rows[: args.limit]
    log.info("Loaded %d episodes from split=%s", len(indexed_rows), args.split)

    objects_by_map = load_objects(OBJECTS_JSON)
    log.info("Loaded CityRefer objects for %d maps", len(objects_by_map))

    if args.attempt_root:
        ((orig_idx, row),) = indexed_rows
        ep = open_episode_dir(args.attempt_root, episode_id=args.episode_id)
        print(ep.dir, flush=True)
        try:
            _dispatch_one(row=row, orig_idx=orig_idx, idx=0, total=1,
                          objects_by_map=objects_by_map, cfg=cfg, split=args.split, recorder=ep.recorder,
                          episode_dir=ep.dir)
        finally:
            ep.close()
        return 0

    out_root = Path(
        args.out
        or f"output/{benchmark()}/{benchmark()}_{args.split}_"
        f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        _root_lock = episodes.hold_root(out_root)  # held until the process ends
    except episodes.RootHeld as exc:
        log.error("%s", exc)
        return 2

    try:
        _emit_run_manifest(args, out_root=out_root, indexed_rows=indexed_rows)
    except _rm.ManifestIdentityError as exc:
        log.error("%s", exc)
        return 2

    for i, (orig_idx, row) in enumerate(indexed_rows):
        try:
            ep_meta = parse_episode(row, row_index=orig_idx)
        except (KeyError, ValueError, TypeError) as exc:
            log.error("row %d: unparseable episode: %s", orig_idx, exc)
            continue
        ep = open_episode_dir(out_root, episode_id=ep_meta.episode_id)
        try:
            # The same dispatch bench_fullset's --attempt-root children run.
            result = _dispatch_one(
                row=row,
                orig_idx=orig_idx,
                idx=i,
                total=len(indexed_rows),
                objects_by_map=objects_by_map,
                cfg=cfg,
                split=args.split,
                recorder=ep.recorder,
                episode_dir=ep.dir,
            )
            result["workdir"] = str(ep.dir.relative_to(out_root))
            episodes.append_row(out_root, result)
        finally:
            ep.close()

    return 0
