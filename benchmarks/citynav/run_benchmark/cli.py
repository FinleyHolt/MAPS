"""main(): argparse, config/env/endpoint bootstrap, and the batch loop.

Example:
  python -m benchmarks.citynav.run_benchmark \
      --split val_seen --config configs/citynav.yaml --limit 5
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

from benchmarks import sandbox
from benchmarks.bench import Bench, block_asset_refusal
from benchmarks.citynav.adapter import (
    episode_ids,
    find_row,
    load_objects,
    load_split,
    parse_episode,
)
from benchmarks.citynav.paths import selected_bench
from benchmarks.citynav.run_benchmark.dispatch import dispatch_one
from benchmarks.run_record import episodes
from benchmarks.run_record import run_manifest as _rm
from benchmarks.run_record.config import load_config
from benchmarks.run_record.episode_workdir import open_episode_dir
from maps.settings import from_environ

log = logging.getLogger("run_benchmark")

_PR_SET_PDEATHSIG = 1


def _die_with_parent(environ) -> None:
    """Have the kernel SIGKILL this attempt when the driver that spawned it
    dies: the attempt sits in its own session, so nothing else reaps it. A
    driver already gone before the request is armed is caught by the
    parent-pid check."""
    ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    parent = environ.get(episodes.PARENT_PID_ENV)
    if parent and os.getppid() != int(parent):
        os._exit(1)


def _emit_run_manifest(args, *, bench: Bench, out_root: Path, indexed_rows: list) -> None:
    """Write this run's receipt at the run root: a direct run's root is one
    run (write-once identity)."""
    _rm.emit_or_verify(
        out_root,
        _rm.build_manifest(
            bench=bench,
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


def _build_parser(bench: Bench) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {bench.name} navigation benchmark.")
    parser.add_argument("--split", default=bench.default_split)
    parser.add_argument(
        "--config",
        default=f"configs/{bench.name}.yaml",
        help="the run config YAML (default %(default)s)",
    )
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
    parser.add_argument(
        "--memory-gib",
        type=sandbox.gib,
        default=sandbox.ATTEMPT_MEMORY_GIB,
        help="this process's memory cap in GiB (default %(default)s; bench_fullset passes its "
        "--attempt-memory-gib): a block that allocates past it raises MemoryError, served as "
        "the block's error",
    )
    return parser


def main(bench: Bench | None = None) -> int:
    """The runner's CLI, for ``bench``; unset, for the bench the environment
    selects (``paths.selected_bench``), which is how an attempt child of
    either driver learns its bench. A run flies model code in this process, so
    it confines itself unless a driver spawned it (``--attempt-root``: the
    driver sandboxes it)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    if bench is None:
        bench = selected_bench(os.environ)
    # A direct run records the bench it flew by name.
    if "SEARCH_BENCHMARK_NAME" not in bench.environ:
        bench = bench.with_environ({"SEARCH_BENCHMARK_NAME": bench.name})

    parser = _build_parser(bench)
    args = parser.parse_args()
    if args.attempt_root:
        if not args.episode_id:
            parser.error("--attempt-root flies one episode: name it with --episode-id")
        # The driver sandboxes its attempts; the child takes the memory cap itself.
        sandbox.cap_memory(args.memory_gib)
        _die_with_parent(bench.environ)

    try:
        bench.assert_available()
    except (OSError, RuntimeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    raw_rows = load_split(bench.split_path(args.split))
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
    if not args.attempt_root:
        # The driver checks its selection's blocks before it spawns an attempt; a direct
        # run checks its own before its --out exists.
        maps = (e.split("::", 1)[0] for e in episode_ids(indexed_rows))
        if (why := block_asset_refusal(bench, maps)) is not None:
            print(f"FATAL: {why}", file=sys.stderr)
            return 2
        sandbox.confine(
            [Path(args.out or f"output/{bench.name}"), *sandbox.ortho_caches([bench.rgbd_dir])],
            args.memory_gib,
            log.info,
            [*sandbox.bench_inputs(bench), args.config],
            flies=True,
        )
    if args.seed_base is not None:
        bench = bench.with_environ({"SEARCH_SEED_BASE": str(args.seed_base)})
        log.info("Deterministic seed base: SEARCH_SEED_BASE=%d", args.seed_base)
    cfg = load_config(args.config)
    # The engine's settings, built once from the run's environment.
    settings = from_environ(bench.environ)
    try:
        bench = _rm.install_endpoint(bench, settings)
    except _rm.EndpointRefused as exc:
        log.error("%s", exc)
        return 2
    log.info(
        "MISSION: one reasoner conversation per episode -- requested "
        "viewpoints fly the moment they land, the plain reply ends the "
        "mission, and the commit is the last flight's target. A "
        "deterministic discrete 6-action replay over the flown stops "
        "produces the submittable CityNav trajectory (OSR/SPL/NE)."
    )
    log.info("Loaded %d episodes from split=%s", len(indexed_rows), args.split)

    objects_by_map = load_objects(bench.objects_json)
    log.info("Loaded CityRefer objects for %d maps", len(objects_by_map))

    if args.attempt_root:
        ((orig_idx, row),) = indexed_rows
        episode = parse_episode(row, row_index=orig_idx)
        ep = open_episode_dir(args.attempt_root, episode_id=args.episode_id, benchmark=bench.name)
        print(ep.dir, flush=True)
        try:
            dispatch_one(
                episode=episode,
                idx=0,
                total=1,
                objects_by_map=objects_by_map,
                cfg=cfg,
                bench=bench,
                settings=settings,
                split=args.split,
                recorder=ep.recorder,
                routing_key=args.episode_id,
                episode_dir=ep.dir,
            )
        finally:
            ep.close()
        return 0

    out_root = Path(
        args.out
        or f"output/{bench.name}/{bench.name}_{args.split}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        _root_lock = episodes.hold_root(out_root)  # held until the process ends
    except episodes.RootHeld as exc:
        log.error("%s", exc)
        return 2

    try:
        _emit_run_manifest(args, bench=bench, out_root=out_root, indexed_rows=indexed_rows)
    except _rm.ManifestIdentityError as exc:
        log.error("%s", exc)
        return 2

    for i, (orig_idx, row) in enumerate(indexed_rows):
        try:
            episode = parse_episode(row, row_index=orig_idx)
        except (KeyError, ValueError, TypeError) as exc:
            log.error("row %d: unparseable episode: %s", orig_idx, exc)
            continue
        ep = open_episode_dir(out_root, episode_id=episode.episode_id, benchmark=bench.name)
        try:
            # The same dispatch bench_fullset's --attempt-root children run.
            result = dispatch_one(
                episode=episode,
                idx=i,
                total=len(indexed_rows),
                objects_by_map=objects_by_map,
                cfg=cfg,
                bench=bench,
                settings=settings,
                split=args.split,
                recorder=ep.recorder,
                episode_dir=ep.dir,
                routing_key=episode.episode_id,
            )
            result["workdir"] = str(ep.dir.relative_to(out_root))
            episodes.append_row(out_root, result)
        finally:
            ep.close()

    return 0
