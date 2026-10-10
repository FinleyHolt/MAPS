"""main(): argparse, config/env/endpoint bootstrap, and the batch loop.

Example:
  python -m benchmarks.citynav.run_benchmark \
      --split val_seen --config configs/citynav.yaml --limit 5
"""

from __future__ import annotations

from benchmarks.citynav.adapter import load_objects, load_split, parse_episode
from benchmarks.citynav.paths import OBJECTS_JSON, assert_available, split_path
from benchmarks.citynav.run_benchmark.config import (
    _export_compact_env,
    export_withhold,
    load_config,
)
from datetime import datetime
from pathlib import Path
from maps.llm import endpoint_spec as endpoint_spec_module
import argparse
import os
import sys
from benchmarks.citynav.run_benchmark.dispatch import (
    _build_compiler,
    _dispatch_one,
    set_run_context,
)
from benchmarks.citynav.run_benchmark.episode_server import _episode_server_loop
from benchmarks.citynav import episodes
from benchmarks.citynav.run_manifest import settle_effort
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

def _emit_run_manifest(args, *, out_root: Path, indexed_rows: list) -> None:
    """Write this run's receipt at the run root.

    THE BAKE STAMP IS WHY THIS EXISTS. Two tile bakes of the same blocks
    live at the same path under the same name and pass every check that
    compares NAMES; a run flown against the wrong one scores a different
    SR with nothing failing (0.351 against 0.169 on identical episodes,
    measured). ``perception.bake_sha256`` is the content hash that tells
    them apart after the fact, and ``run_report`` prints
    ``bake: UNVERIFIED`` for any run that carries none.

    A direct run's root is one run, so the write-once identity check
    refuses a root reused under another config. The fleet driver emits at
    the fleet root; its warm children return into the episode server
    before reaching here.
    """
    from benchmarks.citynav.adapter import parse_episode
    from benchmarks.citynav import run_manifest as _rm

    episodes: list[str] = []
    for i, raw in indexed_rows:
        try:
            episodes.append(parse_episode(raw, row_index=i).episode_id)
        except (KeyError, ValueError, TypeError):
            continue
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
            sample_offset=0,
            limit=args.limit or 0,
            workers=1,
            allowlist_path=None,
            episodes=episodes,
        ),
        log=log.info,
    )


def main(
    *,
    label: str = "CityNav",
    default_split: str = "val_seen",
    output_root: str = "output/citynav",
) -> int:
    """The runner's CLI. A peer benchmark that reuses the engine
    (``benchmarks.calcity.run_benchmark``) passes its own label, default
    split and output root."""
    # THIS entry point is CityNav's own; a peer benchmark that reuses the
    # engine (benchmarks/calcity) declares its name before calling in, so
    # setdefault leaves the peer's word standing. Nothing further down
    # guesses: episode_workdir.benchmark() refuses an undeclared process.
    os.environ.setdefault("SEARCH_BENCHMARK_NAME", "citynav")

    from benchmarks import stamp_repo_provenance

    stamp_repo_provenance()  # loud EVAL-PROVENANCE audit line -> run.log (which code ran)
    parser = argparse.ArgumentParser(
        description=f"Run the {label} navigation benchmark."
    )
    parser.add_argument("--split", default=default_split)
    parser.add_argument("--config", default=None, help="Path to YAML under configs/")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "The run root: run_manifest.json, episodes.jsonl and one "
            "maps_<stamp>_<benchmark>_<slug>/ directory per episode land "
            f"here. Default {output_root}/"
            "<benchmark>_<split>_<date>_<time>, a fresh root per invocation."
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
        "--episode-server",
        action="store_true",
        help=(
            f"Persistent warm-runner mode for {label} full-set "
            "orchestration. Run the one-time init (config, split, objects, "
            "MapsCompiler) once, emit a {'event':'ready'} line on stdout, "
            "then read newline-delimited JSON run commands from stdin and "
            "execute one episode per command through the same run_episode "
            "path the batch loop uses, each into its own attempt directory. "
            "stdout carries ONLY the protocol JSON; all logging stays on the "
            "per-episode run.log. Not for manual use: bench_fullset spawns it."
        ),
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Episode-server mode only: the per-worker run-root under which "
            "each episode's maps_<stamp>_<benchmark>_<slug>/ workdir is "
            "written."
        ),
    )
    args = parser.parse_args()
    if args.seed_base is not None:
        os.environ["SEARCH_SEED_BASE"] = str(args.seed_base)
        log.info("Deterministic seed base: SEARCH_SEED_BASE=%d", args.seed_base)

    assert_available()
    cfg = load_config(args.config)
    _export_compact_env(cfg.compact)
    export_withhold(cfg)
    # Row provenance reads the run facts only this process's argv knows:
    # the config the runner actually loaded and the row cap.
    set_run_context(
        config_path=cfg.source_path,
        limit=args.limit if (args.limit and args.limit > 0) else None,
    )
    # Endpoint discovery — ONE probe per process, before any inference or
    # perception call. The served model id, context window (-> compaction
    # budget + max_tokens pin) and service URLs are DISCOVERED from the live
    # endpoints and frozen; any divergence (unreachable endpoint, >1 served
    # model, missing window, unloadable tokenizer, dead/foreign service)
    # aborts HERE, naming the fact — never a silent mid-episode 404/400.
    # Offline/replay runs declare SEARCH_ENDPOINT_OFFLINE=1 + the explicit
    # SEARCH_ENDPOINT_SPEC_OVERRIDE pin.
    try:
        spec = endpoint_spec_module.discover_from_env()
    except endpoint_spec_module.EndpointDiscoveryError as exc:
        log.error("endpoint discovery failed: %s", exc)
        return 2
    endpoint_spec_module.install_spec(spec)
    settle_effort(spec.model_id)
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
    # disambiguated episode_id the dashboard picked.
    indexed_rows: list[tuple[int, dict]] = list(enumerate(raw_rows))
    if args.episode_id:
        # Fast path: ``<base>#<row_index>`` pins the exact row, so we
        # don't have to re-parse every row to find a match.
        if "#" in args.episode_id:
            base, _, suffix = args.episode_id.rpartition("#")
            try:
                idx = int(suffix)
            except ValueError:
                idx = -1
            if 0 <= idx < len(indexed_rows):
                candidate = indexed_rows[idx]
                if parse_episode(candidate[1]).episode_id == base:
                    indexed_rows = [candidate]
                else:
                    log.error(
                        "Row %d in split=%s does not match prefix %s",
                        idx,
                        args.split,
                        base,
                    )
                    return 2
            else:
                log.error("Row index %d out of range in split=%s", idx, args.split)
                return 2
        else:
            indexed_rows = [
                (i, r)
                for i, r in indexed_rows
                if parse_episode(r).episode_id == args.episode_id
            ]
            if not indexed_rows:
                log.error(
                    "No episode with id=%s in split=%s", args.episode_id, args.split
                )
                return 2
            # CityNav can have multiple descriptions sharing one bare
            # ``map::ann`` id. A dashboard-driven single-episode run wants
            # one; the dashboard addresses episodes by the disambiguated form.
            indexed_rows = indexed_rows[:1]
    if args.limit and args.limit > 0:
        indexed_rows = indexed_rows[: args.limit]
    log.info("Loaded %d episodes from split=%s", len(indexed_rows), args.split)

    objects_by_map = load_objects(OBJECTS_JSON)
    log.info("Loaded CityRefer objects for %d maps", len(objects_by_map))

    # Persistent warm-runner: the one-time init (config + env, split,
    # objects, MapsCompiler) is done. Build the compiler and hand off to the
    # stdin-driven episode loop; the batch path below is never reached in
    # server mode.
    if args.episode_server:
        return _episode_server_loop(
            args,
            cfg=cfg,
            objects_by_map=objects_by_map,
            compiler=_build_compiler(cfg),
            raw_rows=raw_rows,
        )

    out_root = Path(
        args.out
        or f"{output_root}/{benchmark()}_{args.split}_"
        f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        _root_lock = episodes.hold_root(out_root)  # held until the process ends
    except episodes.RootHeld as exc:
        log.error("%s", exc)
        return 2

    from benchmarks.citynav.run_manifest import ManifestIdentityError

    try:
        _emit_run_manifest(args, out_root=out_root, indexed_rows=indexed_rows)
    except ManifestIdentityError as exc:
        log.error("%s", exc)
        return 2

    # The active sensor's description is patched into the reasoner's system
    # prompt, constraining ``sam_concepts`` to surfaces the rig can observe.
    compiler = _build_compiler(cfg)

    for i, (orig_idx, row) in enumerate(indexed_rows):
        try:
            ep_meta = parse_episode(row, row_index=orig_idx)
        except (KeyError, ValueError, TypeError) as exc:
            log.error("row %d: unparseable episode: %s", orig_idx, exc)
            continue
        ep = open_episode_dir(out_root, episode_id=ep_meta.episode_id)
        try:
            # Shared with the persistent episode server so both paths run
            # the identical run_episode + row-write + infra-token code (this
            # batch loop is the bit-identical correctness oracle).
            result = _dispatch_one(
                row=row,
                orig_idx=orig_idx,
                idx=i,
                total=len(indexed_rows),
                compiler=compiler,
                objects_by_map=objects_by_map,
                cfg=cfg,
                recorder=ep.recorder,
                episode_dir=ep.dir,
            )
            result["workdir"] = str(ep.dir.relative_to(out_root))
            episodes.append_row(out_root, result)
        finally:
            ep.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
