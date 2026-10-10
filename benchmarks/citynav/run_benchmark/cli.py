"""main(): argparse, config/env/endpoint bootstrap, the batch loop,
and the run summary.

Example:
  python -m benchmarks.citynav.run_benchmark \
      --split val_seen --config configs/citynav.yaml --limit 5
"""

from __future__ import annotations

from benchmarks.citynav.adapter import load_objects, load_split, parse_episode
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.metrics import aggregate
from benchmarks.citynav.paths import OBJECTS_JSON, assert_available, split_path
from benchmarks.citynav.run_benchmark.config import (
    _export_compact_env,
    load_config,
)
from datetime import datetime
from pathlib import Path
from maps.llm import endpoint_spec as endpoint_spec_module
from maps.search import MapsCompiler
import argparse
import os
import sys
import time
from benchmarks.citynav.run_benchmark.dispatch import (
    _build_compiler,
    _dispatch_one,
    set_run_context,
)
from benchmarks.citynav.run_benchmark.episode_server import _episode_server_loop
from benchmarks.citynav import episodes
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

BENCHMARK_LABEL = os.environ.get("SEARCH_BENCHMARK_LABEL", "CityNav")
DEFAULT_CONFIG = os.environ.get(
    "SEARCH_BENCHMARK_DEFAULT_CONFIG", "configs/citynav.yaml"
)
DEFAULT_SPLIT = os.environ.get("SEARCH_BENCHMARK_DEFAULT_SPLIT", "val_seen")
OUTPUT_ROOT = os.environ.get("SEARCH_BENCHMARK_OUTPUT_ROOT", "output/citynav")
LAUNCH_MODULE = os.environ.get(
    "SEARCH_BENCHMARK_LAUNCH_MODULE", "benchmarks.citynav.launch"
)


def _emit_run_manifest(args, *, out_root: Path, indexed_rows: list) -> None:
    """Write this run's receipt at the run root.

    THE BAKE STAMP IS WHY THIS EXISTS. Two tile bakes of the same blocks
    live at the same path under the same name and pass every check that
    compares NAMES; a run flown against the wrong one scores a different
    SR with nothing failing (0.351 against 0.169 on identical episodes,
    measured). ``perception.bake_sha256`` is the content hash that tells
    them apart after the fact, and ``run_report`` prints
    ``bake: UNVERIFIED`` for any run that carries none.

    Until now only ``bench_fullset`` emitted one, so every direct run --
    the smoke gate, ``launch.py``, a hand-run arm -- produced numbers with
    no bake identity at all. The fleet driver emits at the fleet root and
    its warm children emit none (they return into the episode server before
    reaching here); a direct run's root is one run, so the write-once
    identity check upstream refuses a root reused under another config.

    ``--no-llm`` is the one run with nothing to stamp: the manifest
    records the served endpoint as a measured identity field, and that
    mode installs no endpoint spec. It says so rather than writing a
    manifest with a hole in it.
    """
    if args.no_llm:
        log.info("run_manifest: skipped (--no-llm installs no endpoint spec)")
        return
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
            # bench_fullset owns shuffling and offsetting, and reporting a
            # seed this run never applied would be a fabricated field.
            sample_seed=0,
            sample_offset=0,
            limit=args.limit or 0,
            workers=1,
            allowlist_path=None,
            episodes=episodes,
        ),
        log=log.info,
    )


def main() -> int:
    # THIS entry point is CityNav's own; a peer benchmark that reuses the
    # engine (benchmarks/calcity) declares its name before calling in, so
    # setdefault leaves the peer's word standing. Nothing further down
    # guesses: episode_workdir.benchmark() refuses an undeclared process.
    os.environ.setdefault("SEARCH_BENCHMARK_NAME", "citynav")

    from benchmarks import stamp_repo_provenance

    stamp_repo_provenance()  # loud EVAL-PROVENANCE audit line -> run.log (which code ran)
    parser = argparse.ArgumentParser(
        description=f"Run the {BENCHMARK_LABEL} navigation benchmark."
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
            f"here. Default {OUTPUT_ROOT}/"
            "<benchmark>_<split>_<date>_<time>, a fresh root per invocation."
        ),
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip compiler entirely")
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "A launcher-minted episode directory: this process runs ONE "
            "episode into it and treats it as the run root. The launcher "
            "redirects stdout+stderr into <workdir>/run.log."
        ),
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Run only the episode with this id (for dashboard-driven single runs).",
    )
    parser.add_argument(
        "--record-llm",
        default=None,
        metavar="DIR",
        help="Write every LLM request/response into DIR as a content-addressed cache.",
    )
    parser.add_argument(
        "--replay-llm",
        default=None,
        metavar="DIR",
        help="Serve LLM responses from DIR; a cache miss raises instead of calling the network.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Deterministic per-turn seed base (wins over SEARCH_SEED_BASE).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            f"Canonical {BENCHMARK_LABEL} eval: scout-driven termination "
            f"+ descent landing. Clears fast-mode env vars. Uses "
            f"{DEFAULT_CONFIG} unless --config explicitly overrides."
        ),
    )
    parser.add_argument(
        "--resume-from-workdir",
        default=None,
        metavar="DIR",
        help=(
            "Resume from a prior run's workdir. ``SearchLoop."
            "restore_from_workdir`` rebuilds the target spec, the SAM "
            "concept union, and compile_seq from the source's "
            "compile/turn_NNNN/finalized/ records before the step loop "
            "starts. CLI value wins over YAML (maps to "
            "cfg.resume_from_workdir)."
        ),
    )
    parser.add_argument(
        "--resume-turn",
        type=int,
        default=None,
        help=(
            "Optional explicit turn index. ``None`` (default) picks "
            "the latest compile/turn_<NNNN>/ whose finalized reasoner "
            "carries reconstructable sam_concepts."
        ),
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        choices=("ingest",),
        help=(
            "Targeted-profile escape hatch. ``--stop-after ingest`` "
            "exits after the pre-flight graph seed, before the mission "
            "conversation opens, so no flight spend is incurred."
        ),
    )
    parser.add_argument(
        "--episode-server",
        action="store_true",
        help=(
            f"Persistent warm-runner mode for {BENCHMARK_LABEL} full-set "
            "orchestration. Run the one-time init (CuPy/NVRTC, config, split, objects, "
            "MapsCompiler) once, emit a {'event':'ready'} line on stdout, "
            "then read newline-delimited JSON run commands from stdin and "
            "execute one episode per command through the same run_episode "
            "path the batch loop uses, reproducing launch.py's per-episode "
            "workdir + .in_flight.json + heartbeat contract. stdout carries "
            "ONLY the protocol JSON; all logging stays on the per-episode "
            "run.log. Not for manual use; the manual/SPA path is "
            f"python -m {LAUNCH_MODULE}."
        ),
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Episode-server mode only: the per-worker run-root under which "
            f"each episode's maps_<stamp>_{benchmark()}_<slug>/ workdir and "
            "the shared "
            ".in_flight.json lockfile are written (mirrors launch.py "
            "--run-root)."
        ),
    )
    args = parser.parse_args()
    if args.eval:
        if args.config is None:
            args.config = DEFAULT_CONFIG
        # Clear fast-mode ablations that would make the run non-canonical.
        os.environ.pop("SEARCH_SKIP_VLM_PERCEPTION", None)

    if args.record_llm and args.replay_llm:
        log.error("--record-llm and --replay-llm are mutually exclusive")
        return 2
    if args.record_llm:
        from maps.llm.http import set_cache_mode

        set_cache_mode("record", args.record_llm)
        log.info("LLM cache mode=record dir=%s", args.record_llm)
    elif args.replay_llm:
        from maps.llm.http import set_cache_mode

        set_cache_mode("replay", args.replay_llm)
        log.info(
            "LLM cache mode=replay dir=%s (cache miss = hard fail)", args.replay_llm
        )
    if args.seed_base is not None:
        os.environ["SEARCH_SEED_BASE"] = str(args.seed_base)
        log.info("Deterministic seed base: SEARCH_SEED_BASE=%d", args.seed_base)

    assert_available()
    cfg = load_config(args.config)
    _export_compact_env(cfg.compact)
    # Row provenance reads the run facts only this process's argv knows:
    # the config the runner actually loaded and the row cap.
    set_run_context(
        config_path=cfg.source_path,
        limit=args.limit if (args.limit and args.limit > 0) else None,
    )
    # Push the runner-side cfg field the engine reads from env. The YAML
    # OWNS this fact: a stale export left over from an earlier arm's shell
    # must never beat the config this process is about to fly — an arm is
    # a config, not an export. A discarded ambient value is logged, not
    # honored, so the operator sees the collision instead of the wrong arm.
    _ambient_dtm = os.environ.get("SEARCH_DETECTED_TARGET_MODE")
    if _ambient_dtm is not None and _ambient_dtm != str(cfg.detected_target_mode):
        log.info(
            "SEARCH_DETECTED_TARGET_MODE=%s from the shell discarded — the "
            "config owns detected_target_mode=%s",
            _ambient_dtm,
            cfg.detected_target_mode,
        )
    os.environ["SEARCH_DETECTED_TARGET_MODE"] = str(cfg.detected_target_mode)
    # Endpoint discovery — ONE probe per process, before any inference or
    # perception call. The served model id, context window (-> compaction
    # budget + max_tokens pin) and service URLs are DISCOVERED from the live
    # endpoints and frozen; any divergence (unreachable endpoint, >1 served
    # model, missing window, unloadable tokenizer, dead/foreign service)
    # aborts HERE, naming the fact — never a silent mid-episode 404/400.
    # Offline/replay runs declare SEARCH_ENDPOINT_OFFLINE=1 + the explicit
    # SEARCH_ENDPOINT_SPEC_OVERRIDE pin. ``--no-llm`` runs have no served-LLM
    # facts to discover and skip the install (spec_installed() stays False).
    if not args.no_llm:
        try:
            endpoint_spec_module.install_spec(
                endpoint_spec_module.discover_from_env()
            )
        except endpoint_spec_module.EndpointDiscoveryError as exc:
            log.error("endpoint discovery failed: %s", exc)
            return 2
    if args.resume_from_workdir is not None:
        cfg.resume_from_workdir = args.resume_from_workdir
    if args.resume_turn is not None:
        cfg.resume_turn = args.resume_turn
    log.info(
        "resume=%s resume_turn=%s",
        cfg.resume_from_workdir or "disabled",
        cfg.resume_turn if cfg.resume_turn is not None else "auto",
    )
    log.info(
        "MISSION: one reasoner conversation per episode -- requested "
        "viewpoints fly the moment they land, the plain reply ends the "
        "mission, and the commit is the last flight's target. A "
        "deterministic discrete 6-action replay over the flown stops "
        "produces the submittable CityNav trajectory (OSR/SPL/NE)."
    )
    if args.stop_after is not None:
        os.environ["SEARCH_STOP_AFTER"] = args.stop_after
        log.info(
            "SEARCH_STOP_AFTER=%s: the runner exits after the pre-flight "
            "graph seed, without flying.",
            args.stop_after,
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
            # CityNav can have multiple descriptions sharing the legacy
            # ``map::ann`` id. For dashboard-driven single-episode runs we
            # want one — the dashboard now uses the disambiguated form.
            indexed_rows = indexed_rows[:1]
    if args.limit and args.limit > 0:
        indexed_rows = indexed_rows[: args.limit]
    log.info("Loaded %d episodes from split=%s", len(indexed_rows), args.split)

    objects_by_map = load_objects(OBJECTS_JSON)
    log.info("Loaded CityRefer objects for %d maps", len(objects_by_map))

    # Persistent warm-runner: the expensive one-time init (CuPy/NVRTC,
    # config + env, split, objects, MapsCompiler) is now done once.
    # Build the compiler and hand off to the stdin-driven episode loop;
    # the single/batch path below is never reached in server mode.
    if args.episode_server:
        server_compiler: MapsCompiler | None = None
        if not args.no_llm:
            server_compiler = _build_compiler(cfg)
        return _episode_server_loop(
            args,
            cfg=cfg,
            objects_by_map=objects_by_map,
            compiler=server_compiler,
            raw_rows=raw_rows,
        )

    if args.workdir:
        # The launcher minted this directory, wrote the in-flight handle
        # for it, and redirects this process's stdout+stderr into
        # <workdir>/run.log. One episode, and the directory IS the run.
        if len(indexed_rows) != 1:
            log.error("--workdir runs exactly one episode; got %d", len(indexed_rows))
            return 2
        out_root = Path(args.workdir).resolve()
        run_log = False
    else:
        out_root = Path(
            args.out
            or f"{OUTPUT_ROOT}/{benchmark()}_{args.split}_"
            f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        ).resolve()
        run_log = True
    out_root.mkdir(parents=True, exist_ok=True)

    _emit_run_manifest(args, out_root=out_root, indexed_rows=indexed_rows)

    compiler: MapsCompiler | None = None
    if not args.no_llm:
        # Resolve the active sensor here so its description can be patched
        # into the reasoner's system prompt, constraining ``sam_concepts``
        # to surfaces the rig can actually observe.
        compiler = _build_compiler(cfg)

    per_episode: list[dict] = []
    wall_t0 = time.monotonic()
    for i, (orig_idx, row) in enumerate(indexed_rows):
        try:
            ep_meta = parse_episode(row, row_index=orig_idx)
        except (KeyError, ValueError, TypeError) as exc:
            log.error("row %d: unparseable episode: %s", orig_idx, exc)
            continue
        ep = open_episode_dir(
            out_root,
            episode_id=ep_meta.episode_id,
            map_name=ep_meta.map_name,
            split=args.split,
            config_path=args.config,
            # A --workdir run's root IS the attempt's directory, so its
            # ledger row points at ".".
            workdir=out_root if args.workdir else None,
            run_log=run_log,
        )
        try:
            # Shared with the persistent episode server so both paths run
            # the identical run_episode + row-write + OOM-token code (this
            # batch loop is the bit-identical correctness oracle).
            result, _ = _dispatch_one(
                row=row,
                orig_idx=orig_idx,
                idx=i,
                total=len(indexed_rows),
                compiler=compiler,
                objects_by_map=objects_by_map,
                cfg=cfg,
                recorder=ep.recorder,
                episode_dir=ep.dir,
                per_episode=per_episode,
            )
            result["workdir"] = str(ep.dir.relative_to(out_root))
            episodes.append_row(out_root, result)
        finally:
            ep.recorder.set_status("finished")
            ep.close()

    wall_s = time.monotonic() - wall_t0
    summary = aggregate(per_episode)
    summary["split"] = args.split
    summary["wall_s"] = round(wall_s, 1)

    log.info("── Results (%s, %d eps) ──", args.split, len(per_episode))
    for k, v in summary.items():
        if k in ("config", "by_referent"):
            continue
        log.info("  %s: %s", k, v)
    if summary.get("by_referent"):
        log.info("  subset SR:")
        for name, sub in summary["by_referent"].items():
            if "SR_pct" in sub:
                log.info(
                    "    %s (%d): SR=%.2f%%", name, sub["n_episodes"], sub["SR_pct"]
                )
    # Per-run LLM token rollup. Sourced from the NetMeter JSONL the
    # llm_http chokepoint writes; reading the whole window here gives a
    # single end-of-run prompt / cached-token + cache-hit readout.
    raw_path = os.environ.get("NET_METER_LOG", "").strip()
    if raw_path:
        try:
            from maps.llm.net_meter import summarize_window

            nm_summary = summarize_window(Path(raw_path).expanduser())
            total_prompt = nm_summary.get("total_prompt_tokens", 0)
            total_cached = nm_summary.get("total_cached_tokens", 0)
            total_completion = nm_summary.get("total_completion_tokens", 0)
            total_reasoning = nm_summary.get("total_reasoning_tokens", 0)
            n_calls = nm_summary.get("n_calls", 0)
            ratio_pct = 100.0 * total_cached / total_prompt if total_prompt else 0.0
            log.info(
                "── LLM tokens (n=%d): prompt=%d cached=%d (%.1f%%) "
                "completion=%d reasoning=%d ──",
                n_calls,
                total_prompt,
                total_cached,
                ratio_pct,
                total_completion,
                total_reasoning,
            )
            for stage_name, by in (nm_summary.get("by_stage") or {}).items():
                pt = by.get("prompt_tokens", 0)
                ch = by.get("cached_tokens", 0)
                pct = 100.0 * ch / pt if pt else 0.0
                log.info(
                    "  %s: prompt=%d cached=%d (%.1f%%) completion=%d calls=%d",
                    stage_name,
                    pt,
                    ch,
                    pct,
                    by.get("completion_tokens", 0),
                    by.get("n", 0),
                )
        except OSError as exc:
            log.warning("LLM token rollup read failed at %s: %s", raw_path, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
