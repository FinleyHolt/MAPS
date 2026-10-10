"""CLI launcher for ``benchmarks.citynav.run_benchmark``.

Single source of truth for spawning one episode. Owns workdir creation,
env-var wiring, upstream preflight, and the on-disk lockfile that names
the active run.

Run shape::

    python -m benchmarks.citynav.launch --episode-id <id> [--split S] [--map M] \
        [--config <yaml>] [--run-root output/citynav_runs] [--repo-root .] \
        [--force] [--preflight-only] [--no-preflight] [--tail]

On a successful spawn, one JSON object is written to stdout::

    {"pid": 12345, "workdir": "/abs/...", "log_path": "/abs/.../run.log",
     "episode_id": "...", "map": "...", "split": "...",
     "started_ts": 1714839920.0}

Exit codes:
    0 - runner spawned (or --preflight-only succeeded)
    2 - preflight probe failed
    3 - episode_id couldn't resolve to a split row, or --map mismatch
    4 - .in_flight.json already pinned a live pid and --force was not passed

Filesystem contract (under ``--run-root``):
    .in_flight.json - pid, workdir, log_path, episode_id, map, split, started_ts

Written atomically on launch and cleared by a detached reaper
subprocess when the runner exits.
"""

from __future__ import annotations


import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

from benchmarks.citynav import adapter
from benchmarks.citynav.preflight import preflight_services
from benchmarks.citynav.episode_workdir import RUN_LOG, mint_workdir

from pathlib import Path


DEFAULT_RUN_ROOT = "output/citynav_runs"
# The benchmark default is the strict-upstream profile: byte-faithful
# upstream CityNav observation (orthographic nadir cropclient, upstream
# NE/SR eval) at the original paper's 100-150 m start band and ~100
# primitive-action budget, for the honest paper-comparable SR. The
# start band matters: GeoNav (arXiv 2504.09587, SS5.4.2 Table 5) is the
# only paper to vary an observation-geometry knob on CityNav and report
# SR, and its 50 m altitude gives ~3.6x SR vs 100 m via better GSD --
# a 40-60 m profile reads higher but is not comparable to the
# original-altitude paper number.
DEFAULT_CONFIG = "configs/citynav.yaml"

SPLIT_PROBE_ORDER = ("train_seen", "val_seen", "val_unseen", "test_unseen")

# Generic seams used by peer benchmarks that reuse the execution engine.
# CityNav remains the default and carries no knowledge of those benchmarks.
RUNNER_MODULE = os.environ.get(
    "SEARCH_BENCHMARK_RUNNER_MODULE", "benchmarks.citynav.run_benchmark"
)
LAUNCH_MODULE = os.environ.get(
    "SEARCH_BENCHMARK_LAUNCH_MODULE", "benchmarks.citynav.launch"
)
BENCHMARK_LABEL = os.environ.get("SEARCH_BENCHMARK_LABEL", "CityNav")
EPISODE_EXAMPLE = os.environ.get(
    "SEARCH_BENCHMARK_EPISODE_EXAMPLE", "cambridge_block_21::0#1982"
)


# ---------------------------------------------------------------------------
# Filesystem primitives
# ---------------------------------------------------------------------------


def _resume_workdir_type(raw: str) -> str:
    """Reject empty / nonexistent --resume-from-workdir values.

    Shell foot-gun: `CKPT=... python -m benchmarks.citynav.launch
    --resume-from-workdir "$CKPT"` expands `$CKPT` before the inline env
    var is set, passing `""` here. Argparse default=None turned that into
    a silent no-op resume.
    """
    s = raw.strip()
    if not s:
        raise argparse.ArgumentTypeError(
            "--resume-from-workdir got an empty path "
            "(did `$VAR` expand to empty? use `export VAR=... ` on its "
            "own line before invoking, not an inline `VAR=... python ...` prefix)"
        )
    if not Path(s).is_dir():
        raise argparse.ArgumentTypeError(
            f"--resume-from-workdir path does not exist or is not a directory: {s!r}"
        )
    return s


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_map(episode_id: str) -> str:
    """Derive the full map/block prefix from an episode_id."""
    return episode_id.split("::", 1)[0].split("#", 1)[0]


def resolve_repo_root(override: str | None) -> Path:
    """Resolve the upstream repo root.

    Precedence: explicit override -> ``SEARCH_AGENTIC_REPO_ROOT`` env ->
    walk up from this file looking for the default config marker.
    """
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("SEARCH_AGENTIC_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / DEFAULT_CONFIG).exists():
            return cand
    raise RuntimeError(
        "could not locate repo root; pass --repo-root or set SEARCH_AGENTIC_REPO_ROOT"
    )


def resolve_split(episode_id: str) -> str | None:
    """Return the first split whose CSV contains ``episode_id``, or None.

    Probes splits in the order ``train_seen, val_seen, val_unseen,
    test_unseen``. ``benchmarks.citynav.paths.split_path`` reads from
    ``SEARCH_CITYNAV_DATA`` so this works on both laptop and HPC layouts.
    """
    from benchmarks.citynav.paths import SPLIT_FILES, split_path

    for split in SPLIT_PROBE_ORDER:
        if split not in SPLIT_FILES:
            continue
        try:
            with open(split_path(split)) as f:
                rows = json.load(f)
        except (FileNotFoundError, ValueError):
            continue
        for i, raw in enumerate(rows):
            try:
                map_name = adapter.map_name_for_row(raw)
                ann_ids = raw.get("ann_ids") or [0]
                base_id = f"{map_name}::{ann_ids[0]}"
                row_episode_id = f"{base_id}#{i}"
            except (KeyError, ValueError, TypeError):
                continue
            if row_episode_id == episode_id:
                return split
            if base_id == episode_id:
                return split
    return None


# ---------------------------------------------------------------------------
# Preflight (endpoint probes live in benchmarks.citynav.preflight)
# ---------------------------------------------------------------------------


def _city_root_from_map_name(map_name: str) -> str:
    return map_name.split("_block_", 1)[0] if "_block_" in map_name else map_name


def preflight_artifacts(
    *,
    map_name: str,
    config_path: Path,
    repo_root: Path | None = None,
    auto_bake: bool = True,
) -> None:
    """Ensure local CityNav bakes for this episode exist before launching."""
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR, TILES_DIR, tile_stem

    repo = repo_root or Path(__file__).resolve().parents[2]
    stem = Path(tile_stem(map_name))
    tile_suffixes = (
        "_meta.json",
        "_features.json",
        "_graph.npz",
        "_nodes.json",
    )
    missing_tiles = [Path(f"{stem}{suffix}") for suffix in tile_suffixes]
    missing_tiles = [p for p in missing_tiles if not p.exists()]

    # The ortho observation reads the upstream-native rasters; require them.
    missing_rasters = [
        CITYNAV_NATIVE_RGBD_DIR / f"{map_name}.tif",
        CITYNAV_NATIVE_RGBD_DIR / f"{map_name}.png",
    ]
    missing_rasters = [p for p in missing_rasters if not p.exists()]

    if not missing_tiles and not missing_rasters:
        print(
            f"[preflight] local CityNav assets ready (tiles={TILES_DIR})",
            file=sys.stderr,
        )
        return

    if auto_bake:

        def run_bake(label: str, cmd: list[str]) -> None:
            print(f"[preflight] baking missing {label}...", file=sys.stderr)
            print(f"[preflight]   {' '.join(cmd)}", file=sys.stderr)
            try:
                result = subprocess.run(cmd, cwd=repo)
            except FileNotFoundError as exc:
                print(
                    f"[preflight] bake command failed before launch: {exc}",
                    file=sys.stderr,
                )
                return
            if result.returncode != 0:
                print(
                    f"[preflight] bake command exited rc={result.returncode}",
                    file=sys.stderr,
                )

        if missing_tiles:
            # Tile sidecars come from the canonical multi-step bake; there
            # is no single-block auto-bake. Fail loud with the recipe.
            print(
                f"[preflight] tile sidecars missing for {map_name}; "
                f"run benchmarks/citynav/data_prep/bake_tiles.sh "
                f"before launching",
                file=sys.stderr,
            )
        if missing_rasters:
            run_bake(
                "ortho rasters",
                [
                    str(repo / "benchmarks" / "citynav" / "data_prep" / "rasterize_citynav_native.sh"),
                    "--blocks",
                    map_name,
                ],
            )
        missing_tiles = [Path(f"{stem}{suffix}") for suffix in tile_suffixes]
        missing_tiles = [p for p in missing_tiles if not p.exists()]
        missing_rasters = [
            CITYNAV_NATIVE_RGBD_DIR / f"{map_name}.tif",
            CITYNAV_NATIVE_RGBD_DIR / f"{map_name}.png",
        ]
        missing_rasters = [p for p in missing_rasters if not p.exists()]

        if not missing_tiles and not missing_rasters:
            print(
                f"[preflight] local CityNav assets ready (tiles={TILES_DIR})",
                file=sys.stderr,
            )
            return

    print(
        f"[preflight] FAIL: local CityNav assets missing for {map_name} "
        f"(config={config_path})",
        file=sys.stderr,
    )
    if missing_tiles:
        print("[preflight] missing tile artifacts:", file=sys.stderr)
        for path in missing_tiles:
            print(f"[preflight]   {path}", file=sys.stderr)
        print(
            "[preflight] rebuild tiles with:",
            file=sys.stderr,
        )
        print(
            f"[preflight]   SEARCH_CITYNAV_TILES={TILES_DIR} "
            f"./benchmarks/citynav/data_prep/bake_tiles.sh",
            file=sys.stderr,
        )
    if missing_rasters:
        print("[preflight] missing ortho rasters:", file=sys.stderr)
        for path in missing_rasters:
            print(f"[preflight]   {path}", file=sys.stderr)
        print(
            "[preflight] generate upstream-native rasters with:",
            file=sys.stderr,
        )
        print(
            "[preflight]   RASTERIZE_PYTHON=/path/to/rasterize/bin/python "
            "benchmarks/citynav/data_prep/rasterize_citynav_native.sh",
            file=sys.stderr,
        )
        print(
            "[preflight] SensatUrban .ply files must be present under "
            "$CITYNAV_ROOT/raw_ply/{train,test} or $CITYNAV_ROOT/raw_ply_fast.",
            file=sys.stderr,
        )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Lockfile + reaper
# ---------------------------------------------------------------------------


def _in_flight_path(run_root: Path) -> Path:
    return run_root / ".in_flight.json"


def read_in_flight(run_root: Path) -> dict | None:
    """Read ``.in_flight.json``; return None if missing or pid is dead."""
    path = _in_flight_path(run_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pid = int(payload.get("pid", 0))
    if not _is_alive(pid):
        return None
    return payload


def force_stop(pid: int, timeout_s: float = 5.0) -> bool:
    """SIGTERM the process group, then SIGKILL after ``timeout_s``."""
    if pid <= 0 or not _is_alive(pid):
        return False
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return not _is_alive(pid)


def _spawn_reaper(pid: int, run_root: Path) -> None:
    """Detached subprocess that clears the lockfile when ``pid`` exits."""
    script = (
        "import os, sys, time\n"
        "pid = int(sys.argv[1])\n"
        "lock = sys.argv[2]\n"
        "while True:\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except ProcessLookupError:\n"
        "        break\n"
        "    except PermissionError:\n"
        "        pass\n"
        "    time.sleep(2)\n"
        "try:\n"
        "    os.unlink(lock)\n"
        "except OSError:\n"
        "    pass\n"
    )
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(pid),
            str(_in_flight_path(run_root)),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _write_handle(run_root: Path, handle: dict) -> None:
    _atomic_write_json(_in_flight_path(run_root), handle)


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


_STARTUP_AUTO_BAKE_CODE = r"""
import importlib
import json
import os
import sys
from pathlib import Path

launch_module = importlib.import_module(os.environ["SEARCH_BENCHMARK_LAUNCH_MODULE"])
preflight_artifacts = launch_module.preflight_artifacts

map_name = os.environ["SEARCH_CITYNAV_STARTUP_MAP"]
config_path = Path(os.environ["SEARCH_CITYNAV_STARTUP_CONFIG"])
repo_root = Path(os.environ["SEARCH_CITYNAV_STARTUP_REPO"])
runner_cmd = json.loads(os.environ["SEARCH_CITYNAV_RUNNER_CMD_JSON"])
benchmark_label = os.environ.get("SEARCH_BENCHMARK_LABEL", "CityNav")

print(
    f"[launch] startup: checking local {benchmark_label} artifacts",
    file=sys.stderr,
    flush=True,
)
preflight_artifacts(
    map_name=map_name,
    config_path=config_path,
    repo_root=repo_root,
    auto_bake=True,
)
print("[launch] startup: local artifacts ready; starting benchmark", file=sys.stderr, flush=True)
os.execvpe(runner_cmd[0], runner_cmd, os.environ)
"""


def spawn_runner(
    *,
    repo_root: Path,
    run_root: Path,
    episode_id: str,
    split: str,
    map_name: str,
    config: Path,
    resume_from_workdir: str | None = None,
    resume_turn: int | None = None,
    auto_bake: bool = True,
) -> dict[str, Any]:
    """Create the workdir, pre-bake metadata, spawn run_benchmark, return the handle."""
    run_root.mkdir(parents=True, exist_ok=True)
    workdir = mint_workdir(run_root, episode_id)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / RUN_LOG

    py = sys.executable
    cmd = [
        py,
        "-m",
        RUNNER_MODULE,
        "--split",
        split,
        "--episode-id",
        episode_id,
        "--config",
        str(config),
        "--workdir",
        str(workdir),
    ]
    if resume_from_workdir:
        cmd.extend(["--resume-from-workdir", resume_from_workdir])
    if resume_turn is not None:
        cmd.extend(["--resume-turn", str(resume_turn)])
    # Tri-state: None means rely on the runner's dataclass / config
    # default. True / False forward the matching flag to the runner's
    # BooleanOptionalAction.

    env = os.environ.copy()
    for key in ("LANG", "LC_ALL"):
        if env.get(key) in (None, "", "C", "POSIX"):
            env[key] = "C.utf8"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    spawn_cmd = cmd
    if auto_bake:
        env["SEARCH_BENCHMARK_LAUNCH_MODULE"] = LAUNCH_MODULE
        env["SEARCH_CITYNAV_STARTUP_MAP"] = map_name
        env["SEARCH_CITYNAV_STARTUP_CONFIG"] = str(config)
        env["SEARCH_CITYNAV_STARTUP_REPO"] = str(repo_root)
        env["SEARCH_CITYNAV_RUNNER_CMD_JSON"] = json.dumps(cmd)
        spawn_cmd = [py, "-c", _STARTUP_AUTO_BAKE_CODE]

    log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
    if auto_bake:
        log_fh.write(
            f"# startup: local {BENCHMARK_LABEL} artifact auto-bake enabled\n"
        )
    log_fh.write(f"# runner: {' '.join(cmd)}\n")
    log_fh.flush()

    proc = subprocess.Popen(
        spawn_cmd,
        cwd=str(repo_root),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    handle = {
        "pid": proc.pid,
        "workdir": str(workdir),
        "log_path": str(log_path),
        "episode_id": episode_id,
        "map": map_name,
        "split": split,
        "started_ts": datetime.now().timestamp(),
    }
    _write_handle(run_root, handle)
    _spawn_reaper(proc.pid, run_root)
    return handle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {LAUNCH_MODULE}",
        description=f"Launch a {BENCHMARK_LABEL} run; write .in_flight.json.",
    )
    p.add_argument(
        "--episode-id",
        help=(
            f"Episode id, e.g. {EPISODE_EXAMPLE}. Required unless "
            "--preflight-only."
        ),
    )
    p.add_argument(
        "--split", default=None, help="Override split; auto-resolved if omitted."
    )
    p.add_argument(
        "--map",
        dest="map_name",
        default=None,
        help="Override map; derived from episode_id if omitted.",
    )
    p.add_argument(
        "--config", default=None, help=f"Config YAML path (default: {DEFAULT_CONFIG})."
    )
    p.add_argument(
        "--run-root",
        default=DEFAULT_RUN_ROOT,
        help=f"Lockfile + workdir parent (default: {DEFAULT_RUN_ROOT}).",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="Upstream repo root (default: walk up from this module).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="SIGTERM any in-flight run before launching.",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Probe upstream services and exit.",
    )
    p.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip upstream service probes; local artifact auto-bake still runs.",
    )
    p.add_argument(
        "--no-auto-bake",
        action="store_true",
        help=(
            f"Do not run local {BENCHMARK_LABEL} bake commands when required "
            "artifacts are missing."
        ),
    )
    # Resume-from-checkpoint surface. Each flag overrides the matching
    # YAML field on the loaded config; YAML is the fallback when omitted.
    p.add_argument(
        "--resume-from-workdir",
        default=None,
        metavar="DIR",
        type=_resume_workdir_type,
        help=(
            "Resume from a prior run's workdir. The runner restores "
            "target_spec, last_intel_text, SAM video concepts, and "
            "compile_seq from the source's on-disk compile artifacts "
            "and skips pre-flight compile."
        ),
    )
    p.add_argument(
        "--resume-turn",
        type=int,
        default=None,
        help=(
            "Optional explicit turn index. Defaults to the latest "
            "compile/turn_<NNNN>/ that carries a finalized reasoner "
            "with reconstructable sam_concepts."
        ),
    )
    p.add_argument(
        "--tail",
        action="store_true",
        help="Block on run.log; clean lockfile on child exit.",
    )
    return p


def _run_tail(log_path: Path, child_pid: int, run_root: Path) -> int:
    """Foreground mode: stream run.log until the child exits, then clean up."""
    print(
        f"[launch] tailing {log_path} (Ctrl-C stops the tail, NOT the run)",
        file=sys.stderr,
    )
    child_status = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                try:
                    waited, status = os.waitpid(child_pid, os.WNOHANG)
                except ChildProcessError:
                    waited, status = (child_pid, 0)
                if waited == child_pid:
                    if os.WIFEXITED(status):
                        child_status = os.WEXITSTATUS(status)
                    elif os.WIFSIGNALED(status):
                        child_status = 128 + os.WTERMSIG(status)
                    break
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
            for line in f:
                sys.stdout.write(line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(
                "[launch] tail interrupted; run continues in background",
                file=sys.stderr,
            )
            return 0
    try:
        _in_flight_path(run_root).unlink()
    except OSError:
        pass
    return child_status


def main(argv: list[str] | None = None) -> int:
    # THIS entry point is CityNav's own; a peer benchmark that reuses the
    # engine (benchmarks/calcity) declares its name before calling in, so
    # setdefault leaves the peer's word standing. Nothing further down
    # guesses: episode_workdir.benchmark() refuses an undeclared process.
    os.environ.setdefault("SEARCH_BENCHMARK_NAME", "citynav")

    args = _build_parser().parse_args(argv)

    repo_root = resolve_repo_root(args.repo_root)
    config = Path(args.config) if args.config else (repo_root / DEFAULT_CONFIG)
    if not config.is_absolute():
        config = (repo_root / config).resolve()

    if args.preflight_only:
        preflight_services()
        return 0

    if not args.episode_id:
        print(
            "error: --episode-id is required (unless --preflight-only)", file=sys.stderr
        )
        return 2

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (repo_root / run_root).resolve()

    map_name = args.map_name or resolve_map(args.episode_id)
    derived = resolve_map(args.episode_id)
    if args.map_name and args.map_name != derived:
        print(
            f"error: --map={args.map_name!r} does not match prefix derived from "
            f"episode_id={derived!r}",
            file=sys.stderr,
        )
        return 3

    auto_bake = os.environ.get(
        "SEARCH_BENCHMARK_AUTO_BAKE",
        os.environ.get("SEARCH_CITYNAV_AUTO_BAKE", "1"),
    ).strip().lower()
    auto_bake_enabled = (
        not args.no_auto_bake and auto_bake not in {"0", "false", "no", "off"}
    )
    if not auto_bake_enabled:
        preflight_artifacts(
            map_name=map_name,
            config_path=config,
            repo_root=repo_root,
            auto_bake=False,
        )
    if not args.no_preflight:
        from maps.llm.endpoint_spec import install_spec

        # The preflight already measured the served facts; install them so
        # this process reads the validated spec instead of re-probing.
        install_spec(preflight_services())

    # Force-kill any in-flight pid before resolving split (preflight probes
    # must succeed even when the prior pid hogs the GPU; killing it first
    # would tear down its inference server).
    existing = read_in_flight(run_root)
    if existing is not None:
        if not args.force:
            print(
                f"error: in-flight run pid={existing['pid']} workdir={existing['workdir']}",
                file=sys.stderr,
            )
            print("       pass --force to kill it first", file=sys.stderr)
            return 4
        print(f"[launch] --force: stopping pid={existing['pid']}", file=sys.stderr)
        force_stop(int(existing["pid"]))
        try:
            _in_flight_path(run_root).unlink()
        except OSError:
            pass

    split = args.split or resolve_split(args.episode_id)
    if split is None:
        print(
            f"error: episode_id={args.episode_id!r} not found in any split",
            file=sys.stderr,
        )
        return 3

    handle = spawn_runner(
        repo_root=repo_root,
        run_root=run_root,
        episode_id=args.episode_id,
        split=split,
        map_name=map_name,
        config=config,
        resume_from_workdir=args.resume_from_workdir,
        resume_turn=args.resume_turn,
        auto_bake=auto_bake_enabled,
    )
    print(json.dumps(handle), flush=True)

    if args.tail:
        return _run_tail(Path(handle["log_path"]), int(handle["pid"]), run_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
