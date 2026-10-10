"""Warm-runner stdin protocol: one persistent process executes
episodes on command for bench_fullset.
"""

from __future__ import annotations

from benchmarks.citynav.adapter import parse_episode
from benchmarks.citynav.run_benchmark.config import RunConfig
from datetime import datetime
from pathlib import Path
from maps.search import MapsCompiler
import argparse
import ctypes
import json
import logging
import os
import signal
import sys
from benchmarks.citynav.run_benchmark.dispatch import _dispatch_one
from benchmarks.citynav.run_benchmark.mission_engine import MISSION_DEADLINE_ENV
from benchmarks.citynav.episode_workdir import RUN_LOG, open_episode_dir

log = logging.getLogger("run_benchmark")


# The driver's pid, set by bench_fullset when it spawns this runner.
PARENT_PID_ENV = "CITYNAV_BENCH_PARENT_PID"
_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Have the kernel SIGKILL this runner when the driver that spawned it
    dies. The runner sits in its own session, so nothing else reaps it
    mid-episode. A driver already gone before the request is armed is
    caught by the parent-pid check."""
    parent = os.environ.get(PARENT_PID_ENV)
    if not parent:
        return
    ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != int(parent):
        os._exit(1)


def _resolve_indexed_row(
    indexed_rows: list[tuple[int, dict]], split: str, episode_id: str
) -> tuple[int, dict] | None:
    """Resolve ``episode_id`` to its ``(orig_idx, row)`` pair.

    Same precedence as ``main``'s single-episode resolution: the
    ``<base>#<row_index>`` suffix pins the exact row (fast path);
    otherwise the first row whose parsed id matches. Returns None when
    the id does not resolve in this split.
    """
    if "#" in episode_id:
        base, _, suffix = episode_id.rpartition("#")
        try:
            idx = int(suffix)
        except ValueError:
            return None
        if 0 <= idx < len(indexed_rows):
            candidate = indexed_rows[idx]
            try:
                if parse_episode(candidate[1]).episode_id == base:
                    return candidate
            except (KeyError, ValueError, TypeError):
                return None
        return None
    for orig_idx, r in indexed_rows:
        try:
            if parse_episode(r).episode_id == episode_id:
                return (orig_idx, r)
        except (KeyError, ValueError, TypeError):
            continue
    return None


def _episode_server_loop(
    args: argparse.Namespace,
    *,
    cfg: RunConfig,
    objects_by_map: dict,
    compiler: MapsCompiler | None,
    raw_rows: list[dict],
) -> int:
    """Persistent warm-runner: one-time init is done; serve episodes.

    Reads newline-delimited JSON commands on stdin and runs one episode
    per ``{"cmd":"run",...}`` through the SAME ``_dispatch_one`` /
    ``run_episode`` path the batch loop uses, reproducing launch.py's
    per-episode workdir + ``.in_flight.json``. stdout carries ONLY the protocol JSON; logging stays on the
    per-episode ``run.log``.
    """
    from benchmarks.citynav.launch import (
        _atomic_write_json,
        _in_flight_path,
        resolve_map,
    )

    _die_with_parent()
    run_root = Path(args.run_root or "output/citynav_runs")
    if not run_root.is_absolute():
        run_root = (Path.cwd() / run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    indexed_rows: list[tuple[int, dict]] = list(enumerate(raw_rows))
    root_logger = logging.getLogger()
    out = sys.stdout

    def emit(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    def clear_lockfile() -> None:
        try:
            _in_flight_path(run_root).unlink()
        except OSError:
            pass

    emit({"event": "ready", "pid": pid})
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cmd.get("cmd") == "shutdown":
                break
            if cmd.get("cmd") != "run":
                continue
            seq = cmd.get("seq")
            episode_id = str(cmd.get("episode_id"))
            split = str(cmd.get("split") or args.split)
            resolved = _resolve_indexed_row(indexed_rows, split, episode_id)
            if resolved is None:
                emit(
                    {
                        "event": "done",
                        "seq": seq,
                        "episode_id": episode_id,
                        "reason": "episode_not_found",
                    }
                )
                continue
            orig_idx, row = resolved

            # --- per-episode directory ---------------------------------
            ep = open_episode_dir(
                run_root,
                episode_id=episode_id,
            )
            workdir = ep.dir
            log_path = workdir / RUN_LOG
            recorder = ep.recorder
            # The wall-clock backstop's deadline. The DRIVER picks it: it
            # owns the cap and the moment it started this episode's clock,
            # and a warm child serves many episodes, so this cannot ride the
            # spawn env. Absent means CLEARED, never carried over -- last
            # episode's deadline would stop this one's first pass on arrival.
            _deadline = cmd.get("deadline_epoch")
            if _deadline is None:
                os.environ.pop(MISSION_DEADLINE_ENV, None)
            else:
                os.environ[MISSION_DEADLINE_ENV] = repr(float(_deadline))
            if compiler is not None:
                # workdir is a pure artifact sink on MapsCompiler;
                # reassigning it per episode lands its compile/ tree in
                # the right directory without touching any compile
                # decision (results stay bit-identical to the batch
                # oracle, which leaves it None).
                compiler.workdir = workdir

            handle = {
                "pid": pid,
                "workdir": str(workdir),
                "log_path": str(log_path),
                "episode_id": episode_id,
                "map": resolve_map(episode_id),
                "split": split,
                "started_ts": datetime.now().timestamp(),
            }
            _atomic_write_json(_in_flight_path(run_root), handle)
            emit({"event": "started", "seq": seq, **handle})

            try:
                _dispatch_one(
                    row=row,
                    orig_idx=orig_idx,
                    idx=0,
                    total=1,
                    compiler=compiler,
                    objects_by_map=objects_by_map,
                    cfg=cfg,
                    recorder=recorder,
                    episode_dir=workdir,
                    per_episode=[],
                )
                # The verdict is in <workdir>/episode.json; the driver
                # reads it there and writes the ledger row.
                emit(
                    {
                        "event": "done",
                        "seq": seq,
                        "episode_id": episode_id,
                        "workdir": str(workdir),
                    }
                )
            finally:
                ep.close()
    finally:
        clear_lockfile()
    return 0
