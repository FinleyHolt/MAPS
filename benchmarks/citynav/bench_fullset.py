"""Continuous full-split benchmark driver.

Runs the search stack across a whole split at a fixed worker concurrency,
one fresh ``run_benchmark --attempt-root`` child per attempt.
Resumable and idempotent: every episode that already has a qualifying row
in ``episodes.jsonl`` is skipped, so re-running the same command continues
coverage rather than restarting.

Seeded random episode order
---------------------------
The split is enumerated block-ordered, so walking it in order front-loads
``birmingham_block_0`` and any partial run is a block-0 sample, not a
representative one. The pending list is therefore shuffled by a fixed
seed (``--sample-seed`` / ``BENCH_SAMPLE_SEED``, default 1234) BEFORE
``--limit`` is applied, so a given ``(seed, limit)`` selects the
identical N episodes. ``--episode-allowlist`` filters the split before
the shuffle.

Per-episode dedup
-----------------
An episode is "done" (skipped on resume) once it has a row in
``episodes.jsonl`` whose ``status`` is not ``infra`` (the attempt never
got a fair run). Infra rows never gate, so the next pass re-attempts them; they are
retained as an audit trail. Among several qualifying attempts the reader
scores the first-started complete one (``analysis.rows.pick_attempts``).

Budgets: a run-count budget (``--limit``) and a wall-time budget
(``--max-wall-s``: stop claiming new episodes after S seconds, let
in-flight finish, retain results). Both default to unlimited.

One condition stops a run that has not spent a budget: mass instant
failure. When the last ``ABORT_WINDOW`` completions are
``ABORT_FRACTION`` infra casualties -- no prediction, excluded from SR,
re-queued -- the run cannot produce results (a missing tile bake is the
canonical case) and it stops with exit 3, naming the dominant cause and
the tail of a representative worker log. Nothing is scored, so a resume
re-flies exactly those episodes once the cause is fixed.

Outputs (under ``--out-root``, default ``output/citynav_fullset``):
    episodes.jsonl  one row per attempt (status ok | timed_out | infra;
                    ``workdir`` relative to this directory)
    summary.txt     ``analysis.run_report`` over the root, refreshed after
                    every attempt, under the driver's infra lines
    driver.log      orchestrator log
    w<id>/          per-worker directory: server.log (every attempt
                    child's stderr), and one
                    maps_<stamp>_<benchmark>_<slug>/ directory per attempt
                    holding episode.json, run.log and flight/
"""

from __future__ import annotations


import argparse
import contextlib
import io
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

from benchmarks.citynav import episodes
from benchmarks.citynav.analysis import rows as _rows

REPO = Path(__file__).resolve().parents[2]


# A peer benchmark reuses this harness by naming its adapter here (e.g.
# benchmarks.calcity._citynav_bridge): the module's ``activate()`` rebinds
# ``benchmarks.citynav.paths`` to the peer's contract before any split or
# preflight read. Unset = CityNav, no-op.
def _activate_bridge_from_env() -> None:
    bridge = os.environ.get("SEARCH_BENCHMARK_BRIDGE", "").strip()
    if bridge:
        import importlib

        importlib.import_module(bridge).activate()


_activate_bridge_from_env()

# Full strict-compliant CityNav: the upstream cropclient observation
# (ortho backend) at the paper's 100-150 m spawn band with the
# ~100-step budget, for the honest paper-comparable SR.
_CONFIG_OVERRIDE = os.environ.get("CITYNAV_BENCH_CONFIG", "").strip()
CONFIG = (
    Path(_CONFIG_OVERRIDE)
    if _CONFIG_OVERRIDE and Path(_CONFIG_OVERRIDE).is_absolute()
    else (REPO / _CONFIG_OVERRIDE).resolve()
    if _CONFIG_OVERRIDE
    else REPO / "configs/citynav.yaml"
)

DEFAULT_WORKERS = 6
# The runner each worker process spawns; a peer benchmark names its own
# (which self-activates its bridge).
RUNNER_MODULE = os.environ.get(
    "SEARCH_BENCHMARK_RUNNER_MODULE", "benchmarks.citynav.run_benchmark"
)
# The ITERATION split. test_unseen is reserved for the one final run and
# must be NAMED on the command line, never inherited from a default.
DEFAULT_SPLIT = "train_seen"
DEFAULT_EPISODE_TIMEOUT_S = 4 * 3600
# Share of the episode cap held back from FLYING, so the model still has
# turns to reply and commit after the pass's wall-clock backstop fires. The
# backstop trades the tail of a pass for a scored row: calcity::6#6 spent its
# whole cap inside one explore and recorded no prediction at all.
_DEADLINE_RESERVE_FRAC = 0.1
# How often the driver asks the reasoner's ``/models`` while an attempt
# flies; two failures in a row kill the attempt as ``vllm_down_midrun``.
VLLM_PROBE_S = 60.0
# How long claiming waits between probes of a reasoner that is down.
VLLM_PAUSE_S = 120.0
# Pause between worker thread starts, so the first attempts do not all
# cold-start at once.
WORKER_STAGGER_S = 2.0
# Infra attempts an episode gets before the run leaves it out and reports
# it as persistent infra with its errors (``--max-infra-attempts``).
MAX_INFRA_ATTEMPTS = 3
# Wall-time budget for a continuous run (0 = unlimited; run until the
# selection / run-count budget is exhausted).
DEFAULT_MAX_WALL_S = 0
# Mass-instant-failure abort. A tree deployed without its tile bake fails
# EVERY episode in seconds. The driver recognizes the shape
# (``no_prediction_unclassified``), but a warning line in the END-OF-RUN
# summary alone lets the run grind to completion and report the problem
# an hour late. When the last ABORT_WINDOW
# completions are ABORT_FRACTION infra casualties the run is not
# producing results and stops, loudly, naming the dominant cause and the
# tail of a representative worker log.
#
# A trailing WINDOW, not a session rate: a rate over the whole session
# cannot fire after a healthy first hour (100 good rows would need ~900
# bad ones to cross), and a mid-run infra death is the same waste.
#
# Absolute count before rate: the window must FILL before the gate can
# fire, so two failures out of two is never evidence.
ABORT_WINDOW = 12
ABORT_FRACTION = 0.9


_log_lock = threading.Lock()


def _enumerate_split(split: str) -> list[str]:
    from benchmarks.citynav.adapter import load_split, parse_episode
    from benchmarks.citynav.paths import SPLIT_FILES, split_path

    if split not in SPLIT_FILES:
        raise SystemExit(f"unknown split {split!r}; known: {sorted(SPLIT_FILES)}")
    rows = load_split(split_path(split))
    out: list[str] = []
    for i, raw in enumerate(rows):
        try:
            out.append(parse_episode(raw, row_index=i).episode_id)
        except (KeyError, ValueError, TypeError):
            continue
    return out


# The native thread pools an attempt child starts: one thread each for BLAS
# (OpenMP, OpenBLAS, MKL), OpenCV and the tokenizer (Rayon), and two malloc
# arenas. Uncapped, every pool starts a thread per core in every child, and
# a thread that fails to start crashes its episode. A value the shell set
# wins.
RUNNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
    "RAYON_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
}


def _runner_env() -> dict[str, str]:
    """The environment an attempt child is spawned under."""
    env = {**RUNNER_THREAD_ENV, **os.environ}
    # Unbuffered, so ``server.log`` gets every line as it happens: redirected
    # to a file, Python block-buffers, and a child that dies fast would leave
    # nothing behind.
    env["PYTHONUNBUFFERED"] = "1"
    # cli.PARENT_PID_ENV, named here so the driver stays off the engine's
    # imports.
    env["CITYNAV_BENCH_PARENT_PID"] = str(os.getpid())
    return env


# The ONE predicate for "this attempt never got a fair run" (excluded from
# SR, re-queued), shared with the readers so the driver and a reader of the
# finished run reach the same verdict. A timeout is not one: it is the
# algorithm missing within its budget.
_infra_casualty = _rows.is_infra_abort


def _infra_cause(row: dict) -> str:
    """What killed this attempt, for the abort banner: the ``error`` every
    infra row carries."""
    return str(row.get("error") or "").strip() or "infra"


def _done_qualify(row: dict) -> str | None:
    """The episode_id when ``row`` marks its episode done (a string id, not
    infra), else ``None``."""
    eid = row.get("episode_id")
    if isinstance(eid, str) and not _infra_casualty(row):
        return eid
    return None


def _missing_block_assets(episode_ids: list[str]) -> list[Path]:
    """The files the selected episodes' blocks need and do not have: the
    bake's per-block files, and the raster pair the ortho reads --
    ``<map>.tif`` beside ``<map>_rgb.tif`` (a georeferenced map) or
    ``<map>.png`` (CityNav's rasterized one). Read off ``paths`` at call
    time, which the bench's bridge rebinds."""
    from benchmarks.citynav import paths  # noqa: PLC0415

    missing: list[Path] = []
    for map_name in sorted({e.split("::", 1)[0] for e in episode_ids}):
        stem = paths.tile_stem(map_name)
        missing += [p for p in (Path(f"{stem}{s}") for s in paths.BAKE_OUTPUT_SUFFIXES)
                    if not p.exists()]
        rgbd = Path(paths.CITYNAV_NATIVE_RGBD_DIR)
        if not (rgbd / f"{map_name}.tif").exists():
            missing.append(rgbd / f"{map_name}.tif")
        if not any((rgbd / f"{map_name}{s}").exists() for s in ("_rgb.tif", ".png")):
            missing.append(rgbd / f"{map_name}_rgb.tif")
    return missing


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.workers = max(1, int(args.workers))
        self.split = args.split
        # The split and seed stamp every row's provenance (dispatch), and
        # ab_report refuses to pair runs whose splits differ or went
        # unrecorded.
        os.environ["CITYNAV_BENCH_SPLIT"] = str(self.split)
        if getattr(args, "sample_seed", None) is not None:
            os.environ["CITYNAV_BENCH_SAMPLE_SEED"] = str(args.sample_seed)
        self.episode_timeout_s = int(args.episode_timeout_s)
        self.max_infra_attempts = max(1, int(args.max_infra_attempts))
        # episode_id -> its infra attempts' errors, counted.
        self._infra_errors: dict[str, Counter] = {}
        self.out_root = Path(args.out_root)
        if not self.out_root.is_absolute():
            self.out_root = (REPO / self.out_root).resolve()
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.out_root / episodes.EPISODES_JSONL
        self.driver_log = self.out_root / "driver.log"
        self.summary_file = self.out_root / "summary.txt"
        self.limit = int(args.limit)
        self.sample_seed = int(args.sample_seed)
        # Newline-delimited episode_ids, applied before the seeded shuffle.
        self.allowlist_path: Path | None = None
        if getattr(args, "episode_allowlist", None):
            ap = Path(args.episode_allowlist)
            if not ap.is_absolute():
                ap = (REPO / ap).resolve()
            self.allowlist_path = ap
        # Dispatch state, all guarded by _dispatch_lock. _in_flight keeps a
        # worker that finishes early off a sibling's live episode; _recent is
        # the mass-infra window as (cause or None, log path).
        self._selection: list[str] = []
        self._cursor = 0
        self._dispatch_lock = threading.Lock()
        self._done_cache: set[str] = set()
        self._in_flight: set[str] = set()
        self._recent: deque[tuple[str | None, str]] = deque(maxlen=ABORT_WINDOW)
        self._abort: str | None = None
        # --max-wall-s: stop claiming once spent; in-flight attempts finish.
        self.max_wall_s = int(args.max_wall_s)
        self._deadline = (
            time.monotonic() + self.max_wall_s if self.max_wall_s > 0 else None
        )
        self.log(f"split={self.split} sample_seed={self.sample_seed}")

    # -- logging / results -------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with _log_lock:
            print(line, flush=True)
            with self.driver_log.open("a") as f:
                f.write(line + "\n")

    def _append_result(self, row: dict) -> None:
        episodes.append_row(self.out_root, row)
        # The write and the done-cache add are one step to a claimer, so a
        # finished episode is never claimed again; infra rows stay
        # claimable up to max_infra_attempts.
        eid = _done_qualify(row)
        with self._dispatch_lock:
            if eid is not None:
                self._done_cache.add(eid)
            elif _infra_casualty(row) and isinstance(row.get("episode_id"), str):
                self._infra_errors.setdefault(row["episode_id"], Counter())[
                    _infra_cause(row)] += 1
        self._check_mass_infra(row)

    @staticmethod
    def _log_excerpt(log_path: str, n: int = 15) -> str:
        """The tail of a worker log, indented, for the abort message."""
        try:
            text = Path(log_path).read_text(errors="replace")
        except OSError as exc:
            return f"    (no readable log at {log_path or '?'}: {exc})"
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return f"    ({log_path} is empty)"
        return "\n".join(f"    | {ln}" for ln in lines[-n:])

    def _check_mass_infra(self, row: dict) -> None:
        """Stop claiming once ABORT_FRACTION of a full ABORT_WINDOW are infra
        casualties (``_infra_casualty``: a timeout never counts). Nothing is
        written or scored; every counted row is already re-queued, so a
        resume after the fix re-flies exactly these episodes."""
        cause = _infra_cause(row) if _infra_casualty(row) else None
        # The log that explains the row: the attempt's own run.log, or the
        # worker's server.log for an attempt that never got a workdir.
        wd = row.get("workdir")
        log_path = (
            self.out_root / wd / "run.log"
            if wd
            else self.out_root / f"w{row.get('worker_id', 0)}" / "server.log"
        )
        with self._dispatch_lock:
            self._recent.append((cause, str(log_path)))
            if self._abort is not None or len(self._recent) < ABORT_WINDOW:
                return
            bad = [(c, lp) for c, lp in self._recent if c]
            if len(bad) < ABORT_FRACTION * ABORT_WINDOW:
                return
            counts = Counter(c for c, _ in bad)
            top, k = counts.most_common(1)[0]
            log_path = next(
                (lp for c, lp in reversed(bad) if c == top and lp), ""
            )
            self._abort = (
                f"mass instant failure: {len(bad)} of the last "
                f"{ABORT_WINDOW} attempts produced no prediction and were "
                f"excluded as infra casualties (dominant cause {top!r}, "
                f"{k}x). This run is not producing results; stopping instead "
                f"of spending the rest of the allocation on it."
            )
        # Outside the lock: self.log takes its own.
        self.log("!! ABORTING RUN — " + self._abort)
        self.log(f"!! by cause: {dict(counts.most_common())}")
        self.log(f"!! representative worker log: {log_path or '(none recorded)'}")
        self.log(self._log_excerpt(log_path))
        scored = sum(1 for r in _rows.iter_json_rows(self.episodes_file)
                     if not _infra_casualty(r))
        self.log(
            ("!! nothing was scored" if not scored else f"!! {scored} attempt(s) were scored")
            + f"; the {len(bad)} infra attempts are excluded and re-queued, so a "
            "resume into this out-root re-flies them once the cause is fixed."
        )

    def _resume_done(self) -> set[str]:
        """The episodes this root's ledger already answered. Read once per
        pass: this run holds the root, so ``_append_result`` keeps the set
        current from then on."""
        return {e for r in _rows.iter_json_rows(self.episodes_file)
                if (e := _done_qualify(r)) is not None}

    def _next_claimed_episode(self, worker_id: int) -> str | None:
        """The next episode not done, not in flight here and not past its
        infra attempts; ``None`` once nothing is left, the wall budget is
        spent, or the run aborted (workers drain, in-flight attempts
        finish)."""
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return None
        if self._abort is not None:
            return None
        with self._dispatch_lock:
            done = self._done_cache
            n = len(self._selection)
            scanned = 0
            while scanned < n:
                eid = self._selection[self._cursor % n]
                self._cursor += 1
                scanned += 1
                if (
                    eid in done
                    or eid in self._in_flight
                    or self._infra_attempts(eid) >= self.max_infra_attempts
                ):
                    continue
                self._in_flight.add(eid)
                return eid
            return None

    def _infra_attempts(self, eid: str) -> int:
        return self._infra_errors.get(eid, Counter()).total()

    def _persistent_infra(self, episodes_: list[str]) -> str:
        """The line naming the selected episodes left out after
        ``max_infra_attempts`` infra attempts each, with their errors
        counted, or "" when there are none."""
        with self._dispatch_lock:
            stuck = [e for e in episodes_ if e not in self._done_cache
                     and self._infra_attempts(e) >= self.max_infra_attempts]
            errors: Counter = Counter()
            for e in stuck:
                errors.update(self._infra_errors[e])
        if not stuck:
            return ""
        hist = ", ".join(f"{k}={v}" for k, v in errors.most_common())
        return (
            f"persistent infra: {len(stuck)} episode(s) after "
            f"{self.max_infra_attempts} infra attempts each ({hist}); "
            f"--max-infra-attempts above {self.max_infra_attempts} flies them again"
        )

    # -- health gate -------------------------------------------------------

    def _vllm_healthy(self) -> bool:
        """True when the reasoner endpoint answers ``/models``, asked with
        the chat path's bearer (hosted endpoints auth ``/v1/models``)."""
        url = os.environ.get("SEARCH_REASONER_URL", "").strip().rstrip("/")
        req = urllib.request.Request(f"{url}/models")
        key = os.environ.get("SEARCH_REASONER_API_KEY")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return 200 <= resp.status < 400
        except (OSError, ValueError):
            return False

    def _wait_for_vllm(self) -> None:
        if self._vllm_healthy():
            return
        self.log("vLLM down; pausing launches until it returns")
        while not self._vllm_healthy():
            time.sleep(VLLM_PAUSE_S)
        self.log("vLLM back; resuming")

    # -- one launch cycle --------------------------------------------------

    def _wait(self, proc: subprocess.Popen) -> str:
        """Block until the attempt child exits: ``exited``; or hits the
        episode cap: ``timeout``; or the reasoner fails two ``/models``
        probes ``VLLM_PROBE_S`` apart while it flies: ``vllm_dead`` (the
        episode is doomed, so this bounds the waste to minutes instead of
        the cap)."""
        deadline = time.monotonic() + self.episode_timeout_s
        down = 0
        while (left := deadline - time.monotonic()) > 0:
            try:
                proc.wait(timeout=min(VLLM_PROBE_S, left))
                return "exited"
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                break
            down = 0 if self._vllm_healthy() else down + 1
            if down >= 2:
                return "vllm_dead"
        return "timeout"

    def _run_attempt(self, worker_id: int, episode_id: str) -> dict:
        """One attempt in a fresh child: its ledger row.

        The row is the child's own ``episode.json`` plus what only this
        driver knows (``worker_id``, ``workdir`` relative to the out-root,
        the wall it measured), with ``status`` overridden on the driver's
        paths: ``timed_out`` for the wall-clock cap; ``infra`` for a lost
        reasoner (``vllm_down_midrun``), a child that died with no record
        (``killed_midrun``), and one that exited clean with none
        (``no_prediction_unclassified``).
        """
        t0 = time.monotonic()
        worker_root = self.out_root / f"w{worker_id}"
        worker_root.mkdir(parents=True, exist_ok=True)
        env = _runner_env()
        # The mission's wall-clock backstop (mission_engine.MISSION_DEADLINE_ENV),
        # an absolute epoch stamped as the cap's clock starts, so the cold
        # start is charged against it exactly as the driver charges it. THE
        # RESERVE IS WHAT ENDING COSTS: the backstop stops FLYING, and the
        # model still needs a turn or two to reply and commit, which is the
        # difference between a scored row and the missing one a kill leaves.
        env["CITYNAV_EPISODE_DEADLINE_EPOCH"] = repr(
            time.time() + self.episode_timeout_s * (1.0 - _DEADLINE_RESERVE_FRAC))
        cmd = [sys.executable, "-m", RUNNER_MODULE, "--config", str(CONFIG),
               "--split", self.split, "--episode-id", episode_id,
               "--attempt-root", str(worker_root)]
        # The child prints its workdir as its first stdout line; an unlinked
        # file holds it, so no pipe can fill and stall the child.
        with tempfile.TemporaryFile("w+") as out, \
                (worker_root / "server.log").open("a") as err:
            try:
                # Its own session, so a kill takes the whole process group.
                proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=err,
                                        text=True, start_new_session=True, env=env)
            except OSError as exc:
                return {
                    "episode_id": episode_id,
                    "worker_id": worker_id,
                    "status": episodes.STATUS_INFRA,
                    "error": f"attempt spawn: {type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - t0, 1),
                    "ts": round(time.time(), 1),
                }
            verdict = self._wait(proc)
            if verdict != "exited":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            out.seek(0)
            first = out.readline().strip()
        elapsed = time.monotonic() - t0
        workdir = Path(first) if first else None
        where = {"workdir": str(workdir.relative_to(self.out_root))} if workdir else {}
        killed = {
            "episode_id": episode_id,
            "worker_id": worker_id,
            "status": episodes.STATUS_INFRA,
            "elapsed_s": round(elapsed, 1),
            "ts": round(time.time(), 1),
            **where,
        }
        landed = episodes.read_episode_json(workdir) if workdir else None
        if verdict == "vllm_dead":
            # Killed now and re-queued: the attempt stays pending until the
            # reasoner is back, instead of grinding to the cap or scoring as
            # a false algorithm miss.
            self.log(f"[w{worker_id}] VLLM-DEAD mid-episode {episode_id} after "
                     f"{elapsed:.0f}s; killed + re-queued (not scored)")
            return {**killed, "error": "vllm_down_midrun"}
        if verdict == "timeout":
            self.log(f"[w{worker_id}] TIMEOUT {episode_id} after {elapsed:.0f}s; killed")
        elif landed is None and proc.returncode != 0:
            # SIGKILLed (a cgroup OOM) or crashed before landing a record:
            # never reached a decision, so re-queued, never scored.
            self.log(f"[w{worker_id}] CRASHED mid-episode {episode_id} after "
                     f"{elapsed:.0f}s (exit {proc.returncode}); re-queued (not scored)")
            return {**killed, "error": "killed_midrun"}
        # The child's own verdict decides (dispatch classifies the episode's
        # exception chain); the driver adds only what it alone saw.
        row = {
            **(landed or {"episode_id": episode_id}),
            "worker_id": worker_id,
            "elapsed_s": round(elapsed, 1),
            "ts": round(time.time(), 1),
            **where,
        }
        if landed is None and verdict == "exited":
            # Fail-closed floor: a child that exits clean with NO record is
            # an unrecognized instant failure, not a completion (a missing
            # tile bake once laundered 762 of these as scored rows).
            row["status"] = episodes.STATUS_INFRA
            row["error"] = "no_prediction_unclassified"
        elif row.get("status") == episodes.STATUS_INFRA:
            pass
        elif verdict == "timeout":
            # The wall-clock cap is an algorithm failure (in the denominator,
            # never infra) whether or not a record was committed.
            row["status"] = episodes.STATUS_TIMED_OUT
        else:
            row["status"] = episodes.STATUS_OK
        return row

    def run_one(self, worker_id: int, episode_id: str) -> None:
        """Fly one attempt and land its row. An infra attempt is flown again
        by a later claim, up to ``max_infra_attempts``."""
        self.log(f"[w{worker_id}] START {episode_id}")
        row = self._run_attempt(worker_id, episode_id)
        m = row.get("metrics") or {}
        summary = f"NE={m.get('NE')} SR={m.get('SR')}" if m else "MISSING"
        self.log(
            f"[w{worker_id}] END {episode_id} status={row['status']} "
            f"elapsed={row['elapsed_s']:.0f}s {summary}"
        )
        self._append_result(row)
        self._write_summary()

    # -- aggregate / summary ----------------------------------------------

    def _write_summary(self) -> None:
        """``summary.txt``: the driver's own lines (infra by cause, persistent
        infra, an abort) above ``analysis.run_report.report`` over the root,
        so the live view scores by the readers' one SR rule."""
        from benchmarks.citynav.analysis import run_report  # noqa: PLC0415

        infra = Counter(str(r.get("error") or "?")
                        for r in _rows.iter_json_rows(self.episodes_file)
                        if r.get("status") == episodes.STATUS_INFRA)
        lines = [
            f"split: {self.split}   workers: {self.workers}   "
            f"episode_cap: {self.episode_timeout_s}s",
            f"infra (re-queued, not scored): {infra.total()}"
            + ("  by cause: " + ", ".join(f"{c}:{k}" for c, k in infra.most_common())
               if infra else ""),
            self._persistent_infra(self._selection),
            f"!! ABORTED: {self._abort}" if self._abort else "",
        ]
        buf = io.StringIO()
        # Under the log lock: redirect_stdout is process-wide, and log() is
        # the driver's only other stdout writer.
        with _log_lock, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                run_report.report(self.out_root, allow_partial=True)
            except SystemExit as exc:  # no scored row yet
                print(exc)
        text = "\n".join(x for x in lines if x) + "\n" + buf.getvalue().lstrip("\n")
        self.summary_file.write_text(text)

    # -- main loop ---------------------------------------------------------

    def worker_loop(self, worker_id: int) -> None:
        while (eid := self._next_claimed_episode(worker_id)) is not None:
            try:
                self._wait_for_vllm()
                self.run_one(worker_id, eid)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[w{worker_id}] FAIL {eid}: {type(exc).__name__}: {exc}")
                try:
                    self._append_result({
                        "episode_id": eid,
                        "worker_id": worker_id,
                        "status": episodes.STATUS_INFRA,
                        "error": f"{type(exc).__name__}: {exc}",
                        "ts": round(time.time(), 1),
                    })
                except OSError as werr:
                    # The ledger itself cannot take a row: no worker can
                    # land a result, so the run stops, loudly.
                    with self._dispatch_lock:
                        self._abort = self._abort or (
                            f"[w{worker_id}] cannot write {self.episodes_file}: {werr}"
                        )
                    self.log(f"!! ABORTING RUN — {self._abort}")
                    return
            finally:
                with self._dispatch_lock:
                    self._in_flight.discard(eid)

    def _select_episodes(self) -> list[str]:
        episodes = _enumerate_split(self.split)
        if self.allowlist_path is not None:
            try:
                lines = self.allowlist_path.read_text().splitlines()
            except OSError as exc:
                raise RuntimeError(
                    f"--episode-allowlist read failed: {self.allowlist_path}: {exc}"
                ) from exc
            allowed = {ln.strip() for ln in lines if ln.strip()}
            before = len(episodes)
            episodes = [e for e in episodes if e in allowed]
            unknown = sorted(allowed - set(episodes))
            self.log(
                f"--episode-allowlist {self.allowlist_path}: "
                f"{len(allowed)} ids requested -> {len(episodes)} match "
                f"the {self.split} split (from {before} total); "
                f"{len(unknown)} ids unknown to the split"
            )
        random.Random(self.sample_seed).shuffle(episodes)
        if self.limit > 0:
            episodes = episodes[: self.limit]
        return episodes

    def dry_run(self) -> int:
        """Selection / dedup preview. Launches nothing."""
        episodes = self._select_episodes()
        done = self._resume_done()
        pending = [e for e in episodes if e not in done]
        print(
            f"sample_seed: {self.sample_seed}   split: {self.split}   "
            f"limit: {self.limit}"
        )
        print(
            f"selected: {len(episodes)}   done: "
            f"{len(done & set(episodes))}   pending: {len(pending)}"
        )
        print(f"first {min(20, len(episodes))} selected (post-shuffle):")
        for e in episodes[:20]:
            mark = "done" if e in done else "PEND"
            print(f"  [{mark}] {e}")
        return 0

    def run(self) -> int:
        try:
            lock = episodes.hold_root(self.out_root)
        except episodes.RootHeld as exc:
            self.log(f"FATAL: {exc}")
            return 2
        try:
            return self._run()
        finally:
            lock.close()

    def _run(self) -> int:
        episodes = self._select_episodes()
        self._selection = episodes
        missing = _missing_block_assets(episodes)
        if missing:
            self.log(
                "FATAL: block assets missing for the selection (every episode on "
                "these blocks would land infra):\n"
                + "".join(f"  {p}\n" for p in missing)
                + "  tiles: benchmarks/citynav/data_prep/bake_tiles.sh, or python -m "
                "benchmarks.calcity.data_prep.bake_tiles\n"
                "  rasters: benchmarks/citynav/data_prep/rasterize_citynav_native.sh, "
                "or python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster "
                "<calcity_rgb.tif>"
            )
            return 2
        # Written before anything flies; a resume under another identity is
        # refused (run_manifest.emit_or_verify).
        from benchmarks.citynav import run_manifest as _rm

        try:
            fresh_manifest = _rm.build_manifest(
                out_root=self.out_root,
                config_path=CONFIG,
                split=self.split,
                sample_seed=self.sample_seed,
                sample_offset=0,
                limit=self.limit,
                workers=self.workers,
                allowlist_path=self.allowlist_path,
                episodes=episodes,
            )
            _rm.emit_or_verify(self.out_root, fresh_manifest, log=self.log)
        except _rm.ManifestIdentityError as exc:
            self.log(f"FATAL: {exc}")
            return 2
        sel_set = set(episodes)
        pass_idx = 0
        while True:
            pass_idx += 1
            if self._deadline is not None and time.monotonic() >= self._deadline:
                self.log(
                    f"wall budget reached ({self.max_wall_s}s); stopping "
                    f"(results retained, aggregate whatever completed)"
                )
                self._write_summary()
                return 0
            self._cursor = 0
            self._done_cache = self._resume_done()
            self._infra_errors = {}
            for r in _rows.iter_json_rows(self.episodes_file):
                if _infra_casualty(r) and isinstance(r.get("episode_id"), str):
                    self._infra_errors.setdefault(r["episode_id"], Counter())[
                        _infra_cause(r)] += 1
            pending = [e for e in episodes if e not in self._done_cache
                       and self._infra_attempts(e) < self.max_infra_attempts]
            self.log(
                f"pass {pass_idx}: split={self.split} selection={len(episodes)} "
                f"done={len(self._done_cache & sel_set)} pending={len(pending)} "
                f"workers={self.workers}"
            )
            persistent = self._persistent_infra(episodes)
            if persistent:
                self.log(persistent)
            if not pending:
                self.log(f"nothing left to fly: {len(episodes)} episodes selected")
                self._write_summary()
                return 0
            threads = []
            for wid in range(self.workers):
                t = threading.Thread(target=self.worker_loop, args=(wid,), daemon=False)
                t.start()
                threads.append(t)
                time.sleep(WORKER_STAGGER_S)
            for t in threads:
                t.join()
            self.log(f"pass {pass_idx} drained")
            # Nonzero, so a launcher sees a failed run rather than a fast one.
            if self._abort is not None:
                self.log(f"ABORTED: {self._abort}")
                self._write_summary()
                return 3


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.citynav.bench_fullset",
        description="Continuous full-test-set CityNav benchmark at fixed concurrency.",
    )
    p.add_argument("--split", default=DEFAULT_SPLIT, help=f"(default {DEFAULT_SPLIT})")
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"concurrency (default {DEFAULT_WORKERS}; empirical optimum)",
    )
    p.add_argument(
        "--out-root",
        default="output/citynav_fullset",
        help="results + per-worker run-roots (default output/citynav_fullset)",
    )
    p.add_argument(
        "--episode-timeout-s",
        type=int,
        default=DEFAULT_EPISODE_TIMEOUT_S,
        help=f"per-episode wall cap (default {DEFAULT_EPISODE_TIMEOUT_S}s = 4h)",
    )
    p.add_argument(
        "--max-wall-s",
        type=int,
        default=DEFAULT_MAX_WALL_S,
        help="continuous-run wall budget (default 0 = run until the "
        "selection / --limit run-count budget is exhausted). >0: stop "
        "claiming new episodes after this many seconds; in-flight "
        "episodes finish and results are retained.",
    )
    p.add_argument(
        "--max-infra-attempts",
        type=int,
        default=MAX_INFRA_ATTEMPTS,
        help="infra attempts an episode gets before the run leaves it out and "
        f"reports it as persistent infra (default {MAX_INFRA_ATTEMPTS})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap episodes AFTER the seeded shuffle (0 = full split; "
        "LIMIT=N + a fixed --sample-seed is the A/B subset selector)",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.environ.get("BENCH_SAMPLE_SEED", "1234")),
        help="seed for the pre-limit episode shuffle (default from "
        "BENCH_SAMPLE_SEED, else 1234). Fixes WHICH N episodes run so "
        "an A/B across code versions is on the identical subset.",
    )
    p.add_argument(
        "--episode-allowlist",
        default=None,
        help="path to a newline-delimited file of episode_ids. When set, "
        "only those ids run (intersected with the enumerated split). The "
        "seeded shuffle + --limit still apply on top of the allowlist. "
        "Used by the ablation harness to pin a fixed N-episode subset "
        "for matched-pair statistical power.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print seed, selected episodes and done/pending; launch nothing",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from benchmarks import stamp_repo_provenance

    # Without a logging config the stamp below falls to logging.lastResort
    # (WARNING) and the fleet driver's EVAL-PROVENANCE block reaches no log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    stamp_repo_provenance()  # loud EVAL-PROVENANCE audit line (which code ran)
    args = _build_parser().parse_args(argv)
    driver = Driver(args)
    if args.dry_run:
        return driver.dry_run()
    # Endpoint discovery: probe the vLLM /v1/models facts + every service /health ONCE, so a fleet run
    # fails HERE — before spawning workers — on an unreachable endpoint,
    # a wrong-model serve, or a mis-ported SAM, instead of churning
    # per-episode infra aborts forever.
    from maps.llm.endpoint_spec import (
        EndpointDiscoveryError,
        discover_from_env,
        install_spec,
    )

    from benchmarks.citynav.run_manifest import settle_effort

    try:
        spec = discover_from_env()
    except EndpointDiscoveryError as exc:
        print(f"FATAL: endpoint discovery failed: {exc}", file=sys.stderr)
        return 2
    install_spec(spec)
    settle_effort(spec.model_id)
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
