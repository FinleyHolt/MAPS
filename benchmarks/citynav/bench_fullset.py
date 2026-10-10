"""Continuous full-split benchmark driver.

Runs the search stack across a whole split at a fixed worker concurrency,
one persistent ``run_benchmark --episode-server`` child per worker.
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
    summary.txt     live human-readable aggregate (SR, coverage, ETA)
    driver.log      orchestrator log
    w<id>/          per-worker run-root: server.log (the warm child's
                    stderr), .in_flight.json while an episode runs, and one
                    maps_<stamp>_<benchmark>_<slug>/ directory per attempt
                    holding episode.json, run.log and flight/
"""

from __future__ import annotations


import argparse
import json
import logging
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque

from pathlib import Path

from benchmarks.citynav import episodes
from benchmarks.citynav.analysis import rows as _rows
from benchmarks.citynav.preflight import reasoner_alive

REPO = Path(__file__).resolve().parents[2]


# A peer benchmark reuses this harness by naming its adapter here (e.g.
# benchmarks.calcity._citynav_bridge): the module's ``activate()`` rebinds
# ``benchmarks.citynav.paths`` to the peer's contract before any split or
# preflight read. Same seam family as ``launch.py``'s
# SEARCH_BENCHMARK_RUNNER_MODULE; unset = CityNav, no-op.
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
# (which self-activates its bridge), same seam as launch.py's.
RUNNER_MODULE = os.environ.get(
    "SEARCH_BENCHMARK_RUNNER_MODULE", "benchmarks.citynav.run_benchmark"
)
# The ITERATION split. test_unseen is reserved for the one final run and
# must be NAMED on the command line, never inherited from a default.
DEFAULT_SPLIT = "train_seen"
DEFAULT_EPISODE_TIMEOUT_S = int(
    os.environ.get("CITYNAV_BENCH_EPISODE_TIMEOUT_S", str(4 * 3600))
)
# Share of the episode cap held back from FLYING, so the model still has
# turns to reply and commit after the pass's wall-clock backstop fires. The
# backstop trades the tail of a pass for a scored row: calcity::6#6 spent its
# whole cap inside one explore and recorded no prediction at all.
_DEADLINE_RESERVE_FRAC = 0.1
POLL_INTERVAL_S = 5.0
# Infra attempts an episode gets before the run leaves it out and reports
# it as persistent infra with its errors (``--max-infra-attempts``).
MAX_INFRA_ATTEMPTS = 3
# Wall-time budget for a continuous run (0 = unlimited; run until the
# selection / run-count budget is exhausted).
DEFAULT_MAX_WALL_S = int(os.environ.get("CITYNAV_BENCH_MAX_WALL_S", "0"))
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
ABORT_WINDOW = int(os.environ.get("CITYNAV_BENCH_ABORT_WINDOW", "12"))
ABORT_FRACTION = float(os.environ.get("CITYNAV_BENCH_ABORT_FRACTION", "0.9"))


# Proactive warm-runner recycle. The persistent ``run_benchmark
# --episode-server`` accumulates RSS across the episodes it serves
# (per-block ObjectGraph + mask cache + CPython allocator high-water);
# a long-lived child under the Slurm cgroup ``--mem`` cap can get
# SIGKILLed mid-flight. Recycling (kill, then lazy respawn on the
# next attempt) after every large block AND after a fixed episode count
# resets the child's RSS to its cold baseline. ``0`` disables the
# count-based recycle (the per-block recycle still fires).
WARM_RUNNER_RECYCLE_AFTER = int(os.environ.get("CITYNAV_BENCH_RECYCLE_AFTER", "8"))
# Substrings of map names whose per-block graphs are large enough that a
# single episode should trigger a recycle once it finishes, so the next
# episode starts from a cold RSS baseline rather than stacking on top.
# Env-overridable comma-list (empty disables the per-map recycle; the
# count-based recycle still fires) so it can be tuned from a measurement.
WARM_RUNNER_RECYCLE_MAP_SUBSTRINGS = tuple(
    s.strip()
    for s in os.environ.get("CITYNAV_BENCH_RECYCLE_MAPS", "cambridge").split(",")
    if s.strip()
)
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


# Sentinel pushed onto a runner's stdout queue when its stdout closes
# (the child process exited). Distinct from queue-empty (still running).
_RUNNER_EOF = object()


def _success_criterion() -> str:
    """The criterion SR is scored on, read at call time: the bench's bridge
    rebinds it (ring for CityNav, shape for Cal City)."""
    from benchmarks.citynav import paths  # noqa: PLC0415

    return paths.SUCCESS_CRITERION


# The native thread pools a warm runner starts: one thread each for BLAS
# (OpenMP, OpenBLAS, MKL), OpenCV and the tokenizer (Rayon), and two malloc
# arenas. Uncapped, every pool starts a thread per core in every runner, and
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
    """The environment a warm runner is spawned under."""
    env = {**RUNNER_THREAD_ENV, **os.environ}
    # Unbuffered, so the worker's ``server.log`` gets every line as it
    # happens: redirected to a file, Python block-buffers, and a worker that
    # dies fast would leave a 0-byte log.
    env["PYTHONUNBUFFERED"] = "1"
    # episode_server.PARENT_PID_ENV, named here so the driver stays off
    # the engine's imports.
    env["CITYNAV_BENCH_PARENT_PID"] = str(os.getpid())
    return env


class _WarmRunner:
    """One persistent ``run_benchmark --episode-server`` per bench worker.

    The expensive cold start (CuPy/NVRTC, config, split, objects,
    MapsCompiler) is paid once at ``spawn``; every subsequent episode
    reuses it. Episodes are driven over the child's own stdin/stdout
    pipes (newline-delimited JSON). The child runs in its own session, so
    it arms a parent-death signal (``episode_server._die_with_parent``): a
    SIGKILLed driver leaves no runner flying on.
    """

    def __init__(self, run_root: Path, split: str) -> None:
        self.run_root = run_root
        self.split = split
        self.proc: subprocess.Popen | None = None
        self._q: "queue.Queue[object]" = queue.Queue()
        self._reader: threading.Thread | None = None
        self._seq = 0
        self._early_done: dict | None = None
        # PUBLIC, because it is the only record of why a child died before it
        # could say anything on stdout: the spawn failure row below carries
        # this path so the abort banner can quote the child's own stderr.
        self.server_log = run_root / "server.log"
        # Episodes this child has served since its last spawn; drives the
        # count-based proactive recycle (WARM_RUNNER_RECYCLE_AFTER).
        self._episodes_since_spawn = 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _drain_stdout(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            self._q.put(_RUNNER_EOF)

    def _await_event(
        self,
        events: set[str],
        *,
        timeout_s: float,
        what: str,
        seq: int | None = None,
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                obj = self._q.get(timeout=5.0)
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(f"warm runner died before {what}")
                continue
            if obj is _RUNNER_EOF:
                raise RuntimeError(f"warm runner stdout closed before {what}")
            if isinstance(obj, dict) and obj.get("event") in events:
                # Seq discipline (mirrors _wait_for_exit): when waiting on
                # a specific episode, drain — never return — a stale
                # started/done from a prior seq. Without this a late
                # episode-N ``done`` is consumed as episode N+1's start
                # and run_episode returns N's workdir, mis-scoring N+1.
                if (
                    seq is not None
                    and obj.get("seq") is not None
                    and obj.get("seq") != seq
                ):
                    continue
                return obj
            # ignore any other protocol line
        raise RuntimeError(f"warm runner timed out before {what}")

    def spawn(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            RUNNER_MODULE,
            "--episode-server",
            "--split",
            self.split,
            "--config",
            str(CONFIG),
            "--run-root",
            str(self.run_root),
        ]
        env = _runner_env()
        self._q = queue.Queue()
        self._early_done = None
        self._episodes_since_spawn = 0
        slog = open(self.server_log, "a", buffering=1)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=slog,
            text=True,
            start_new_session=True,
            env=env,
        )
        self._reader = threading.Thread(
            target=self._drain_stdout, args=(self.proc,), daemon=True
        )
        self._reader.start()
        # Block on the one-time cold start (CuPy/NVRTC + compiler ctor).
        self._await_event({"ready"}, timeout_s=600.0, what="ready")

    def run_episode(
        self, episode_id: str, *, deadline_epoch: float | None = None
    ) -> tuple[dict, int]:
        """Dispatch one episode; block until the child created the
        per-episode workdir + .in_flight.json and emitted ``started``.
        Returns (launch-style handle, seq)."""
        if not self.alive():
            raise RuntimeError("warm runner not alive")
        self._seq += 1
        seq = self._seq
        self._episodes_since_spawn += 1
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps(
                {
                    "cmd": "run",
                    "episode_id": episode_id,
                    "split": self.split,
                    "seq": seq,
                    # Per-episode, because the child is WARM and serves many:
                    # a spawn-time env would hand episode 40 episode 1's
                    # deadline. ``None`` clears it child-side.
                    "deadline_epoch": deadline_epoch,
                }
            )
            + "\n"
        )
        self.proc.stdin.flush()
        self._early_done = None
        obj = self._await_event(
            {"started", "done"}, timeout_s=600.0, what="episode start", seq=seq
        )
        if obj.get("event") == "done":
            # Resolved before flight (e.g. episode_not_found). Synthesize
            # a handle; _wait_for_exit returns immediately on _early_done.
            self._early_done = obj
            return (
                {
                    "pid": obj.get("pid", self.proc.pid),
                    "workdir": obj.get("workdir", ""),
                    "log_path": obj.get("log_path", ""),
                    "episode_id": episode_id,
                    "map": "",
                    "split": self.split,
                    "started_ts": time.time(),
                },
                seq,
            )
        handle = {
            k: obj.get(k)
            for k in (
                "pid",
                "workdir",
                "log_path",
                "episode_id",
                "map",
                "split",
                "started_ts",
            )
        }
        return handle, seq

    def kill(self) -> None:
        """SIGTERM the child's process group, SIGKILL after 5 s.

        Process-group (not bare pid) because the warm runner spawns
        SearchLoop / executor threads + child renders; mirrors
        launch.force_stop's proven semantics.
        """
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def should_recycle(self, map_name: str) -> bool:
        """Whether to recycle this child after the episode it just served.

        True if the served block is one of the large
        (``WARM_RUNNER_RECYCLE_MAP_SUBSTRINGS``) maps, or the child has
        served at least ``WARM_RUNNER_RECYCLE_AFTER`` episodes since its
        last spawn. Recycling bounds cumulative RSS growth under the
        Slurm cgroup ``--mem`` cap.
        """
        if any(s in (map_name or "") for s in WARM_RUNNER_RECYCLE_MAP_SUBSTRINGS):
            return True
        return (
            WARM_RUNNER_RECYCLE_AFTER > 0
            and self._episodes_since_spawn >= WARM_RUNNER_RECYCLE_AFTER
        )

    def clear_lockfile(self) -> None:
        try:
            (self.run_root / ".in_flight.json").unlink()
        except OSError:
            pass

    def shutdown(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.stdin.close()
            except (OSError, ValueError):
                pass
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            if proc.poll() is None:
                self.kill()
        self.clear_lockfile()


def _infra_casualty(row: dict) -> bool:
    """True when this row is an infra abort: no prediction, excluded from
    SR, re-queued for a full redo.

    The ONE predicate for "this episode never got a fair run", shared by
    the done-set below and the mass-instant-failure abort. A hard episode
    timeout with no prediction is deliberately NOT one of these — that is
    the ALGORITHM failing to answer within its budget, a legitimate
    CityNav miss.

    A mission that ends ``ended_by=transient_failures`` (the reasoner
    endpoint failed MAX_ROUND_TRANSIENT calls in a row) IS one of these:
    the serve died under the episode mid-search, the row's record is
    whatever flight happened to be last before the outage. Measured on
    ``f40088a:calcity_full_replyrepair``: a serve OOM storm forced 50/1022
    episodes to SR 0.000 this way. ``compaction_infeasible``,
    ``oversize_rejection`` and ``turn_cap`` stay scored — those are the
    algorithm's own budget.

    The predicate itself lives in ``analysis.rows``, because the READER of
    a finished run has to reach the same verdict this driver does. It was
    restated here instead, and the two diverged on exactly the
    ``transient_failures`` clause: the driver excluded such a row and
    re-queued the episode, while the reader scored the excluded attempt
    and discarded the re-run as a replicate. On a reasoning-effort arm
    whose failure was deterministic, that disagreement re-queued the same
    episode forever.
    """
    return _rows.is_infra_abort(row)


def _infra_cause(row: dict) -> str:
    """What killed this episode, in words the abort banner can act on.

    Every ``infra`` row carries an ``error``: the producer writes one for
    each of its own verdicts (``mission ended_by=transient_failures`` for a
    serve that died under the mission), and the driver writes one for each of its
    kill paths.
    """
    return str(row.get("error") or "").strip() or "infra"


def _done_qualify(row: dict) -> str | None:
    """Done-set membership predicate, shared by the pass-start read and the
    result writer.

    An episode counts as done once it has a row with a string
    ``episode_id`` whose status is not ``infra`` (the attempt never got a
    fair run: the next pass re-attempts it). Excluded rows stay as an
    audit trail. Returns the episode_id when the row
    qualifies, else ``None``.
    """
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
        # The split is the ground truth an SR is a score AGAINST, so it belongs
        # in the artifact, not only in the launch command. Dispatch stamps it
        # into every row's provenance, and ab_report REFUSES to pair two runs
        # whose splits differ -- or whose splits went unrecorded (null is
        # fatal there, not a silent pass).
        os.environ["CITYNAV_BENCH_SPLIT"] = str(self.split)
        if getattr(args, "sample_seed", None) is not None:
            os.environ["CITYNAV_BENCH_SAMPLE_SEED"] = str(args.sample_seed)
        self.episode_timeout_s = int(args.episode_timeout_s)
        self.max_infra_attempts = max(1, int(args.max_infra_attempts))
        # episode_id -> its infra attempts' errors, counted; read from the
        # ledger at each pass start and kept current by _append_result.
        # Guarded by _dispatch_lock.
        self._infra_errors: dict[str, Counter] = {}
        self.loop = bool(args.loop)
        self.out_root = Path(args.out_root)
        if not self.out_root.is_absolute():
            self.out_root = (REPO / self.out_root).resolve()
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.out_root / episodes.EPISODES_JSONL
        self.driver_log = self.out_root / "driver.log"
        self.summary_file = self.out_root / "summary.txt"
        self.limit = int(args.limit)
        self.sample_seed = int(args.sample_seed)
        self.sample_offset = int(getattr(args, "sample_offset", 0) or 0)
        # Optional episode-allowlist file: newline-delimited episode_ids.
        # Intersected with the enumerated split in ``_select_episodes``
        # before the seeded shuffle, so an (allowlist, seed) pair selects
        # the same N episodes. Set by the ablation harness; production
        # runs leave it None.
        # A --from-manifest run's source record; main() assigns it after
        # apply_replay_args forced the recorded selection into args. run()
        # gates the re-derived resolution against it before emitting.
        self.replay_source: dict | None = None
        self.allowlist_path: Path | None = None
        if getattr(args, "episode_allowlist", None):
            ap = Path(args.episode_allowlist)
            if not ap.is_absolute():
                ap = (REPO / ap).resolve()
            self.allowlist_path = ap
        # Continuous-dispatch shared state (populated in run()).
        self._selection: list[str] = []
        self._cursor = 0
        self._dispatch_lock = threading.Lock()
        self._done_cache: set[str] = set()
        # Episodes this process's workers are RUNNING right now. "Done"
        # means a row landed in episodes.jsonl, so it says nothing about
        # an episode still in flight — without this, a worker that
        # finishes early re-hands a peer worker's live episode to itself.
        # Guarded by _dispatch_lock, like the cursor and the done cache.
        self._in_flight: set[str] = set()
        # Mass-instant-failure gate: the last ABORT_WINDOW completions as
        # (cause, log path) — cause None on a row that produced a result —
        # and the one-line reason the run stopped (None while healthy).
        # Both guarded by _dispatch_lock.
        self._recent: deque[tuple[str | None, str]] = deque(maxlen=ABORT_WINDOW)
        self._abort: str | None = None
        self._t_start = time.monotonic()
        # Rows THIS driver session appended — the only honest numerator
        # for the session-throughput line (all-time count over session
        # elapsed once read 458 ep/h on a resumed run).
        self._session_done = 0
        # Wall-time budget: a continuous run's standard stop condition.
        # 0 = run until the selection is exhausted (or --limit's
        # run-count budget). >0 = stop claiming new episodes once this
        # many seconds elapse; in-flight episodes finish, results are
        # retained, the summary is written. All lanes start within the
        # ~2 s/lane stagger, so a shared deadline is effectively aligned.
        self.max_wall_s = int(args.max_wall_s)
        self._deadline = (
            self._t_start + self.max_wall_s if self.max_wall_s > 0 else None
        )
        self._split_size = 0
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
        self._session_done += 1
        # Dispatch-time invariant: a claim must never return an episode
        # whose qualifying row has been written this run. The write and
        # the done-cache add are one step from a claimer's view (claims
        # take _dispatch_lock). Infra rows do not qualify and stay
        # re-claimable up to max_infra_attempts.
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
        """The tail of a worker log, indented, for the abort message. The
        point of the gate is that the next actor does not go spelunking."""
        try:
            text = Path(log_path).read_text(errors="replace")
        except OSError as exc:
            return f"    (no readable log at {log_path or '?'}: {exc})"
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return f"    ({log_path} is empty)"
        return "\n".join(f"    | {ln}" for ln in lines[-n:])

    def _check_mass_infra(self, row: dict) -> None:
        """Stop the run once mass instant failure is already conclusive.

        The classifier upstream already recognizes every shape this fires
        on; what was missing was acting on it before the allocation was
        spent. The window counts only rows the done-set EXCLUDES (the
        shared ``_infra_casualty`` predicate), so an episode timeout — the
        algorithm failing to answer, a legitimate miss — can never fill it,
        and a healthy run's occasional infra attempt cannot either: the gate
        needs ABORT_FRACTION of a FULL window.

        Nothing is written and nothing is scored. Every counted row was
        already excluded + re-queued, so a resume after the fix re-flies
        exactly these episodes.
        """
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
        # Logged OUTSIDE the lock (self.log takes its own) and at detection
        # time rather than at the end, which is the whole point.
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
        # Hand this worker the next undone episode, or None when nothing
        # is left for it to take.
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return None
        # Mass instant failure: stop claiming. Workers drain, run() reports
        # and exits nonzero. Same shape as the wall-budget stop — in-flight
        # episodes finish, nothing already written is touched.
        if self._abort is not None:
            return None
        # _in_flight keeps this process's own workers off each other's
        # episodes (the cursor alone cannot — it wraps modulo the
        # selection, so it would re-offer a live episode as soon as a worker
        # finished early); the done cache keeps a wrapped cursor off
        # episodes already finished since the pass started.
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
            # One full scan found nothing claimable: everything selected
            # is done or running on this process's own workers, and a
            # sibling finishing its episode leaves nothing new to take.
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
        return reasoner_alive()

    def _wait_for_vllm(self) -> None:
        if self._vllm_healthy():
            return
        self.log("vLLM down; pausing launches until it returns")
        while not self._vllm_healthy():
            time.sleep(120)
        self.log("vLLM back; resuming")

    # -- one launch cycle --------------------------------------------------

    def _wait_for_exit(self, runner: "_WarmRunner", seq: int, timeout_s: float) -> str:
        """Block until the warm runner reports this episode finished.

        Returns one of:
        "exited"    — the episode produced its result: a ``done``/
                      ``fatal`` for THIS seq. The runner ran to a decision
                      (a committed prediction, or a clean no-commit) and
                      its landed ``episode.json`` carries the verdict.
        "crashed"   — the runner process ended (stdout EOF) WITHOUT a
                      ``done``/``fatal`` for this seq: SIGKILL (Slurm
                      cgroup OOM) or a non-``Exception`` escape
                      (MemoryError surfaced pre-GIL, an exception before
                      the per-episode try). The episode never reached a
                      decision and landed no row, so it is an infra
                      casualty (excluded from SR + re-queued), not a
                      scored zero. A clean completion
                      ALWAYS emits ``done`` first (run_benchmark emits it
                      even when the episode caught an exception and
                      committed nothing), and the FIFO queue delivers that
                      ``done`` before the later EOF, so reaching the EOF
                      branch is a reliable crash signal.
        "timeout"   — hit the per-episode wall cap.
        "vllm_dead" — the shared vLLM died WHILE this episode was in
                      flight (>= 2 consecutive 60s-spaced probes), so the
                      episode is doomed; bounds wasted time to ~2 min
                      instead of up to the 4 h cap. Same semantics and
                      cadence as the pre-warm-runner implementation.
        """
        early = runner._early_done
        if early is not None and early.get("seq") == seq:
            return "exited"
        deadline = time.monotonic() + timeout_s
        last_vllm_check = time.monotonic()
        vllm_down_streak = 0
        while time.monotonic() < deadline:
            try:
                obj: object = runner._q.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                obj = None
            if obj is _RUNNER_EOF:
                # stdout closed with no done/fatal for this seq: the
                # process was SIGKILLed (cgroup OOM) or crashed without
                # emitting a result. Infra casualty, not a scored miss.
                return "crashed"
            if (
                isinstance(obj, dict)
                and obj.get("event") in ("done", "fatal")
                and obj.get("seq") == seq
            ):
                return "exited"
            now = time.monotonic()
            if now - last_vllm_check >= 60.0:
                last_vllm_check = now
                if self._vllm_healthy():
                    vllm_down_streak = 0
                else:
                    vllm_down_streak += 1
                    if vllm_down_streak >= 2:
                        return "vllm_dead"
        return "timeout"

    @staticmethod
    def _landed_row(workdir: Path) -> dict | None:
        """The attempt's own ``episode.json`` once it has landed, else None."""
        return episodes.read_episode_json(workdir)

    def _run_attempt(
        self, worker_id: int, episode_id: str, runner: "_WarmRunner"
    ) -> dict:
        """One attempt: its ledger row.

        The row is the runner's own ``episode.json`` plus what only this
        driver knows (``worker_id``, ``workdir`` relative to the out-root,
        the wall it measured), with ``status`` overridden on the driver's
        kill paths: ``timed_out`` for the wall-clock cap, ``infra`` for a
        runner that died, lost its reasoner, or exited with no record.
        """
        t0 = time.monotonic()
        # The pass's wall-clock backstop, as an absolute epoch, stamped at
        # the same instant the kill clock above starts -- so a slow cold
        # start or preflight below is charged against it exactly as the
        # driver charges it. THE RESERVE IS WHAT ENDING COSTS: the backstop
        # stops FLYING, and the model still needs a turn or two to reply and
        # commit, which is the difference between a scored row and the
        # MISSING one a SIGKILL leaves.
        deadline_epoch = time.time() + self.episode_timeout_s * (
            1.0 - _DEADLINE_RESERVE_FRAC
        )
        try:
            # Respawn-on-dead: a crashed / OOM-killed / vLLM-dead-killed
            # runner is replaced with a fresh process. This re-pays cold
            # start exactly once, then amortizes again.
            if not runner.alive():
                runner.clear_lockfile()
                runner.spawn()
            handle, seq = runner.run_episode(
                episode_id, deadline_epoch=deadline_epoch)
        except Exception as exc:  # spawn / dispatch failure ~ launcher exit
            # The warm runner could not even start/dispatch the episode
            # (process spawn failed under host/GPU contention, lockfile
            # race, transient exec error). The algorithm never ran, so
            # this is infra: excluded from SR + the done-set and fully
            # re-queued. No ``workdir``: an episode that never started has
            # none, and the child's own stderr (``w<id>/server.log``) is
            # the only record of why -- ``_check_mass_infra`` reads it there.
            return {
                "episode_id": episode_id,
                "worker_id": worker_id,
                "status": episodes.STATUS_INFRA,
                "error": f"warm-runner spawn/dispatch: {type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - t0, 1),
                "ts": round(time.time(), 1),
            }
        workdir = Path(handle["workdir"])
        status = self._wait_for_exit(runner, seq, self.episode_timeout_s)
        elapsed = time.monotonic() - t0
        killed = {
            "episode_id": episode_id,
            "worker_id": worker_id,
            "status": episodes.STATUS_INFRA,
            "elapsed_s": round(elapsed, 1),
            "ts": round(time.time(), 1),
            "workdir": str(workdir.relative_to(self.out_root)),
        }
        if status == "vllm_dead":
            # The shared vLLM died mid-episode. This run is doomed (its
            # LLM calls hang/fail); killing it now and recording an infra
            # row means the episode stays pending and is re-attempted once
            # vLLM is back, instead of grinding to the 4 h cap or being
            # scored as a false algorithm miss.
            self.log(
                f"[w{worker_id}] VLLM-DEAD mid-episode {episode_id} after "
                f"{elapsed:.0f}s; killing + re-queueing (not scored)"
            )
            runner.kill()
            return {**killed, "error": "vllm_down_midrun"}
        if status == "crashed":
            # The runner process died mid-episode (SIGKILL from the Slurm
            # cgroup OOM-killer, or a non-Exception escape) WITHOUT
            # emitting a done for this seq. It never reached a decision —
            # its episode.json is absent. An infra casualty, NOT an algorithm miss:
            # excluded from SR + the done-set and re-queued for a full redo
            # (up to max_infra_attempts), never scored on an un-flown
            # binding. runner.kill() is a safe no-op here (the process is
            # already dead; kill() returns immediately when poll() != None).
            runner.kill()
            self.log(
                f"[w{worker_id}] CRASHED mid-episode {episode_id} after "
                f"{elapsed:.0f}s; killing + re-queueing (not scored)"
            )
            return {**killed, "error": "killed_midrun"}
        if status == "timeout":
            self.log(
                f"[w{worker_id}] TIMEOUT {episode_id} after {elapsed:.0f}s; killing"
            )
            runner.kill()
        # The runner's own verdict decides (dispatch classifies the episode's
        # exception chain); the driver adds only what it alone saw.
        landed = self._landed_row(workdir)
        row = {
            **(landed or {"episode_id": episode_id}),
            "worker_id": worker_id,
            "elapsed_s": round(elapsed, 1),
            "ts": round(time.time(), 1),
            "workdir": str(workdir.relative_to(self.out_root)),
        }
        if landed is None and status == "exited":
            # Fail-closed floor: a runner that exits clean with NO record is
            # not a completion but an unrecognized instant failure (a missing
            # tile bake once laundered 762 of these as scored rows). Excluded
            # from SR and re-queued.
            row["status"] = episodes.STATUS_INFRA
            row["error"] = "no_prediction_unclassified"
        elif row.get("status") == episodes.STATUS_INFRA:
            pass
        elif status == "timeout":
            # The wall-clock cap is the binding bound under an unbounded step
            # budget, so readers count these beside SR. An algorithm failure
            # (in the denominator, never infra) whether or not a record was
            # committed.
            row["status"] = episodes.STATUS_TIMED_OUT
        else:
            row["status"] = episodes.STATUS_OK
        # Proactive recycle: this child served the episode to completion,
        # so its RSS now carries that block's ObjectGraph + allocator
        # high-water. If the block is large or the child has served enough
        # episodes, kill it now; the respawn-on-dead path at the top of
        # the next _run_attempt re-pays the cold start exactly once and
        # the next episode starts from a cold RSS baseline. The
        # crashed / vllm_dead / timeout paths already killed the runner,
        # so this only fires on a clean completion.
        if runner.should_recycle(handle.get("map") or ""):
            self.log(
                f"[w{worker_id}] RECYCLE runner after {episode_id} "
                f"(map={handle.get('map') or '?'}, "
                f"served={runner._episodes_since_spawn})"
            )
            runner.kill()
        return row

    def run_one(self, worker_id: int, episode_id: str, runner: "_WarmRunner") -> None:
        """Fly one attempt and land its row. An infra attempt is flown again
        by a later claim, up to ``max_infra_attempts``."""
        self.log(f"[w{worker_id}] START {episode_id}")
        row = self._run_attempt(worker_id, episode_id, runner)
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
        # Selection is the SHARED rule (analysis.rows.pick_attempts): an
        # episode's result is its FIRST fully complete attempt, ordered by
        # attempt START time. Last-row-wins scored whichever replicate
        # FINISHED last -- episodes.jsonl appends on completion, and under
        # the pre-f22e8fd dispatch which replicate that was came down to
        # how long each one happened to fly. This is the line that moved
        # citynav_train_seen_2026-07-15_12-45-54 from SR_eff 2/10 to 3/10.
        #
        # Infra rows are not real attempts: the episode never got a fair
        # run, so it is re-queued. The reader drops them from the scored
        # set -- an infra casualty must not depress SR_eff as a false zero
        # -- and the audit counts come off the same parse.
        attempts, stats = _rows.read_attempts([self.episodes_file], tier="ledger")
        rows = [a.row for a in _rows.pick_attempts(attempts).values()]
        # Audit count, not a score: break the excluded set down by cause so
        # a systematic instant-failure (which shrinks the SR denominator
        # while stalling coverage) is loud instead of a quiet exclusion.
        infra_causes: dict[str, int] = {}
        for _r in _rows.iter_json_rows(self.episodes_file):
            if _r.get("status") == episodes.STATUS_INFRA:
                _c = str(_r.get("error") or "?")
                infra_causes[_c] = infra_causes.get(_c, 0) + 1
        n = len(rows)
        sr = [
            float(m["SR"])
            for r in rows
            for m in [_rows.metrics(r)]
            if isinstance(m.get("SR"), (int, float))
        ]
        osr = [
            float(m["OSR"])
            for r in rows
            for m in [_rows.metrics(r)]
            if isinstance(m.get("OSR"), (int, float))
        ]
        ne = [
            float(m["NE"])
            for r in rows
            for m in [_rows.metrics(r)]
            if isinstance(m.get("NE"), (int, float))
        ]
        n_succ = sum(1 for x in sr if x >= 1)
        n_pred = len(sr)
        elapsed_h = (time.monotonic() - self._t_start) / 3600.0
        lines = [
            "full-set continuous benchmark",
            f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"criterion: {_success_criterion()}",
            f"split: {self.split}   workers: {self.workers}   "
            f"episode_cap: {self.episode_timeout_s}s",
            f"wall budget: {self.max_wall_s}s "
            f"(elapsed {time.monotonic() - self._t_start:.0f}s)"
            if self.max_wall_s > 0
            else "",
            "",
            f"coverage:   {n} / {self._split_size} episodes "
            f"({100 * n / self._split_size:.1f}%)"
            if self._split_size
            else f"coverage: {n}",
            f"with pred:  {n_pred}   no pred: {n - n_pred}",
            f"infra (re-queued, not scored): {stats.infra}",
            "  by cause: "
            + ", ".join(f"{c}:{k}" for c, k in sorted(infra_causes.items(), key=lambda x: -x[1]))
            if infra_causes
            else "",
            f"!! {infra_causes.get('no_prediction_unclassified', 0)} unrecognized instant "
            "failures excluded — investigate before trusting coverage"
            if infra_causes.get("no_prediction_unclassified", 0) > max(2, n // 50)
            else "",
            self._persistent_infra(self._selection),
            f"!! ABORTED: {self._abort}" if self._abort else "",
            f"SR_eff:     {n_succ}/{n} = {n_succ / n:.4f}" if n else "SR_eff: -",
            f"SR(w/pred): {n_succ}/{n_pred} = {n_succ / n_pred:.4f}"
            if n_pred
            else "SR(w/pred): -",
            f"OSR:        {sum(osr) / len(osr):.4f}" if osr else "OSR: -",
            f"mean_NE:    {sum(ne) / len(ne):.2f} m "
            f"(straight line from the final commit to the goal; no path term)"
            if ne
            else "mean_NE: -",
            f"throughput: {self._session_done / elapsed_h:.2f} ep/h "
            f"(this driver session: {self._session_done} rows)"
            if elapsed_h > 0
            else "",
        ]
        if self._split_size and n < self._split_size and n > 0 and elapsed_h > 0:
            eta_h = (self._split_size - n) / (n / elapsed_h)
            lines.append(
                f"ETA to full coverage: ~{eta_h:.1f} h "
                f"({(self._split_size - n)} episodes left)"
            )
        self.summary_file.write_text("\n".join(x for x in lines if x != "") + "\n")

    # -- main loop ---------------------------------------------------------

    def worker_loop(self, worker_id: int) -> None:
        # One persistent run_benchmark --episode-server per worker. Lazily
        # spawned on the first episode (so Driver.run's staggered 2 s
        # thread starts also stagger the cold starts), reused for every
        # subsequent episode, respawned by _run_attempt if it dies, and
        # shut down when the shared selection is globally exhausted.
        runner = _WarmRunner(self.out_root / f"w{worker_id}", self.split)
        try:
            while True:
                eid = self._next_claimed_episode(worker_id)
                if eid is None:
                    return
                # Write-once backstop (standalone only): a claim for an
                # episode already completed this run is dropped loudly
                # instead of flown — a re-fly would append a duplicate row
                # whose replicate outcome is a coin flip under reasoner
                # nondeterminism. Unreachable while _next_claimed_episode
                # and _append_result share _dispatch_lock; kept as the
                # guard against a future claim path that forgets.
                with self._dispatch_lock:
                    dup = eid in self._done_cache
                    if dup:
                        self._in_flight.discard(eid)
                if dup:
                    self.log(
                        f"[w{worker_id}] DUP-CLAIM {eid}: result row "
                        "already written this run; dropping claim"
                    )
                    continue
                try:
                    self._wait_for_vllm()
                    self.run_one(worker_id, eid, runner)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[w{worker_id}] FAIL {eid}: {type(exc).__name__}: {exc}")
                    try:
                        self._append_result(
                            {
                                "episode_id": eid,
                                "worker_id": worker_id,
                                "status": episodes.STATUS_INFRA,
                                "error": f"{type(exc).__name__}: {exc}",
                                "ts": round(time.time(), 1),
                            }
                        )
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
                    # The episode is no longer running here, however it
                    # ended. _append_result already added its qualifying
                    # row to the done cache, so releasing the hold cannot
                    # re-offer it; infra rows did not qualify and are
                    # re-claimable by design.
                    with self._dispatch_lock:
                        self._in_flight.discard(eid)
        finally:
            runner.shutdown()

    def _select_episodes(self) -> list[str]:
        episodes = _enumerate_split(self.split)
        # Optional allowlist filter applied BEFORE the seeded shuffle so
        # an (allowlist, seed) pair picks the same N episodes (matched-
        # pair power for ablations).
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
        # Shuffle the full block-ordered list with the fixed seed BEFORE
        # --limit, so a given (seed, limit) is the SAME N episodes (the
        # correct A/B subset) and any partial run is a representative
        # random sample, not block-0-biased.
        random.Random(self.sample_seed).shuffle(episodes)
        # Optional disjoint-slice offset (applied AFTER shuffle, BEFORE
        # limit) so two clusters sharing a seed run non-overlapping
        # slices of the same fixed subset.
        if self.sample_offset > 0:
            episodes = episodes[self.sample_offset :]
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
        self._split_size = len(episodes)
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
        # The run's ONE machine-written record, emitted with everything
        # resolved and nothing flown: measured endpoint identity,
        # inlined config, tiered env. Re-entry into this out_root (a resume)
        # verifies identity against the stored manifest and
        # REFUSES drift — a resumed run under a different bake/model/config
        # is two runs wearing one run root's name.
        from benchmarks.citynav import run_manifest as _rm

        try:
            fresh_manifest = _rm.build_manifest(
                    out_root=self.out_root,
                config_path=CONFIG,
                split=self.split,
                sample_seed=self.sample_seed,
                sample_offset=self.sample_offset,
                limit=self.limit,
                workers=self.workers,
                allowlist_path=self.allowlist_path,
                episodes=episodes,
            )
            if self.replay_source is not None:
                _rm.verify_replay(self.replay_source, fresh_manifest)
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
                if not self.loop:
                    return 0
                self.log("--loop set; archiving results and re-running the split")
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.episodes_file.rename(
                    self.episodes_file.with_suffix(f".jsonl.pass{pass_idx}.{ts}")
                )
                continue
            threads = []
            for wid in range(self.workers):
                t = threading.Thread(target=self.worker_loop, args=(wid,), daemon=False)
                t.start()
                threads.append(t)
                time.sleep(2.0)
            for t in threads:
                t.join()
            self.log(f"pass {pass_idx} drained")
            # Mass instant failure detected mid-pass: the workers have
            # drained, so report and leave. Nonzero so a launcher (and an
            # autoresearch loop that queues arm after arm) sees a failed
            # run rather than a suspiciously fast successful one.
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
        "--sample-offset",
        type=int,
        default=int(os.environ.get("BENCH_SAMPLE_OFFSET", "0")),
        help="skip the first N episodes of the seeded shuffle BEFORE "
        "--limit (default from BENCH_SAMPLE_OFFSET, else 0). With a "
        "shared --sample-seed, (offset, limit) carves a disjoint slice "
        "of the same fixed subset so two clusters can split one run "
        "(e.g. offset=0 limit=400 vs offset=400 limit=100).",
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
        "--loop",
        action="store_true",
        help="after full coverage, archive results and run the split again",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print seed, selected episodes and done/pending; launch nothing",
    )
    p.add_argument(
        "--from-manifest",
        default=None,
        help="replay a recorded run: load its run_manifest.json, force the "
        "recorded selection (split/seed/offset/limit/allowlist), and gate "
        "the re-derived live facts (git sha, config sha, bake, served "
        "model, selection sha) against the record, refusing loud on any "
        "mismatch. A git mismatch is reconciled by hand (no auto-checkout).",
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
    replay_source = None
    if args.from_manifest:
        from benchmarks.citynav import run_manifest as _rm

        try:
            replay_source = _rm.apply_replay_args(args)
        except _rm.ManifestIdentityError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
    driver = Driver(args)
    driver.replay_source = replay_source
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
