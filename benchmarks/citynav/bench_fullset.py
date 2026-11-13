"""Continuous full-split benchmark driver: one ``run_benchmark --attempt-root`` child per attempt.

The split is shuffled by ``--sample-seed`` (after ``--episode-allowlist``, before
``--limit``), so a ``(seed, limit)`` pair always selects the same episodes. A
re-run into the same ``--out-root`` resumes: an episode with a non-infra row in
``episodes.jsonl`` is done, and an infra attempt is flown again up to
``--max-infra-attempts``. The run stops when nothing is left, the wall budget
(``--max-wall-s``) is spent, or a window of infra casualties aborts it (exit 3).
The out-root's layout is in ``benchmarks/citynav/README.md``.
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

from benchmarks import sandbox
from benchmarks.bench import Bench, missing_bake
from benchmarks.citynav.adapter import episode_ids, load_split
from benchmarks.citynav.analysis import run_report
from benchmarks.citynav.paths import selected_bench
from benchmarks.run_record import episodes
from benchmarks.run_record import rows as _rows
from benchmarks.run_record import run_manifest as _rm
from benchmarks.run_record.config import resolve_config
from maps.llm.endpoint_spec import models_url
from maps.settings import from_environ

REPO = Path(__file__).resolve().parents[2]
#: The config a run flies when ``CITYNAV_BENCH_CONFIG`` names none.
DEFAULT_CONFIG = "configs/citynav.yaml"

# The runner each attempt child runs.
RUNNER_MODULE = "benchmarks.citynav.run_benchmark"
# Share of the episode cap held back from flying, so the model still has
# turns to reply and commit after the mission's wall-clock backstop fires.
_DEADLINE_RESERVE_FRAC = 0.1
# How often the driver asks the reasoner's ``/models`` and the SAM replicas'
# ``/health`` while an attempt flies; two failures in a row kill the attempt
# as ``reasoner_down_midrun`` or ``sam_down_midrun``: an episode flying blind
# spends reasoner calls on a result it cannot keep.
REASONER_PROBE_S = 60.0
# How often a launch held for a reasoner or a segmenter that is down asks again.
REASONER_WAIT_S = 120.0
# Pause between worker thread starts, so the first attempts do not all
# cold-start at once.
WORKER_STAGGER_S = 2.0
# The run aborts when ABORT_FRACTION of the last ABORT_WINDOW completions are
# infra casualties (a missing tile bake fails every episode in seconds); the
# window must fill first.
ABORT_WINDOW = 12
ABORT_FRACTION = 0.9
# The pre-limit shuffle's seed when --sample-seed is not given.
SAMPLE_SEED = 1234


_log_lock = threading.Lock()


def _utc(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _config_path(environ) -> Path:
    """The run's config: ``CITYNAV_BENCH_CONFIG`` under the repo, else ``DEFAULT_CONFIG``."""
    return REPO / (environ.get("CITYNAV_BENCH_CONFIG", "").strip() or DEFAULT_CONFIG)


def _enumerate_split(bench: Bench, split: str) -> list[str]:
    """The split's episode ids; an unknown split raises ``ValueError``."""
    return episode_ids(enumerate(load_split(bench.split_path(split))))


# One thread per native pool (BLAS, OpenCV, Rayon) and two malloc arenas per
# attempt child: uncapped, every child starts a thread per core, and a thread
# that fails to start crashes its episode. A value the shell set wins.
RUNNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
    "RAYON_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
}


def _runner_env(bench: Bench) -> dict[str, str]:
    """The environment an attempt child of ``bench`` is spawned under: this
    process's, under the run's own (the bench's selection, the settled effort)."""
    env = {**RUNNER_THREAD_ENV, **os.environ, **bench.environ}
    # Unbuffered, so a child that dies fast still leaves its lines in server.log.
    env["PYTHONUNBUFFERED"] = "1"
    env[episodes.PARENT_PID_ENV] = str(os.getpid())
    return env


def _infra_cause(row: dict) -> str:
    """What killed this infra attempt: its ``error``."""
    return str(row.get("error") or "").strip() or "infra"


def _causes(counts: Counter) -> str:
    """Infra causes with their counts, commonest first: ``cause (n), ...``."""
    return ", ".join(f"{c} ({k})" for c, k in counts.most_common())


def _offline(environ) -> bool:
    """``SEARCH_ENDPOINT_OFFLINE=1``: the run serves nothing (a pinned spec,
    ``endpoint_spec.discover``), so the driver probes nothing."""
    return environ.get("SEARCH_ENDPOINT_OFFLINE", "").strip() == "1"


def _answers(url: str, bearer: str | None = None) -> bool:
    """True when a GET of ``url`` (``bearer`` as an Authorization header)
    gets a 2xx or 3xx within 10 s."""
    try:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {bearer}"} if bearer else {}
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return 200 <= resp.status < 400
    except (OSError, ValueError):
        return False


def _missing_block_assets(bench: Bench, episode_ids: list[str]) -> list[str]:
    """What the selected episodes' blocks need and do not have: the bake
    (``bench.missing_bake``), and ``<map>.tif`` beside ``<map>_rgb.tif`` or
    ``<map>.png``, each raster line with the command that builds it."""
    build = (
        "python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <calcity_rgb.tif>"
        if bench.name == "calcity"
        else "benchmarks/citynav/data_prep/rasterize_citynav_native.sh"
    )
    missing: list[str] = []
    for map_name in sorted({e.split("::", 1)[0] for e in episode_ids}):
        if why := missing_bake(bench.tile_stem(map_name)):
            missing.append(why)
        rgbd = Path(bench.rgbd_dir)
        if not (rgbd / f"{map_name}.tif").exists():
            missing.append(f"Missing {rgbd / f'{map_name}.tif'} (built by {build})")
        if not any((rgbd / f"{map_name}{s}").exists() for s in ("_rgb.tif", ".png")):
            missing.append(f"Missing {rgbd / f'{map_name}_rgb.tif'} (built by {build})")
    return missing


class Driver:
    #: Whether each attempt flies in a systemd user scope; settled once by _run().
    _scope = False

    def __init__(self, args: argparse.Namespace, bench: Bench, config: Path) -> None:
        self.bench = bench
        self.config = Path(config)
        #: How the reasoner is reached, for the driver's own probes.
        self.endpoint = from_environ(bench.environ).endpoint
        self.workers = args.workers
        self.split = args.split
        self.episode_timeout_s = args.episode_timeout_s
        self.max_infra_attempts = args.max_infra_attempts
        self.memory_gib = args.attempt_memory_gib
        # Whether attempts fly under bubblewrap; settled once by _run().
        self._bwrap = False
        self._infra_errors: dict[str, Counter] = {}
        self.out_root = Path(args.out_root).resolve()
        self.episodes_file = self.out_root / episodes.EPISODES_JSONL
        self.driver_log = self.out_root / "driver.log"
        self.summary_file = self.out_root / "summary.txt"
        self.limit = args.limit
        self.sample_seed = args.sample_seed
        self.allowlist_path = (
            Path(args.episode_allowlist).resolve() if args.episode_allowlist else None
        )
        # Fixed by main() before anything runs; guarded by _dispatch_lock.
        self._selection: list[str] = []
        # The allowlist's count line, for the preview or the driver log.
        self._selection_note: str | None = None
        self._cursor = 0
        self._dispatch_lock = threading.Lock()
        self._done_cache: set[str] = set()
        self._in_flight: set[str] = set()
        self._recent: deque[tuple[str | None, str]] = deque(maxlen=ABORT_WINDOW)
        self._abort: str | None = None
        # --max-wall-s: stop claiming once spent; in-flight attempts finish.
        self.max_wall_s = args.max_wall_s
        self._deadline = time.monotonic() + self.max_wall_s if self.max_wall_s > 0 else None

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with _log_lock:
            print(line, flush=True)
            with self.driver_log.open("a") as f:
                f.write(line + "\n")

    def _absorb(self, row: dict) -> None:
        """Count a ledger row: its episode is done, or it is one more infra
        attempt (``run_record.rows.is_infra_abort``: excluded from SR and
        re-queued; a timeout is a miss, not one) with its cause."""
        eid = row.get("episode_id")
        if not isinstance(eid, str):
            return
        with self._dispatch_lock:
            if _rows.is_infra_abort(row):
                self._infra_errors.setdefault(eid, Counter())[_infra_cause(row)] += 1
            else:
                self._done_cache.add(eid)

    def _absorb_ledger(self) -> None:
        """Count every row this root's ledger already holds."""
        for r in _rows.iter_json_rows(self.episodes_file):
            self._absorb(r)

    def _append_result(self, row: dict) -> None:
        # The write lands before the done-cache add, so a finished episode is
        # never claimed again.
        episodes.append_row(self.out_root, row)
        self._absorb(row)
        self._check_mass_infra(row)

    @staticmethod
    def _log_excerpt(log_path: str) -> str:
        """The tail of a worker log, indented, for the abort message."""
        try:
            text = Path(log_path).read_text(errors="replace")
        except OSError as exc:
            return f"    (no readable log at {log_path}: {exc})"
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return f"    ({log_path} is empty)"
        return "\n".join(f"    | {ln}" for ln in lines[-15:])

    def _check_mass_infra(self, row: dict) -> None:
        """Stop claiming once ABORT_FRACTION of a full ABORT_WINDOW are infra
        casualties; every counted row is already re-queued for a resume."""
        cause = _infra_cause(row) if _rows.is_infra_abort(row) else None
        # The log that explains the row: the attempt's own run.log, or the
        # worker's server.log for an attempt that never got a workdir.
        wd = row.get("workdir")
        log_path = (
            self.out_root / wd / episodes.RUN_LOG
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
            log_path = next(lp for c, lp in reversed(bad) if c == top)
            self._abort = (
                f"mass instant failure: {len(bad)} of the last "
                f"{ABORT_WINDOW} attempts produced no prediction and were "
                f"excluded as infra casualties (dominant cause {top!r}, "
                f"{k}x). This run is not producing results; stopping instead "
                f"of spending the rest of the allocation on it."
            )
        self.log("!! ABORTING RUN: " + self._abort)
        self.log(f"!! by cause: {_causes(counts)}")
        self.log(f"!! representative worker log: {log_path}")
        self.log(self._log_excerpt(log_path))
        scored = len(self._done_cache)
        self.log(
            ("!! nothing was scored" if not scored else f"!! {scored} episode(s) were scored")
            + "; every infra attempt is excluded, so a resume into this "
            "out-root re-flies those still under --max-infra-attempts once "
            "the cause is fixed."
        )

    def _wall_spent(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def _next_claimed_episode(self) -> str | None:
        """The next episode not done, not in flight and not past its infra
        attempts; ``None`` once nothing is left, the wall budget is spent, or
        the run aborted (in-flight attempts finish)."""
        if self._wall_spent() or self._abort is not None:
            return None
        with self._dispatch_lock:
            for _ in range(len(self._selection)):
                eid = self._selection[self._cursor % len(self._selection)]
                self._cursor += 1
                if eid in self._in_flight or not self._left(eid):
                    continue
                self._in_flight.add(eid)
                return eid
        return None

    def _infra_attempts(self, eid: str) -> int:
        return self._infra_errors.get(eid, Counter()).total()

    def _left(self, eid: str) -> bool:
        """Still to fly: not done and not past its infra attempts."""
        return eid not in self._done_cache and self._infra_attempts(eid) < self.max_infra_attempts

    def _persistent_infra(self) -> str:
        """The line naming the selected episodes left out after
        ``max_infra_attempts`` infra attempts, errors counted, or ""."""
        with self._dispatch_lock:
            stuck = [e for e in self._selection if e not in self._done_cache and not self._left(e)]
            errors: Counter = Counter()
            for e in stuck:
                errors.update(self._infra_errors[e])
        if not stuck:
            return ""
        return (
            f"persistent infra: {len(stuck)} episode(s) after "
            f"{self.max_infra_attempts} infra attempts each ({_causes(errors)}); "
            f"--max-infra-attempts above {self.max_infra_attempts} flies them again"
        )

    def _reasoner_answers(self) -> bool:
        """True when the bench's spec's reasoner answers its model list, at
        the URL discovery read it from (``endpoint_spec.models_url``), asked
        with the chat path's bearer (hosted endpoints auth ``/v1/models``)."""
        spec = self.bench.endpoint_spec
        return _answers(models_url(spec.base_url, self.endpoint), self.endpoint.reasoner_api_key)

    def _sam_answers(self) -> bool:
        """True when a SAM replica discovery health-checked answers ``/health``."""
        return any(_answers(f"{url}/health") for url in self.bench.endpoint_spec.sam_urls)

    def _wait_for_reasoner(self) -> bool:
        return self._wait_for("reasoner", self._reasoner_answers)

    def _wait_for_sam(self) -> bool:
        return self._wait_for("sam", self._sam_answers)

    def _wait_for(self, name: str, answers) -> bool:
        """Hold a launch until ``answers()``; False once the wall budget is
        spent or the run aborted while it waited."""
        if _offline(self.bench.environ) or answers():
            return True
        self.log(f"{name} down; pausing launches until it answers")
        while not answers():
            if self._wall_spent() or self._abort is not None:
                return False
            left = (
                self._deadline - time.monotonic() if self._deadline is not None else REASONER_WAIT_S
            )
            time.sleep(max(0.0, min(REASONER_WAIT_S, left)))
        self.log(f"{name} back; resuming")
        return True

    def _wait(self, proc: subprocess.Popen) -> str:
        """Block until the child exits (``exited``), the episode cap passes
        (``timeout``) or the reasoner or SAM fails two probes in a row
        (``<name>_down_midrun``)."""
        deadline = time.monotonic() + self.episode_timeout_s
        down = {"reasoner": 0, "sam": 0}
        while (left := deadline - time.monotonic()) > 0:
            try:
                proc.wait(timeout=min(REASONER_PROBE_S, left))
                return "exited"
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                break
            if _offline(self.bench.environ):
                continue
            for name, answers in (("reasoner", self._reasoner_answers), ("sam", self._sam_answers)):
                down[name] = 0 if answers() else down[name] + 1
                if down[name] >= 2:
                    return f"{name}_down_midrun"
        return "timeout"

    def _run_attempt(self, worker_id: int, episode_id: str) -> dict:
        """One attempt in a fresh child: its ledger row, the child's
        ``episode.json`` plus ``worker_id``, ``workdir`` and the wall measured
        here."""
        t0 = time.monotonic()
        # The wall clock beside it: when the attempt began and ended, in UTC, so a batch's
        # searches per hour is read off its rows.
        started = time.time()
        worker_root = self.out_root / f"w{worker_id}"
        worker_root.mkdir(parents=True, exist_ok=True)
        env = _runner_env(self.bench)
        # A hosted reasoner's prompt cache matches within one key: a worker's
        # episodes, flown one after another, share theirs, so each episode
        # after a worker's first reads the common head (system turn and tools)
        # instead of writing it, at a worker's request rate, under the ~15 a
        # minute a key holds.
        env["SEARCH_LLM_CACHE_KEY"] = f"{self.out_root.name}/w{worker_id}"
        # The mission's wall-clock backstop: an absolute epoch from the cap's
        # start, less the reserve for ending.
        env[episodes.DEADLINE_ENV] = repr(
            time.time() + self.episode_timeout_s * (1.0 - _DEADLINE_RESERVE_FRAC)
        )
        cmd = [
            sys.executable,
            "-m",
            RUNNER_MODULE,
            "--config",
            str(self.config),
            "--split",
            self.split,
            "--episode-id",
            episode_id,
            "--attempt-root",
            str(worker_root),
            "--memory-gib",
            repr(self.memory_gib),
        ]
        if self._bwrap:
            # The child's parent is bubblewrap's init, PID 1 in the attempt's namespace.
            env[episodes.PARENT_PID_ENV] = "1"
            # The driver's discovery cached the tokenizer; the hub cache is read-only inside.
            env["HF_HUB_OFFLINE"] = "1"
            cmd = sandbox.bwrap_argv(*self._attempt_binds(worker_root)) + cmd
        # A unit name unique on the host, so the scope is killed by name with the attempt.
        unit = f"maps-attempt-{os.getpid()}-w{worker_id}-{time.time_ns()}"
        if self._scope:
            cmd = sandbox.scope_argv(unit, self.memory_gib) + cmd
        # The child prints its workdir as its first stdout line; an unlinked
        # file holds it, so no pipe can fill and stall the child.
        with tempfile.TemporaryFile("w+") as out, (worker_root / "server.log").open("a") as err:
            try:
                # Its own session, so a kill takes the whole process group.
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO),
                    stdout=out,
                    stderr=err,
                    text=True,
                    start_new_session=True,
                    env=env,
                )
            except OSError as exc:
                return {
                    "episode_id": episode_id,
                    "worker_id": worker_id,
                    "status": episodes.STATUS_INFRA,
                    "error": f"attempt spawn: {type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - t0, 1),
                }
            verdict = self._wait(proc)
            if verdict != "exited":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if self._scope:
                sandbox.kill_scope(unit)
            out.seek(0)
            first = out.readline().strip()
        elapsed = time.monotonic() - t0
        workdir = Path(first) if first else None
        landed = episodes.read_episode_json(workdir) if workdir else None
        where = {"workdir": str(workdir.relative_to(self.out_root))} if workdir else {}
        crashed = verdict == "exited" and landed is None and proc.returncode != 0
        lost = verdict if verdict.endswith("_down_midrun") else None
        if lost:
            self.log(
                f"[w{worker_id}] {lost.split('_')[0].upper()}-DEAD mid-episode {episode_id} after "
                f"{elapsed:.0f}s; killed + re-queued (not scored)"
            )
        elif verdict == "timeout":
            self.log(f"[w{worker_id}] TIMEOUT {episode_id} after {elapsed:.0f}s; killed")
        elif crashed:
            # Killed by a signal (a cgroup OOM is SIGKILL) or exited with an error before landing a record.
            self.log(
                f"[w{worker_id}] CRASHED mid-episode {episode_id} after "
                f"{elapsed:.0f}s (exit {proc.returncode}); re-queued (not scored)"
            )
        if lost or crashed:
            # Re-queued; a lost service's attempt is pending until it is back.
            return {
                "episode_id": episode_id,
                "worker_id": worker_id,
                "status": episodes.STATUS_INFRA,
                "elapsed_s": round(elapsed, 1),
                **where,
                "error": lost
                or (
                    "killed_midrun"
                    if proc.returncode < 0
                    else f"exited {proc.returncode} with no record"
                ),
            }
        # The child's own verdict decides (dispatch classifies the episode's
        # exception chain); the driver adds only what it alone saw.
        row = {
            **(landed or {"episode_id": episode_id}),
            "worker_id": worker_id,
            "elapsed_s": round(elapsed, 1),
            **where,
            "started_at_utc": _utc(started),
            "finished_at_utc": _utc(started + elapsed),
        }
        if landed is None and verdict == "exited":
            # A child that exits clean with no record failed, unrecognised.
            row |= {"status": episodes.STATUS_INFRA, "error": "no_prediction_unclassified"}
        elif row.get("status") != episodes.STATUS_INFRA:
            # A timeout is counted apart; its record is scored as recorded, and none is a miss.
            row["status"] = (
                episodes.STATUS_TIMED_OUT if verdict == "timeout" else episodes.STATUS_OK
            )
        return row

    def _attempt_binds(self, worker_root: Path) -> tuple[list, list]:
        """What an attempt writes (its worker's attempt root, the ortho cache beside
        the bench's rasters) and reads besides the repo (the bench's inputs, the
        config, the hub cache)."""
        writable = [worker_root, *sandbox.ortho_caches([self.bench.rgbd_dir])]
        return writable, [*sandbox.bench_inputs(self.bench), self.config, sandbox.hf_hub_cache()]

    def run_one(self, worker_id: int, episode_id: str) -> None:
        """Fly one attempt and land its row."""
        self.log(f"[w{worker_id}] START {episode_id}")
        row = self._run_attempt(worker_id, episode_id)
        m = row.get("metrics") or {}
        summary = (
            ""
            if _rows.is_infra_abort(row)
            else f" NE={m.get('NE')} SR={m.get('SR')}"
            if m
            else " MISSING"
        )
        self.log(
            f"[w{worker_id}] END {episode_id} status={row['status']} "
            f"elapsed={row['elapsed_s']:.0f}s{summary}"
        )
        self._append_result(row)
        # ponytail: re-reads the whole ledger per row (O(n^2) over a run, cheap
        # beside an episode); refresh every N rows if the ledger grows past that.
        self._write_summary()

    def _write_summary(self) -> None:
        """``summary.txt``: the driver's infra and abort lines above
        ``analysis.run_report.report`` over the root, once a row has landed."""
        infra = Counter(
            _infra_cause(r)
            for r in _rows.iter_json_rows(self.episodes_file)
            if _rows.is_infra_abort(r)
        )
        lines = [
            f"split: {self.split}   workers: {self.workers}   "
            f"episode_cap: {self.episode_timeout_s}s",
            f"infra attempts (excluded): {infra.total()}"
            + (f"  by cause: {_causes(infra)}" if infra else ""),
            self._persistent_infra(),
            f"!! ABORTED: {self._abort}" if self._abort else "",
        ]
        buf = io.StringIO()
        if not self.episodes_file.is_file():
            self.summary_file.write_text("\n".join(x for x in lines if x) + "\n")
            return
        # Under the log lock: redirect_stdout is process-wide, and log() is
        # the driver's only other stdout writer.
        with _log_lock, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                run_report.report(
                    self.out_root,
                    run_report.benches(self.bench.environ, first=self.bench),
                    allow_partial=True,
                )
            except SystemExit as exc:  # no scored row yet
                print(exc)
        text = "\n".join(x for x in lines if x) + "\n" + buf.getvalue().lstrip("\n")
        self.summary_file.write_text(text)

    def worker_loop(self, worker_id: int) -> None:
        while (eid := self._next_claimed_episode()) is not None:
            try:
                if self._wait_for_reasoner() and self._wait_for_sam():
                    self.run_one(worker_id, eid)
            except Exception as exc:
                self.log(f"[w{worker_id}] FAIL {eid}: {type(exc).__name__}: {exc}")
                try:
                    self._append_result(
                        {
                            "episode_id": eid,
                            "worker_id": worker_id,
                            "status": episodes.STATUS_INFRA,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                except OSError as werr:
                    # No worker can land a result: the run stops.
                    with self._dispatch_lock:
                        self._abort = self._abort or (
                            f"[w{worker_id}] cannot write {self.episodes_file}: {werr}"
                        )
                    self.log(f"!! ABORTING RUN: {self._abort}")
                    return
            finally:
                with self._dispatch_lock:
                    self._in_flight.discard(eid)

    def _select_episodes(self) -> list[str]:
        """The selection, in flying order; a refusal raises ``ValueError``
        naming it. An allowlist's count line is kept in ``_selection_note``."""
        episodes = _enumerate_split(self.bench, self.split)
        if self.allowlist_path is not None:
            try:
                lines = self.allowlist_path.read_text().splitlines()
            except OSError as exc:
                raise ValueError(
                    f"--episode-allowlist read failed: {self.allowlist_path}: {exc}"
                ) from exc
            allowed = {ln.strip() for ln in lines if ln.strip()}
            if not allowed:
                raise ValueError(f"--episode-allowlist names no episode: {self.allowlist_path}")
            before = len(episodes)
            episodes = [e for e in episodes if e in allowed]
            unknown = sorted(allowed - set(episodes))
            if unknown:
                raise ValueError(
                    f"--episode-allowlist names {len(unknown)} ids not in the "
                    f"{self.split} split, e.g. {unknown[0]}"
                )
            self._selection_note = (
                f"--episode-allowlist {self.allowlist_path}: "
                f"{len(allowed)} ids requested -> {len(episodes)} match "
                f"the {self.split} split (from {before} total)"
            )
        random.Random(self.sample_seed).shuffle(episodes)
        if self.limit > 0:
            episodes = episodes[: self.limit]
        return episodes

    def dry_run(self) -> int:
        """Selection preview. Launches nothing and writes nothing."""
        episodes = self._selection
        if self._selection_note:
            print(self._selection_note)
        self._absorb_ledger()
        done = self._done_cache
        pending = [e for e in episodes if self._left(e)]
        print(f"sample_seed: {self.sample_seed}   split: {self.split}   limit: {self.limit}")
        print(
            f"selected: {len(episodes)}   done: "
            f"{len(done & set(episodes))}   pending: {len(pending)}"
        )
        print(f"first {min(20, len(episodes))} selected (post-shuffle):")
        for e in episodes[:20]:
            mark = "done" if e in done else "PEND" if e in pending else "infra"
            print(f"  [{mark}] {e}")
        return 0

    def run(self) -> int:
        self.out_root.mkdir(parents=True, exist_ok=True)
        try:
            lock = episodes.hold_root(self.out_root)
        except episodes.RootHeld as exc:
            self.log(f"FATAL: {exc}")
            return 2
        try:
            return self._run()
        except KeyboardInterrupt:
            # The ledger keeps every landed row, so a relaunch into this root resumes.
            self.log("interrupted (SIGINT): stopping; relaunch into this root to resume")
            return 130
        finally:
            lock.close()

    def _run(self) -> int:
        episodes = self._selection
        if self._selection_note:
            self.log(self._selection_note)
        # Written before anything flies; a resume under another identity is
        # refused (run_manifest.emit_or_verify).
        try:
            fresh_manifest = _rm.build_manifest(
                bench=self.bench,
                out_root=self.out_root,
                config_path=self.config,
                split=self.split,
                sample_seed=self.sample_seed,
                limit=self.limit,
                allowlist_path=self.allowlist_path,
                episodes=episodes,
            )
            _rm.emit_or_verify(self.out_root, fresh_manifest, log=self.log)
        except _rm.ManifestIdentityError as exc:
            self.log(f"FATAL: {exc}")
            return 2
        # A root with a manifest is a run root, flown or not: its ledger exists.
        self.episodes_file.touch()
        self._absorb_ledger()
        pending = [e for e in episodes if self._left(e)]
        self.log(
            f"split={self.split} sample_seed={self.sample_seed} selection={len(episodes)} "
            f"done={len(self._done_cache & set(episodes))} pending={len(pending)} "
            f"workers={self.workers}"
        )
        self._bwrap, self._scope = sandbox.bwrap_works(), sandbox.scope_works()
        self.log(sandbox.mode_line(self._bwrap, self.memory_gib, scope=self._scope))
        # Workers claim until nothing is left, the wall budget is spent, or the
        # run aborts; an infra attempt is re-claimed by whichever worker is free.
        # Daemon workers end with the driver (their children die with it), so
        # none flies on once an interrupt has released the root.
        if pending and not self._wall_spent():
            threads = []
            for wid in range(self.workers):
                t = threading.Thread(target=self.worker_loop, args=(wid,), daemon=True)
                t.start()
                threads.append(t)
                time.sleep(WORKER_STAGGER_S)
            for t in threads:
                t.join()
        persistent = self._persistent_infra()
        if persistent:
            self.log(persistent)
        if self._abort is not None:
            # Nonzero, so a launcher sees a failed run rather than a fast one.
            self.log(f"ABORTED: {self._abort}")
            self._write_summary()
            return 3
        if self._wall_spent():
            self.log(
                f"wall budget reached ({self.max_wall_s}s); stopping "
                f"(results retained, aggregate whatever completed)"
            )
        else:
            self.log(f"nothing left to fly: {len(episodes)} episodes selected")
        self._write_summary()
        return 0


def _at_least_one(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def _build_parser(bench: Bench) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m benchmarks.{bench.name}.bench_fullset",
        description=f"Continuous full-split {bench.name} benchmark at fixed concurrency.",
    )
    p.add_argument("--split", default=bench.default_split, help=f"(default {bench.default_split})")
    p.add_argument(
        "--workers",
        type=_at_least_one,
        default=6,
        help="concurrency (default %(default)s)",
    )
    p.add_argument(
        "--out-root",
        default=f"output/{bench.name}_fullset",
        help="results + per-worker run-roots (default %(default)s)",
    )
    p.add_argument(
        "--episode-timeout-s",
        type=_at_least_one,
        default=4 * 3600,
        help="per-episode wall cap (default %(default)ss = 4h)",
    )
    p.add_argument(
        "--max-wall-s",
        type=int,
        default=0,
        help="wall budget in seconds (default 0 = none): past it no new "
        "episode is claimed; in-flight attempts finish and their rows land",
    )
    p.add_argument(
        "--max-infra-attempts",
        type=_at_least_one,
        default=3,
        help="infra attempts an episode gets before the run leaves it out and "
        "reports it as persistent infra (default %(default)s)",
    )
    p.add_argument(
        "--attempt-memory-gib",
        type=sandbox.gib,
        default=sandbox.ATTEMPT_MEMORY_GIB,
        help="each attempt's memory cap in GiB (default %(default)s): a block that allocates "
        "past it raises MemoryError, served as the block's error",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="episodes kept after the seeded shuffle (default 0 = all); with "
        "one --sample-seed, the same N episodes on every run",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=SAMPLE_SEED,
        help=f"seed for the pre-limit episode shuffle (default {SAMPLE_SEED}). "
        "Fixes WHICH N episodes run so "
        "an A/B across code versions is on the identical subset.",
    )
    p.add_argument(
        "--episode-allowlist",
        default=None,
        help="path to a newline-delimited file of episode_ids. When set, "
        "only those ids run; every id must be in the split, else the run is "
        "refused. The seeded shuffle + --limit still apply on top of the allowlist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print seed, selected episodes and done/pending; launch nothing",
    )
    return p


def main(argv: list[str] | None = None, *, bench: Bench | None = None) -> int:
    """The driver's CLI, for ``bench``; unset, for the bench the environment selects."""
    # Discovery logs at INFO (the spec the run flies); without a config it reaches no console.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    if bench is None:
        bench = selected_bench(os.environ)
    config = _config_path(bench.environ)
    parser = _build_parser(bench)
    args = parser.parse_args(argv)
    # Each refusal comes before an out-root exists or a serve is asked: the config, the
    # bench's data, the selection and its block assets, then the serve, once, so an
    # unreachable or wrong one fails before any worker spawns.
    try:
        resolve_config(config)
        bench.assert_available()
        driver = Driver(args, bench, config)
        driver._selection = driver._select_episodes()
        if missing := _missing_block_assets(bench, driver._selection):
            raise ValueError(
                "block assets missing for the selection (every episode on these blocks would "
                "land infra):\n" + "\n".join(f"  {p}" for p in missing)
            )
        if args.dry_run:
            return driver.dry_run()
        driver.bench = _rm.install_endpoint(bench, from_environ(bench.environ))
    # EndpointRefused is a RuntimeError.
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
