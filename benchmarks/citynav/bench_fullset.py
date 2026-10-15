"""Continuous full-split benchmark driver: one ``run_benchmark --attempt-root`` child per attempt.

The split is shuffled by ``--sample-seed`` (after ``--episode-allowlist``, before
``--limit``), so a ``(seed, limit)`` pair always selects the same episodes. A
re-run into the same ``--out-root`` resumes: an episode with a non-infra row in
``episodes.jsonl`` is done, and an infra attempt is flown again up to
``--max-infra-attempts``. The run stops when nothing is left, the wall budget
(``--max-wall-s``) is spent, or a window of infra casualties aborts it (exit 3).
The out-root's layout is in ``benchmarks/citynav/README.md``.

``--arrivals <stream.json>`` (``benchmarks.capacity.streams``) flies a
capacity session instead: the stream's episodes in arrival order, each
claimable only once its ``arrival_s`` has passed on the session clock, so a
report that lands while every worker is busy waits its turn. Each row then
carries ``arrival_s`` and ``claimed_s`` (session seconds), and the stream is
copied to ``<out-root>/arrivals.json`` for ``benchmarks.capacity.reader``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import logging
import shutil
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

from benchmarks.citynav import episodes, paths
from benchmarks.citynav.analysis import rows as _rows

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / (os.environ.get("CITYNAV_BENCH_CONFIG", "").strip() or "configs/citynav.yaml")

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


def _enumerate_split(split: str) -> list[str]:
    from benchmarks.citynav.adapter import episode_ids, load_split

    try:
        split_file = paths.split_path(split)
    except ValueError as exc:
        raise SystemExit(f"FATAL: {exc}") from None
    return episode_ids(enumerate(load_split(split_file)))


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


def _runner_env() -> dict[str, str]:
    """The environment an attempt child is spawned under."""
    env = {**RUNNER_THREAD_ENV, **os.environ}
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


def _missing_block_assets(episode_ids: list[str]) -> list[Path]:
    """The files the selected episodes' blocks need and do not have: the
    bake's per-block files, and ``<map>.tif`` beside ``<map>_rgb.tif`` or
    ``<map>.png``."""
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
        self.workers = args.workers
        self.split = args.split
        self.episode_timeout_s = args.episode_timeout_s
        self.max_infra_attempts = args.max_infra_attempts
        self._infra_errors: dict[str, Counter] = {}
        self.out_root = Path(args.out_root).resolve()
        self.episodes_file = self.out_root / episodes.EPISODES_JSONL
        self.driver_log = self.out_root / "driver.log"
        self.summary_file = self.out_root / "summary.txt"
        self.limit = args.limit
        self.sample_seed = SAMPLE_SEED if args.sample_seed is None else args.sample_seed
        self.allowlist_path = Path(args.episode_allowlist).resolve() if args.episode_allowlist else None
        self.arrivals_path = Path(args.arrivals).resolve() if args.arrivals else None
        # episode -> arrival_s, and the session clock's zero; a capacity session only.
        self._arrival: dict[str, float] = {}
        self._t0: float | None = None
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
        self._deadline = (
            time.monotonic() + self.max_wall_s if self.max_wall_s > 0 else None
        )

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with _log_lock:
            print(line, flush=True)
            with self.driver_log.open("a") as f:
                f.write(line + "\n")

    def _absorb(self, row: dict) -> None:
        """Count a ledger row: its episode is done, or it is one more infra
        attempt (``analysis.rows.is_infra_abort``: excluded from SR and
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
            ("!! nothing was scored" if not scored else f"!! {scored} attempt(s) were scored")
            + f"; the {len(bad)} infra attempts are excluded and re-queued, so a "
            "resume into this out-root re-flies them once the cause is fixed."
        )

    def _wall_spent(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def _next_claimed_episode(self) -> str | None:
        """The next episode not done, not in flight and not past its infra
        attempts; ``None`` once nothing is left, the wall budget is spent, or
        the run aborted (in-flight attempts finish)."""
        while not self._wall_spent() and self._abort is None:
            # The earliest report still to arrive, which a worker with nothing released waits for.
            due = float("inf")
            with self._dispatch_lock:
                n = len(self._selection)
                scanned = 0
                while scanned < n:
                    eid = self._selection[self._cursor % n]
                    self._cursor += 1
                    scanned += 1
                    if eid in self._in_flight or not self._left(eid):
                        continue
                    if self._session_s() < self._arrival.get(eid, 0.0):
                        due = min(due, self._arrival[eid])
                        continue
                    self._in_flight.add(eid)
                    return eid
            if due == float("inf"):
                return None
            time.sleep(max(0.0, min(due - self._session_s(), 1.0)))
        return None

    def _session_s(self) -> float:
        """Seconds on the capacity session's clock; 0 before it starts."""
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

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
        """True when the installed spec's reasoner answers its model list, at
        the URL discovery read it from (``endpoint_spec.models_url``), asked
        with the chat path's bearer (hosted endpoints auth ``/v1/models``)."""
        from maps.llm.endpoint_spec import models_url, reasoner_api_key, spec  # noqa: PLC0415

        key = reasoner_api_key()
        try:
            req = urllib.request.Request(models_url(spec().base_url))
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return 200 <= resp.status < 400
        except (OSError, ValueError):
            return False

    @staticmethod
    def _sam_answers() -> bool:
        """True when a SAM replica discovery health-checked answers
        ``/health`` (the client steps off a dead replica); True with none
        checked, as discovery skips an empty list."""
        from maps.llm.endpoint_spec import spec, spec_installed  # noqa: PLC0415

        urls = spec().sam_urls if spec_installed() else ()
        if not urls:
            return True
        for url in urls:
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=10.0) as resp:
                    if 200 <= resp.status < 400:
                        return True
            except (OSError, ValueError):
                continue
        return False

    def _wait_for_reasoner(self) -> bool:
        return self._wait_for("reasoner", self._reasoner_answers)

    def _wait_for_sam(self) -> bool:
        return self._wait_for("sam", self._sam_answers)

    def _wait_for(self, name: str, answers) -> bool:
        """Hold a launch until ``answers()``; False once the wall budget is
        spent or the run aborted while it waited."""
        if answers():
            return True
        self.log(f"{name} down; pausing launches until it answers")
        while not answers():
            if self._wall_spent() or self._abort is not None:
                return False
            left = self._deadline - time.monotonic() if self._deadline is not None else REASONER_WAIT_S
            time.sleep(max(0.0, min(REASONER_WAIT_S, left)))
        self.log(f"{name} back; resuming")
        return True

    def _wait(self, proc: subprocess.Popen) -> str:
        """Block until the attempt child exits (``exited``), hits the episode
        cap (``timeout``), or the reasoner or SAM fails two probes
        ``REASONER_PROBE_S`` apart while it flies (``reasoner_dead``,
        ``sam_dead``)."""
        deadline = time.monotonic() + self.episode_timeout_s
        down = {"reasoner_dead": 0, "sam_dead": 0}
        while (left := deadline - time.monotonic()) > 0:
            try:
                proc.wait(timeout=min(REASONER_PROBE_S, left))
                return "exited"
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                break
            for verdict, answers in (("reasoner_dead", self._reasoner_answers), ("sam_dead", self._sam_answers)):
                down[verdict] = 0 if answers() else down[verdict] + 1
                if down[verdict] >= 2:
                    return verdict
        return "timeout"

    def _run_attempt(self, worker_id: int, episode_id: str) -> dict:
        """One attempt in a fresh child: its ledger row, the child's
        ``episode.json`` plus ``worker_id``, ``workdir`` and the wall measured
        here. ``status`` is ``infra`` for a lost reasoner or SAM
        (``reasoner_down_midrun``, ``sam_down_midrun``), a child that a signal killed with no record (``killed_midrun``), that
        exited nonzero with none (``exited N with no record``) or exited clean
        with none (``no_prediction_unclassified``); a record's own ``infra``
        stands; otherwise ``timed_out`` at the cap, else ``ok``.
        """
        t0 = time.monotonic()
        worker_root = self.out_root / f"w{worker_id}"
        worker_root.mkdir(parents=True, exist_ok=True)
        env = _runner_env()
        # The mission's wall-clock backstop: an absolute epoch from the cap's
        # start, less the reserve for ending.
        env[episodes.DEADLINE_ENV] = repr(
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
        landed = episodes.read_episode_json(workdir) if workdir else None
        where = {"workdir": str(workdir.relative_to(self.out_root))} if workdir else {}
        crashed = verdict == "exited" and landed is None and proc.returncode != 0
        lost = {"reasoner_dead": "reasoner_down_midrun", "sam_dead": "sam_down_midrun"}.get(verdict)
        if lost:
            self.log(f"[w{worker_id}] {verdict.upper().replace('_', '-')} mid-episode {episode_id} after "
                     f"{elapsed:.0f}s; killed + re-queued (not scored)")
        elif verdict == "timeout":
            self.log(f"[w{worker_id}] TIMEOUT {episode_id} after {elapsed:.0f}s; killed")
        elif crashed:
            # Killed by a signal (a cgroup OOM is SIGKILL) or exited with an error before landing a record.
            self.log(f"[w{worker_id}] CRASHED mid-episode {episode_id} after "
                     f"{elapsed:.0f}s (exit {proc.returncode}); re-queued (not scored)")
        if lost or crashed:
            # Re-queued; a lost service's attempt is pending until it is back.
            return {"episode_id": episode_id, "worker_id": worker_id,
                    "status": episodes.STATUS_INFRA, "elapsed_s": round(elapsed, 1), **where,
                    "error": lost or ("killed_midrun" if proc.returncode < 0
                                      else f"exited {proc.returncode} with no record")}
        # The child's own verdict decides (dispatch classifies the episode's
        # exception chain); the driver adds only what it alone saw.
        row = {**(landed or {"episode_id": episode_id}), "worker_id": worker_id,
               "elapsed_s": round(elapsed, 1), **where}
        if landed is None and verdict == "exited":
            # A child that exits clean with no record failed, unrecognised.
            row |= {"status": episodes.STATUS_INFRA, "error": "no_prediction_unclassified"}
        elif row.get("status") != episodes.STATUS_INFRA:
            # A timeout is counted apart; its record is scored as recorded, and none is a miss.
            row["status"] = (episodes.STATUS_TIMED_OUT if verdict == "timeout"
                             else episodes.STATUS_OK)
        return row

    def run_one(self, worker_id: int, episode_id: str) -> None:
        """Fly one attempt and land its row."""
        self.log(f"[w{worker_id}] START {episode_id}")
        claimed = self._session_s()
        row = self._run_attempt(worker_id, episode_id)
        if episode_id in self._arrival:
            row |= {"arrival_s": self._arrival[episode_id], "claimed_s": round(claimed, 1)}
        m = row.get("metrics") or {}
        summary = ("" if _rows.is_infra_abort(row)
                   else f" NE={m.get('NE')} SR={m.get('SR')}" if m else " MISSING")
        self.log(
            f"[w{worker_id}] END {episode_id} status={row['status']} "
            f"elapsed={row['elapsed_s']:.0f}s{summary}"
        )
        self._append_result(row)
        self._write_summary()

    def _write_summary(self) -> None:
        """``summary.txt``: the driver's infra and abort lines above
        ``analysis.run_report.report`` over the root, once a row has landed."""
        from benchmarks.citynav.analysis import run_report  # noqa: PLC0415

        infra = Counter(_infra_cause(r)
                        for r in _rows.iter_json_rows(self.episodes_file)
                        if _rows.is_infra_abort(r))
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
                run_report.report(self.out_root, allow_partial=True)
            except SystemExit as exc:  # no scored row yet
                print(exc)
        text = "\n".join(x for x in lines if x) + "\n" + buf.getvalue().lstrip("\n")
        self.summary_file.write_text(text)

    def worker_loop(self, worker_id: int) -> None:
        while (eid := self._next_claimed_episode()) is not None:
            try:
                if self._wait_for_reasoner() and self._wait_for_sam():
                    self.run_one(worker_id, eid)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[w{worker_id}] FAIL {eid}: {type(exc).__name__}: {exc}")
                try:
                    self._append_result({
                        "episode_id": eid,
                        "worker_id": worker_id,
                        "status": episodes.STATUS_INFRA,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
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
        """The selection, in flying order; a refusal raises ``SystemExit``
        with its FATAL line. An allowlist's count line is kept in
        ``_selection_note``."""
        if self.arrivals_path is not None:
            try:
                reports = json.loads(self.arrivals_path.read_text())["reports"]
                self._arrival = {r["episode_id"]: float(r["arrival_s"]) for r in reports}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"FATAL: --arrivals read failed: {self.arrivals_path}: {exc!r}") from exc
            unknown = set(self._arrival) - set(_enumerate_split(self.split))
            if unknown:
                raise SystemExit(f"FATAL: --arrivals names {len(unknown)} episodes not in the {self.split} "
                                 f"split, e.g. {sorted(unknown)[0]}")
            return sorted(self._arrival, key=self._arrival.get)
        episodes = _enumerate_split(self.split)
        if self.allowlist_path is not None:
            try:
                lines = self.allowlist_path.read_text().splitlines()
            except OSError as exc:
                raise SystemExit(f"FATAL: --episode-allowlist read failed: {self.allowlist_path}: {exc}") from exc
            allowed = {ln.strip() for ln in lines if ln.strip()}
            if not allowed:
                raise SystemExit(f"FATAL: --episode-allowlist names no episode: {self.allowlist_path}")
            before = len(episodes)
            episodes = [e for e in episodes if e in allowed]
            unknown = sorted(allowed - set(episodes))
            if unknown:
                raise SystemExit(f"FATAL: --episode-allowlist names {len(unknown)} ids not in the "
                                 f"{self.split} split, e.g. {unknown[0]}")
            self._selection_note = (
                f"--episode-allowlist {self.allowlist_path}: "
                f"{len(allowed)} ids requested -> {len(episodes)} match "
                f"the {self.split} split (from {before} total)"
            )
        random.Random(self.sample_seed).shuffle(episodes)
        if self.limit > 0:
            episodes = episodes[: self.limit]
        return episodes

    def _selected_by(self) -> tuple[str, object]:
        """What picks the selection: a session's stream, else the shuffle seed."""
        if self.arrivals_path is not None:
            return "arrivals", self.arrivals_path
        return "sample_seed", self.sample_seed

    def dry_run(self) -> int:
        """Selection preview. Launches nothing and writes nothing."""
        episodes = self._selection
        if self._selection_note:
            print(self._selection_note)
        self._absorb_ledger()
        done = self._done_cache
        pending = [e for e in episodes if self._left(e)]
        key, value = self._selected_by()
        print(f"{key}: {value}   split: {self.split}   limit: {self.limit}")
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
        finally:
            lock.close()

    def _run(self) -> int:
        # A session's clock cannot resume: its reports would arrive against a
        # fresh zero. Refused before the manifest records a resume.
        if (self.arrivals_path is not None and self.episodes_file.exists()
                and self.episodes_file.stat().st_size):
            self.log(f"FATAL: {self.out_root} already holds a ledger; a capacity session flies "
                     f"into a fresh --out-root")
            return 2
        episodes = self._selection
        if self._selection_note:
            self.log(self._selection_note)
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
                limit=self.limit,
                allowlist_path=self.allowlist_path,
                episodes=episodes,
                arrivals=(hashlib.sha256(self.arrivals_path.read_bytes()).hexdigest()
                          if self.arrivals_path is not None else None),
            )
            _rm.emit_or_verify(self.out_root, fresh_manifest, log=self.log)
        except _rm.ManifestIdentityError as exc:
            self.log(f"FATAL: {exc}")
            return 2
        # A root with a manifest is a run root, flown or not: its ledger exists.
        self.episodes_file.touch()
        self._absorb_ledger()
        if self.arrivals_path is not None:
            shutil.copyfile(self.arrivals_path, self.out_root / "arrivals.json")
        pending = [e for e in episodes if self._left(e)]
        key, value = self._selected_by()
        self.log(
            f"split={self.split} {key}={value} selection={len(episodes)} "
            f"done={len(self._done_cache & set(episodes))} pending={len(pending)} "
            f"workers={self.workers}"
        )
        # Workers claim until nothing is left, the wall budget is spent, or the
        # run aborts; an infra attempt is re-claimed by whichever worker is free.
        # Daemon workers end with the driver (their children die with it), so
        # none flies on once an interrupt has released the root.
        if pending and not self._wall_spent():
            threads = []
            self._t0 = time.monotonic()
            for wid in range(self.workers):
                t = threading.Thread(target=self.worker_loop, args=(wid,), daemon=True)
                t.start()
                threads.append(t)
                # A session's arrivals stagger its starts, and a held worker would delay a report.
                if self.arrivals_path is None:
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m benchmarks.{paths.BENCHMARK}.bench_fullset",
        description=f"Continuous full-split {paths.BENCHMARK} benchmark at fixed concurrency.",
    )
    p.add_argument("--split", default=paths.DEFAULT_SPLIT, help=f"(default {paths.DEFAULT_SPLIT})")
    p.add_argument(
        "--workers",
        type=_at_least_one,
        default=6,
        help="concurrency (default %(default)s)",
    )
    p.add_argument(
        "--out-root",
        default=f"output/{paths.BENCHMARK}_fullset",
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
        "--limit",
        type=int,
        default=0,
        help="episodes kept after the seeded shuffle (default 0 = all); with "
        "one --sample-seed, the same N episodes on every run",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
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
        "--arrivals",
        default=None,
        help="a capacity session's report stream (benchmarks.capacity.streams): its "
        "episodes, each released at its arrival time; refused with --episode-allowlist, "
        "--limit or --sample-seed",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print seed, selected episodes and done/pending; launch nothing",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Discovery logs at INFO (the installed spec); without a config it reaches no console.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.arrivals and (args.episode_allowlist or args.limit
                          or args.sample_seed is not None):
        parser.error("--arrivals selects the session's episodes; drop --episode-allowlist, --limit "
                     "and --sample-seed")
    # The config is refused here, before an out-root exists or a serve is asked.
    from benchmarks.citynav.run_benchmark.config import resolve_config

    try:
        resolved = resolve_config(CONFIG)[0]
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    driver = Driver(args)
    # The selection too, so a refusal makes no out-root and asks no serve.
    try:
        driver._selection = driver._select_episodes()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.dry_run:
        return driver.dry_run()
    # Endpoint discovery once, so an unreachable or wrong serve fails here,
    # before any worker spawns.
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
    settle_effort()
    if resolved.get("policy"):
        # Every baseline arm makes dial-off calls: an arm whose model has none
        # is refused here, once, rather than scored a miss per episode.
        seam = importlib.import_module("baselines.seam")  # by name: no static import of baselines
        try:
            seam.dial_off(spec.model_id)
        except seam.NoDialOff as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
