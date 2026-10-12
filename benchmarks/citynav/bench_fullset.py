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

from benchmarks.citynav import episodes, paths
from benchmarks.citynav.analysis import rows as _rows

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / (os.environ.get("CITYNAV_BENCH_CONFIG", "").strip() or "configs/citynav.yaml")

# The runner each attempt child runs.
RUNNER_MODULE = "benchmarks.citynav.run_benchmark"
# Share of the episode cap held back from flying, so the model still has
# turns to reply and commit after the mission's wall-clock backstop fires.
_DEADLINE_RESERVE_FRAC = 0.1
# How often the driver asks the reasoner's ``/models`` while an attempt
# flies; two failures in a row kill the attempt as ``vllm_down_midrun``.
VLLM_PROBE_S = 60.0
# Pause between worker thread starts, so the first attempts do not all
# cold-start at once.
WORKER_STAGGER_S = 2.0
# The run aborts when ABORT_FRACTION of the last ABORT_WINDOW completions are
# infra casualties (a missing tile bake fails every episode in seconds). A
# trailing window still fires after a healthy first hour; it must fill first.
ABORT_WINDOW = 12
ABORT_FRACTION = 0.9


_log_lock = threading.Lock()


def _enumerate_split(split: str) -> list[str]:
    from benchmarks.citynav.adapter import episode_ids, load_split

    if split not in paths.SPLIT_FILES:
        raise SystemExit(f"unknown split {split!r}; known: {sorted(paths.SPLIT_FILES)}")
    return episode_ids(enumerate(load_split(paths.split_path(split))))


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
    # cli.PARENT_PID_ENV, named here so the driver stays off the engine's imports.
    env["CITYNAV_BENCH_PARENT_PID"] = str(os.getpid())
    return env


def _infra_cause(row: dict) -> str:
    """What killed this infra attempt: its ``error``."""
    return str(row.get("error") or "").strip() or "infra"


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
        self.workers = max(1, int(args.workers))
        self.split = args.split
        # The split and seed stamp every row's provenance (dispatch).
        os.environ["CITYNAV_BENCH_SPLIT"] = str(self.split)
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
        if args.episode_allowlist:
            ap = Path(args.episode_allowlist)
            if not ap.is_absolute():
                ap = (REPO / ap).resolve()
            self.allowlist_path = ap
        # Guarded by _dispatch_lock. _recent is the mass-infra window as
        # (cause or None, log path).
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

    # -- logging / results -------------------------------------------------

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
            return f"    (no readable log at {log_path or '?'}: {exc})"
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
                     if not _rows.is_infra_abort(r))
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
        if self._wall_spent() or self._abort is not None:
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

    def _persistent_infra(self) -> str:
        """The line naming the selected episodes left out after
        ``max_infra_attempts`` infra attempts, errors counted, or ""."""
        with self._dispatch_lock:
            stuck = [e for e in self._selection if e not in self._done_cache
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
        """True when the reasoner endpoint answers its model list, at the URL
        discovery read it from (``endpoint_spec.models_url``), asked with the
        chat path's bearer (hosted endpoints auth ``/v1/models``)."""
        from maps.llm.endpoint_spec import models_url  # noqa: PLC0415

        url = os.environ.get("SEARCH_REASONER_URL", "").strip().rstrip("/")
        req = urllib.request.Request(models_url(url))
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
            time.sleep(120.0)
        self.log("vLLM back; resuming")

    # -- one launch cycle --------------------------------------------------

    def _wait(self, proc: subprocess.Popen) -> str:
        """Block until the attempt child exits (``exited``), hits the episode
        cap (``timeout``), or the reasoner fails two ``/models`` probes
        ``VLLM_PROBE_S`` apart while it flies (``vllm_dead``)."""
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

        The child's ``episode.json`` plus ``worker_id``, ``workdir`` and the
        wall measured here, with ``status`` overridden on the driver's paths:
        ``timed_out`` for the cap; ``infra`` for a lost reasoner
        (``vllm_down_midrun``), a child that died with no record
        (``killed_midrun``), and one that exited clean with none
        (``no_prediction_unclassified``).
        """
        t0 = time.monotonic()
        worker_root = self.out_root / f"w{worker_id}"
        worker_root.mkdir(parents=True, exist_ok=True)
        env = _runner_env()
        # The mission's wall-clock backstop (mission_engine.MISSION_DEADLINE_ENV):
        # an absolute epoch from the cap's start, less the reserve for ending.
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
            # Re-queued: the attempt is pending until the reasoner is back.
            self.log(f"[w{worker_id}] VLLM-DEAD mid-episode {episode_id} after "
                     f"{elapsed:.0f}s; killed + re-queued (not scored)")
            return {**killed, "error": "vllm_down_midrun"}
        if verdict == "timeout":
            self.log(f"[w{worker_id}] TIMEOUT {episode_id} after {elapsed:.0f}s; killed")
        elif landed is None and proc.returncode != 0:
            # SIGKILLed (a cgroup OOM) or crashed before landing a record.
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
            # A child that exits clean with no record failed, unrecognised.
            row["status"] = episodes.STATUS_INFRA
            row["error"] = "no_prediction_unclassified"
        elif row.get("status") == episodes.STATUS_INFRA:
            pass
        elif verdict == "timeout":
            # The cap is an algorithm miss (in the denominator), record or not.
            row["status"] = episodes.STATUS_TIMED_OUT
        else:
            row["status"] = episodes.STATUS_OK
        return row

    def run_one(self, worker_id: int, episode_id: str) -> None:
        """Fly one attempt and land its row."""
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
        """``summary.txt``: the driver's infra and abort lines above
        ``analysis.run_report.report`` over the root."""
        from benchmarks.citynav.analysis import run_report  # noqa: PLC0415

        infra = Counter(_infra_cause(r)
                        for r in _rows.iter_json_rows(self.episodes_file)
                        if r.get("status") == episodes.STATUS_INFRA)
        lines = [
            f"split: {self.split}   workers: {self.workers}   "
            f"episode_cap: {self.episode_timeout_s}s",
            f"infra (re-queued, not scored): {infra.total()}"
            + ("  by cause: " + ", ".join(f"{c}:{k}" for c, k in infra.most_common())
               if infra else ""),
            self._persistent_infra(),
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
        while (eid := self._next_claimed_episode()) is not None:
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
                    # No worker can land a result: the run stops.
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
        self._absorb_ledger()
        done = self._done_cache
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
                limit=self.limit,
                workers=self.workers,
                allowlist_path=self.allowlist_path,
                episodes=episodes,
            )
            _rm.emit_or_verify(self.out_root, fresh_manifest, log=self.log)
        except _rm.ManifestIdentityError as exc:
            self.log(f"FATAL: {exc}")
            return 2
        self._absorb_ledger()
        pending = [e for e in episodes if e not in self._done_cache
                   and self._infra_attempts(e) < self.max_infra_attempts]
        self.log(
            f"split={self.split} sample_seed={self.sample_seed} selection={len(episodes)} "
            f"done={len(self._done_cache & set(episodes))} pending={len(pending)} "
            f"workers={self.workers}"
        )
        # Workers claim until nothing is left, the wall budget is spent, or the
        # run aborts; an infra attempt is re-claimed by whichever worker is free.
        if pending and not self._wall_spent():
            threads = []
            for wid in range(self.workers):
                t = threading.Thread(target=self.worker_loop, args=(wid,), daemon=False)
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.citynav.bench_fullset",
        description="Continuous full-test-set CityNav benchmark at fixed concurrency.",
    )
    p.add_argument("--split", default=paths.DEFAULT_SPLIT, help=f"(default {paths.DEFAULT_SPLIT})")
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="concurrency (default %(default)s; empirical optimum)",
    )
    p.add_argument(
        "--out-root",
        default="output/citynav_fullset",
        help="results + per-worker run-roots (default output/citynav_fullset)",
    )
    p.add_argument(
        "--episode-timeout-s",
        type=int,
        default=4 * 3600,
        help="per-episode wall cap (default %(default)ss = 4h)",
    )
    p.add_argument(
        "--max-wall-s",
        type=int,
        default=0,
        help="continuous-run wall budget (default 0 = run until the "
        "selection / --limit run-count budget is exhausted). >0: stop "
        "claiming new episodes after this many seconds; in-flight "
        "episodes finish and results are retained.",
    )
    p.add_argument(
        "--max-infra-attempts",
        type=int,
        default=3,
        help="infra attempts an episode gets before the run leaves it out and "
        "reports it as persistent infra (default %(default)s)",
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
        "seeded shuffle + --limit still apply on top of the allowlist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print seed, selected episodes and done/pending; launch nothing",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from benchmarks import stamp_repo_provenance

    # Without a logging config the INFO provenance block reaches no log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    stamp_repo_provenance()
    args = _build_parser().parse_args(argv)
    driver = Driver(args)
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
    settle_effort(spec.model_id)
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
