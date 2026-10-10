"""A run that cannot produce results stops instead of grinding to the end.

``bench_fullset`` already classified the shape: a runner that exits clean
with no record, no timeout and no recognized infra token is an
unrecognized instant failure (``no_prediction_unclassified``). The only
response was a warning line in the END-OF-RUN summary, so a tree deployed
without its tile bake failed 282 of 282 episodes across ~15 min of an
exclusive 8xL40S node and reported it an hour late, to nobody.

The gate here is a trailing window of completions. What it must NOT do is
misfire: a hard episode timeout with no record is the ALGORITHM failing
to answer within its budget — a legitimate CityNav miss — and the
classifier is explicit that reclassifying it as infra would inflate SR
and break benchmark compliance. Scattered OOM retries on a healthy run
are normal too.

Exercised, not scanned: real rows go through the real ``_append_result``,
and the assertions are about ``_abort``, the claim path and the
resume-visible contents of episodes.jsonl.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import pytest

from benchmarks.citynav import bench_fullset as bf


def _driver(tmp_path: Path) -> bf.Driver:
    """A Driver with only the fields the result path touches. __init__ is
    bypassed on purpose — it resolves a split, a config and a dispatcher,
    none of which this gate reads."""
    d = bf.Driver.__new__(bf.Driver)
    d.out_root = tmp_path
    d.episodes_file = tmp_path / "episodes.jsonl"
    d.driver_log = tmp_path / "driver.log"
    d._session_done = 0
    d._dispatch_lock = threading.Lock()
    d._done_cache = set()
    d._in_flight = set()
    d._recent = deque(maxlen=bf.ABORT_WINDOW)
    d._abort = None
    d._deadline = None
    d._selection = [f"ep{i}" for i in range(500)]
    d._cursor = 0
    # Only run() and _write_summary read the rest.
    d.summary_file = tmp_path / "summary.txt"
    d.split = "train_seen"
    d.workers = 2
    d.episode_timeout_s = 3600
    d.max_wall_s = 0
    d.sample_seed = 1234
    d.limit = 0
    d.allowlist_path = None
    d.max_infra_attempts = 3
    d._infra_errors = {}
    d._split_size = 0
    d._t_start = 0.0
    return d


def _infra_row(i: int, workdir: str | None = None, cause: str = "no_prediction_unclassified"):
    row = {"episode_id": f"ep{i}", "worker_id": 0, "status": "infra", "error": cause}
    if workdir is not None:
        row["workdir"] = workdir
    return row


def _scored_row(i: int, sr: float = 0.0):
    return {
        "episode_id": f"ep{i}",
        "worker_id": 0,
        "status": "ok",
        "metrics": {"SR": sr, "NE": 42.0},
    }


def _timeout_row(i: int):
    """The load-bearing non-infra shape: the runner hit the per-episode
    wall cap and committed nothing. No metrics, and status ``timed_out``
    — a miss, not a casualty."""
    return {"episode_id": f"ep{i}", "worker_id": 0, "status": "timed_out"}


def test_the_window_floor_is_a_real_number() -> None:
    """Vacuity floor for every threshold assertion below: a window of 1 or
    a fraction of 0 would make the gate fire on the first bad row and pass
    every test here for the wrong reason."""
    assert bf.ABORT_WINDOW >= 8, bf.ABORT_WINDOW
    assert 0.75 <= bf.ABORT_FRACTION <= 1.0, bf.ABORT_FRACTION


def test_mass_instant_failure_stops_the_run(tmp_path: Path) -> None:
    wd = "w0/maps_20260805_101500_citynav_ep0"
    log = tmp_path / wd / "run.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "some preamble\n"
        "FileNotFoundError: Missing /home/x/MAPS-arm/tiles/"
        "citynav_2m_osmv3/birmingham_block_1_meta.json.\n"
        "Run benchmarks/citynav/data_prep/bake_tiles.sh for this block first.\n"
    )
    d = _driver(tmp_path)
    for i in range(bf.ABORT_WINDOW - 1):
        d._append_result(_infra_row(i, wd))
        assert d._abort is None, f"fired after only {i + 1} rows"
    d._append_result(_infra_row(bf.ABORT_WINDOW - 1, wd))
    assert d._abort is not None, "a full window of instant failures did not stop"

    # Loud about WHY, naming the dominant cause and a representative error.
    assert "no_prediction_unclassified" in d._abort
    text = d.driver_log.read_text()
    assert "ABORTING RUN" in text
    assert str(log) in text, "the abort did not name the attempt's run.log to read"
    assert "birmingham_block_1_meta.json" in text, (
        "the abort did not carry the actual error; the next actor still has "
        "to go spelunking in a worker log"
    )

    # And it stops claiming: the workers drain instead of taking more work.
    assert d._next_claimed_episode(0) is None


def test_a_healthy_run_never_aborts(tmp_path: Path) -> None:
    """A slow start and scattered infra retries are normal. Only a run
    whose window is almost entirely casualties is conclusive."""
    d = _driver(tmp_path)
    for i in range(200):
        d._append_result(_infra_row(i) if i % 8 == 0 else _scored_row(i))
    assert d._session_done == 200, "the fixture stopped appending"
    assert d._abort is None, f"a healthy run aborted: {d._abort}"
    assert d._next_claimed_episode(0) is not None


def test_a_burst_shorter_than_the_window_never_aborts(tmp_path: Path) -> None:
    """Absolute counts before rates: two failures out of two is not
    evidence, and neither is a window that has not filled."""
    d = _driver(tmp_path)
    for i in range(bf.ABORT_WINDOW - 1):
        d._append_result(_infra_row(i))
    assert d._abort is None
    assert d._next_claimed_episode(0) is not None


def test_episode_timeouts_are_never_mistaken_for_infra(tmp_path: Path) -> None:
    """The distinction the existing classifier is explicit about: a
    timeout with no record is the ALGORITHM failing, a legitimate miss. A
    run of nothing but timeouts is a bad model, not a bad node, and must
    not be aborted as infra."""
    d = _driver(tmp_path)
    for i in range(bf.ABORT_WINDOW * 3):
        d._append_result(_timeout_row(i))
    assert d._session_done == bf.ABORT_WINDOW * 3
    assert not any(bf._infra_casualty(r) for r in [_timeout_row(0)])
    assert d._abort is None, f"an all-timeout run was aborted as infra: {d._abort}"


def test_a_mid_run_infra_death_still_stops(tmp_path: Path) -> None:
    """A session-wide rate cannot fire after a healthy first hour: 100 good
    rows would need ~900 bad ones to cross 90%. The window is trailing so
    the same failure costs the same to detect whenever it starts."""
    d = _driver(tmp_path)
    for i in range(150):
        d._append_result(_scored_row(i))
    assert d._abort is None
    for i in range(150, 150 + bf.ABORT_WINDOW):
        d._append_result(_infra_row(i))
    assert d._abort is not None, "a mid-run infra death was never detected"


def test_the_abort_leaves_every_episode_re_queued(tmp_path: Path) -> None:
    """Resume safety: the gate writes nothing and scores nothing. Every
    row it counted was already excluded from the done-set, so a resume
    into this out-root re-flies exactly those episodes."""
    d = _driver(tmp_path)
    for i in range(bf.ABORT_WINDOW):
        d._append_result(_infra_row(i))
    assert d._abort is not None
    written = [json.loads(ln) for ln in d.episodes_file.read_text().splitlines()]
    assert len(written) == bf.ABORT_WINDOW, "the abort dropped or added rows"
    assert all(bf._done_qualify(r) is None for r in written), (
        "an aborted episode counts as done, so a resume would skip it"
    )
    assert d._done_cache == set(), "an excluded row reached the done cache"


def test_the_gate_counts_exactly_what_the_done_set_excludes() -> None:
    """One predicate, so the two cannot drift: a row the gate counts must
    be a row resume re-flies, and vice versa. The rows are the shapes the
    producers write: ``status`` is the one verdict, and a mission the
    serve failed out from under lands as ``infra`` with its cause."""
    def _ended_row(i: int, ended_by: str) -> dict:
        serve_failed = ended_by == "transient_failures"
        return {
            "episode_id": f"ep{i}",
            "status": "infra" if serve_failed else "ok",
            "error": f"mission ended_by={ended_by}" if serve_failed else None,
            "metrics": {"SR": 0, "mission": {"ended_by": ended_by}},
        }

    rows = [
        _infra_row(0),
        {"episode_id": "ep1", "status": "infra", "error": "vllm_down_midrun"},
        _scored_row(2),
        _timeout_row(3),
        _ended_row(4, "transient_failures"),
        _ended_row(5, "turn_cap"),
        _ended_row(6, "compaction_infeasible"),
    ]
    counted = [bf._infra_casualty(r) for r in rows]
    assert counted == [True, True, False, False, True, False, False], counted
    for r, c in zip(rows, counted):
        assert (bf._done_qualify(r) is None) == c, r


def test_the_run_exits_nonzero_and_the_summary_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract an autoresearch loop reads: an aborted arm is a FAILED
    run, not a suspiciously fast successful one, and the artifact a human
    opens carries the reason."""
    from benchmarks.citynav import run_manifest as rm

    monkeypatch.setattr(bf, "_missing_block_assets", lambda eps: [])
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: {})
    monkeypatch.setattr(rm, "emit_or_verify", lambda *a, **k: None)
    d = _driver(tmp_path)
    monkeypatch.setattr(d, "_select_episodes", lambda: [f"ep{i}" for i in range(60)])
    # Wall budget as the backstop, so a driver that does NOT abort ends
    # this test with a wrong exit code instead of spinning passes forever
    # (infra rows never qualify as done, so the pending list never shrinks).
    d.max_wall_s = 15
    d._deadline = time.monotonic() + 15

    seq = iter(range(60))

    def one_window(worker_id: int) -> None:
        for i in seq:
            if d._next_claimed_episode(worker_id) is None:
                return
            d._append_result(_infra_row(i))

    monkeypatch.setattr(d, "worker_loop", one_window)
    assert d.run() == 3, "an aborted run reported success"
    assert d._abort is not None
    summary = d.summary_file.read_text()
    assert "ABORTED" in summary, summary
    assert "no_prediction_unclassified" in summary, summary
    # It stopped EARLY: the selection was 60 and the workers drained long
    # before it. Without this the test would pass on a run that finished.
    n_rows = len(d.episodes_file.read_text().splitlines())
    assert n_rows < 60, f"the run did not stop early ({n_rows} rows)"


def test_a_spawn_failure_reads_the_childs_own_log(tmp_path: Path) -> None:
    """The warm runner's stderr is the only record of why it died.

    ``warm runner stdout closed before ready`` is raised on the DRIVER's
    side of the pipe: it says the child exited, never why. One run died
    after 22 h 55 m reporting exactly that and
    ``representative worker log: (none recorded)``, while every failing
    child had written ``endpoint discovery failed: service 'sam'`` into its
    server.log -- the perception serve had gone away. An
    episode that never started has no workdir and no run.log, so the abort
    banner reads ``w<id>/server.log`` for a row without one.
    """
    d = _driver(tmp_path)
    server_log = tmp_path / "w0" / "server.log"
    server_log.parent.mkdir()
    server_log.write_text(
        "07:12:05 INFO  EVAL-PROVENANCE commit=3443a20 dirty=False\n"
        "07:12:05 ERROR endpoint discovery failed: service 'sam' failed its "
        "/health probe: endpoint unreachable: GET http://compute-6-8:8772/health\n"
    )

    class _DeadRunner:
        server_log = None

        def alive(self) -> bool:
            return False

        def clear_lockfile(self) -> None:
            pass

        def spawn(self) -> None:
            raise RuntimeError("warm runner stdout closed before ready")

    runner = _DeadRunner()
    runner.server_log = server_log
    row = d._run_attempt(0, "ep0", runner)

    assert row["status"] == "infra" and "workdir" not in row, row
    assert bf._infra_casualty(row), "a spawn failure must stay an infra casualty"

    for i in range(bf.ABORT_WINDOW):
        d._append_result({**row, "episode_id": f"ep{i}"})
    assert d._abort is not None
    text = d.driver_log.read_text()
    assert str(server_log) in text, "the abort did not name the child's log"
    assert "endpoint discovery failed" in text, (
        "the abort still cannot say why the child died; the next actor has to "
        "go spelunking in a per-worker log to find the dead perception plane"
    )


def test_a_transient_failure_window_names_its_cause(tmp_path: Path) -> None:
    """A mission the serve failed out from under lands as ``infra`` with
    ``error: mission ended_by=transient_failures``, written by the producer.
    Reported as the bare word ``infra``, the dominant cause names
    nothing."""
    d = _driver(tmp_path)
    for i in range(bf.ABORT_WINDOW):
        d._append_result({
            "episode_id": f"ep{i}", "status": "infra",
            "error": "mission ended_by=transient_failures",
            "metrics": {"SR": 0, "mission": {"ended_by": "transient_failures"}},
        })
    assert d._abort is not None, "a window of transient_failures did not stop"
    assert "transient_failures" in d._abort, d._abort
    assert "'infra'" not in d._abort, (
        f"the dominant cause is a label with no content: {d._abort}"
    )


@pytest.mark.parametrize("missing", [None, "w0/nonexistent_attempt"])
def test_a_missing_worker_log_does_not_break_the_abort(
    tmp_path: Path, missing: str | None
) -> None:
    """The excerpt is a convenience; failing to read it must never turn the
    stop into a crash (or, worse, into no stop at all)."""
    d = _driver(tmp_path)
    for i in range(bf.ABORT_WINDOW):
        d._append_result(_infra_row(i, missing))
    assert d._abort is not None
    assert "ABORTING RUN" in d.driver_log.read_text()


def test_the_abort_counts_attempts_and_says_nothing_was_scored_only_when_true(tmp_path: Path) -> None:
    d = _driver(tmp_path)
    d._append_result(_scored_row(100, sr=1.0))
    for i in range(bf.ABORT_WINDOW):
        d._append_result(_infra_row(i))
    assert d._abort is not None and "attempts" in d._abort and "episodes" not in d._abort
    assert "nothing was scored" not in d.driver_log.read_text()

    (tmp_path / "fresh").mkdir()
    fresh = _driver(tmp_path / "fresh")
    for i in range(bf.ABORT_WINDOW):
        fresh._append_result(_infra_row(i))
    assert "nothing was scored" in fresh.driver_log.read_text()


def test_a_landed_crash_row_is_scored_not_infra(tmp_path: Path, monkeypatch) -> None:
    """The runner's own verdict decides: a method exception lands ``ok``
    with no metrics, and the driver keeps it ``ok`` (a scored miss) instead
    of reading its missing metrics as an unrecognized infra death."""
    from benchmarks.citynav import episodes

    d = _driver(tmp_path)
    wd = tmp_path / "w0" / "maps_20260805_101500_citynav_ep0"
    episodes.write_episode_json(wd, {"episode_id": "ep0", "status": "ok",
                                     "error": "RuntimeError: boom", "traceback": "..."})

    class _Runner:
        def alive(self):
            return True

        def run_episode(self, eid, deadline_epoch=None):
            return {"workdir": str(wd), "map": "m"}, 1

        def should_recycle(self, map_name):
            return False

    monkeypatch.setattr(d, "_wait_for_exit", lambda runner, seq, t: "exited")
    row = d._run_attempt(0, "ep0", _Runner())
    assert row["status"] == "ok" and row["error"] == "RuntimeError: boom"
    assert not bf._infra_casualty(row)


def test_an_episode_infra_at_every_capped_attempt_is_not_claimed_again(tmp_path: Path) -> None:
    d = _driver(tmp_path)
    d._selection = ["ep0"]
    for _ in range(d.max_infra_attempts):
        assert d._next_claimed_episode(0) == "ep0"
        d._append_result(_infra_row(0, cause="SamServerOutage"))
        d._in_flight.discard("ep0")
    assert d._next_claimed_episode(0) is None, "a capped infra episode was re-flown"


def test_persistent_infra_is_reported_with_its_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.citynav import episodes
    from benchmarks.citynav import run_manifest as rm

    monkeypatch.setattr(bf, "_missing_block_assets", lambda eps: [])
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: {})
    monkeypatch.setattr(rm, "emit_or_verify", lambda *a, **k: None)
    for cause in ("A", "A", "B"):
        episodes.append_row(tmp_path, _infra_row(0, cause=cause))
    d = _driver(tmp_path)
    monkeypatch.setattr(d, "_select_episodes", lambda: ["ep0"])
    flown: list[int] = []
    monkeypatch.setattr(d, "worker_loop", flown.append)
    d.max_wall_s, d._deadline = 5, time.monotonic() + 5  # bounds a driver that re-flies
    assert d.run() == 0
    assert not flown, "a capped infra episode was handed to a worker"
    for text in (d.summary_file.read_text(), d.driver_log.read_text()):
        assert "persistent infra: 1 episode(s)" in text and "A=2, B=1" in text, text


def test_a_second_run_into_a_held_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two drivers into one root fly the same episodes twice. The first
    holds the root for its lifetime; the second refuses, naming it."""
    import os
    import subprocess
    import sys

    from benchmarks.citynav import run_manifest as rm

    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time\n"
         "from pathlib import Path\n"
         "from benchmarks.citynav import episodes\n"
         "lock = episodes.hold_root(Path(sys.argv[1]))\n"
         "print('held', flush=True)\n"
         "time.sleep(60)\n", str(tmp_path)],
        cwd=Path(bf.__file__).resolve().parents[2], stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(rm, "build_manifest", lambda **kw: pytest.fail("flew"))
        d = _driver(tmp_path)
        monkeypatch.setattr(d, "_select_episodes", lambda: ["ep0"])
        assert d.run() == 2
        assert f"pid {holder.pid}" in d.driver_log.read_text()
    finally:
        holder.kill()
        holder.wait()
    assert os.path.exists(tmp_path / ".bench.lock")


def test_a_ledger_the_worker_cannot_write_stops_the_run_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker whose row cannot land would die with its thread and leave
    the run one worker short, saying nothing in driver.log."""
    from benchmarks.citynav import episodes

    d = _driver(tmp_path)
    d._selection = ["ep0"]

    def full(root, row):
        raise OSError(28, "No space left on device")

    def boom(worker_id, eid, runner):
        raise RuntimeError("the attempt failed")

    monkeypatch.setattr(episodes, "append_row", full)
    monkeypatch.setattr(d, "_wait_for_vllm", lambda: None)
    monkeypatch.setattr(d, "run_one", boom)
    d.worker_loop(0)
    assert d._abort and "No space left on device" in d._abort
    assert "No space left on device" in d.driver_log.read_text()
    assert d._next_claimed_episode(1) is None


def test_the_abort_counts_a_scored_miss_as_scored(tmp_path: Path) -> None:
    d = _driver(tmp_path)
    d._append_result({"episode_id": "ep100", "status": "ok", "error": "RuntimeError: x"})
    for i in range(bf.ABORT_WINDOW):
        d._append_result(_infra_row(i))
    text = d.driver_log.read_text()
    assert "1 attempt(s) were scored" in text and "cannot produce results" not in text


def test_a_selection_whose_block_assets_are_missing_fails_before_flying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block with no bake or no raster fails every episode on it; the run
    names the missing files at startup instead of spending a window of
    infra attempts to find out."""
    from benchmarks.citynav import paths
    from benchmarks.citynav import run_manifest as rm

    tiles, rgbd = tmp_path / "tiles", tmp_path / "rgbd"
    tiles.mkdir()
    rgbd.mkdir()
    for suffix in paths.BAKE_OUTPUT_SUFFIXES:
        (tiles / f"blk{suffix}").write_text("{}")
    (rgbd / "blk.tif").write_bytes(b"")
    monkeypatch.setattr(paths, "TILES_DIR", tiles)
    monkeypatch.setattr(paths, "CITYNAV_NATIVE_RGBD_DIR", rgbd)
    manifests: list[dict] = []
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: manifests.append(kw) or {})
    monkeypatch.setattr(rm, "emit_or_verify", lambda *a, **k: None)
    root = tmp_path / "run"
    root.mkdir()
    d = _driver(root)
    monkeypatch.setattr(d, "_select_episodes", lambda: ["blk::1#0"])
    monkeypatch.setattr(d, "worker_loop", lambda w: None)

    d.max_wall_s, d._deadline = 1, time.monotonic() + 1  # bounds a driver that flies on
    assert d.run() == 2 and not manifests
    assert str(rgbd / "blk_rgb.tif") in d.driver_log.read_text()

    (rgbd / "blk.png").write_bytes(b"")  # the CityNav raster pair
    d.max_wall_s, d._deadline = 1, time.monotonic() + 1
    d.run()
    assert manifests, "a complete raster pair was refused"
