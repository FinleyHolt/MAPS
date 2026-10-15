"""The bench driver: one child per attempt, its row, the claims, the stops.

The children are ``fake_runner``, whose behaviour is the episode id's map.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.citynav import bench_fullset as bf
from benchmarks.citynav import episodes
from benchmarks.citynav import run_manifest as rm
from benchmarks.citynav.analysis.rows import iter_json_rows
from benchmarks.citynav.analysis.tests.ledger import ledger_row

# A pytest.fail inside a worker thread fails its test instead of only warning.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

REPO = Path(bf.__file__).resolve().parents[2]

# map -> (status, error, has a workdir)
OUTCOMES = {
    "ok": ("ok", None, True),
    "miss": ("ok", "RuntimeError: boom", True),
    "infra": ("infra", "SamServerOutage: down", True),
    "sleep": ("timed_out", None, True),
    "kill": ("infra", "killed_midrun", True),
    "norow": ("infra", "no_prediction_unclassified", True),
    "reasoner": ("infra", "reasoner_down_midrun", True),
    "sam": ("infra", "sam_down_midrun", True),
    "exit": ("infra", "exited 3 with no record", True),
    "nodir": ("infra", "exited 2 with no record", False),
}


def _infra(i: int, cause: str = "no_prediction_unclassified", **extra) -> dict:
    return {"episode_id": f"ep{i}", "worker_id": 0, "status": "infra", "error": cause, **extra}


def test_each_child_outcome_lands_its_row(driver, monkeypatch) -> None:
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    monkeypatch.setattr(bf, "REASONER_PROBE_S", 0.2)
    d = driver(selection=[f"{k}::1#{i}" for i, k in enumerate(OUTCOMES)])
    d.episode_timeout_s, d.max_infra_attempts = 3, 1
    # The reasoner is down exactly while the "reasoner" episode flies.
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "_reasoner_answers",
                        lambda: not any(e.startswith("reasoner") for e in d._in_flight))
    # SAM is down exactly while the "sam" episode flies.
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    monkeypatch.setattr(d, "_sam_answers", lambda: not any(e.startswith("sam") for e in d._in_flight))
    t0 = time.time()
    d.worker_loop(0)
    rows = {r["episode_id"].split("::")[0]: r for r in iter_json_rows(d.episodes_file)}
    got = {k: (r["status"], r.get("error"), "workdir" in r) for k, r in rows.items()}
    assert got == OUTCOMES
    assert rows["ok"]["workdir"] == "w0/maps_20260101_000000_citynav_ok"
    assert rows["ok"]["metrics"]["SR"] == 1 and rows["ok"]["worker_id"] == 0
    # The mission's backstop leaves the reserve of the cap for the reply.
    assert t0 < rows["ok"]["deadline"] < t0 + 3 * (1 - bf._DEADLINE_RESERVE_FRAC) + 5
    assert not (d.out_root / "w0" / "maps_20260101_000000_citynav_sleep" / "episode.json").exists()


def test_the_health_gate_asks_where_discovery_listed_the_models(driver, monkeypatch) -> None:
    """An endpoint listing its models away from SEARCH_REASONER_URL/models names the listing in
    SEARCH_REASONER_MODELS_URL: a gate asking the default would hold every launch on a reasoner
    discovery accepted."""
    asked: list[str] = []

    class _Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _open(req, timeout):
        asked.append(req.full_url)
        return _Answer()

    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    monkeypatch.setenv("SEARCH_REASONER_URL", "https://reasoner.example/serve/v1")
    monkeypatch.setenv("SEARCH_REASONER_MODELS_URL", "https://reasoner.example/catalog/models")
    assert driver()._reasoner_answers()
    monkeypatch.delenv("SEARCH_REASONER_MODELS_URL")
    monkeypatch.setenv("SEARCH_REASONER_URL", "https://maps-sol.openai.azure.com/openai/v1")
    assert driver()._reasoner_answers()
    monkeypatch.setenv("SEARCH_REASONER_URL", "http://127.0.0.1:8001/v1")
    assert driver()._reasoner_answers()
    monkeypatch.delenv("SEARCH_REASONER_URL")  # unset: the gate asks where discovery did
    assert driver()._reasoner_answers()
    assert asked == ["https://reasoner.example/catalog/models", "https://maps-sol.openai.azure.com/openai/v1/models",
                     "http://127.0.0.1:8001/v1/models", "http://127.0.0.1:8001/v1/models"]
    monkeypatch.setenv("SEARCH_REASONER_URL", "")  # no URL to ask is a reasoner that does not answer
    assert not driver()._reasoner_answers()


def test_a_window_of_infra_casualties_stops_the_run(driver, tmp_path: Path) -> None:
    """Named by its dominant cause, with the tail of the log that explains it:
    the attempt's run.log, or the worker's server.log for an attempt that
    never opened a workdir. Nothing is scored, so every episode stays
    re-queued."""
    wd = "w0/maps_20260805_101500_citynav_ep0"
    (tmp_path / wd).mkdir(parents=True)
    (tmp_path / wd / "run.log").write_text("FileNotFoundError: birmingham_block_1_meta.json\n")
    (tmp_path / "w1").mkdir()
    (tmp_path / "w1" / "server.log").write_text("ERROR endpoint discovery failed\n")
    d = driver()
    for i in range(bf.ABORT_WINDOW - 1):
        d._append_result(_infra(i, workdir=wd))
        assert d._abort is None, f"fired after only {i + 1} rows"
    d._append_result(_infra(99, worker_id=1))
    assert d._abort and "no_prediction_unclassified" in d._abort
    text = d.driver_log.read_text()
    assert "ABORTING RUN" in text and "nothing was scored" in text
    assert "endpoint discovery failed" in text, "the excerpt did not read server.log"
    assert d._next_claimed_episode() is None
    assert d._done_cache == set()

    fresh = driver(tmp_path / "w0")
    fresh._append_result(_infra(0, cause="mission ended_by=transient_failures",
                                workdir="nonexistent"))
    fresh._append_result({"episode_id": "ep100", "status": "ok", "error": "RuntimeError: x"})
    for i in range(1, bf.ABORT_WINDOW):
        fresh._append_result(_infra(i, cause="mission ended_by=transient_failures"))
    assert "transient_failures" in fresh._abort
    assert "1 attempt(s) were scored" in fresh.driver_log.read_text()


def test_scattered_infra_and_timeouts_never_stop_the_run(driver) -> None:
    d = driver()
    for i in range(200):
        d._append_result(_infra(i) if i % 8 == 0 else
                         {"episode_id": f"ep{i}", "status": "timed_out"})
    assert d._abort is None and d._next_claimed_episode() is not None


def test_claims_hand_each_episode_to_one_worker(driver) -> None:
    d = driver(selection=["a", "b", "c"])
    assert [d._next_claimed_episode() for _ in range(3)] == ["a", "b", "c"]
    assert d._next_claimed_episode() is None, "a live episode was handed out twice"
    d._append_result({"episode_id": "a", "status": "ok"})
    d._in_flight.discard("a")
    d._append_result({"episode_id": "b", "status": "infra", "error": "killed_midrun"})
    d._in_flight.discard("b")
    d._in_flight.discard("c")  # released with no row
    assert d._next_claimed_episode() == "b" and d._next_claimed_episode() == "c"
    d.max_infra_attempts = 1
    d._in_flight.clear()
    assert d._next_claimed_episode() == "c", "a capped infra episode was re-flown"


def test_the_summary_is_the_run_report_under_the_infra_lines(driver) -> None:
    d = driver()
    d.episodes_file.write_text("".join(json.dumps(r) + "\n" for r in (
        ledger_row("c::3#6534", "20260725_111500", 0),
        ledger_row("c::3#6534", "20260725_110958", 1),
        ledger_row("e::1#1", "20260725_110000", 1),
        ledger_row("g::2#2", "20260725_110100", status="infra", error="reasoner_down_midrun"),
    )))
    d._write_summary()
    out = d.summary_file.read_text()
    assert "infra attempts (excluded): 1  by cause: reasoner_down_midrun (1)" in out
    # The first-STARTED attempt of c scores, not the one that finished last.
    assert "SR = 2/2 (100.0%)" in out, out

    (d.out_root / "empty").mkdir()
    empty = driver(d.out_root / "empty")
    empty._write_summary()
    assert empty.summary_file.read_text().splitlines()[1:] == ["infra attempts (excluded): 0"]


@pytest.fixture
def no_flight(monkeypatch):
    monkeypatch.setattr(bf, "_missing_block_assets", lambda eps: [])
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: {})
    monkeypatch.setattr(rm, "emit_or_verify", lambda *a, **k: None)


def test_an_aborted_run_exits_3_and_the_summary_says_why(driver, monkeypatch, no_flight) -> None:
    d = driver()
    monkeypatch.setattr(d, "_select_episodes", lambda: [f"ep{i}" for i in range(60)])
    d._deadline = time.monotonic() + 15  # bounds a driver that never aborts
    seq = iter(range(60))

    def one_window(worker_id: int) -> None:
        for i in seq:
            if d._next_claimed_episode() is None:
                return
            d._append_result(_infra(i))

    monkeypatch.setattr(d, "worker_loop", one_window)
    assert d.run() == 3
    assert "ABORTED" in d.summary_file.read_text()
    assert len(d.episodes_file.read_text().splitlines()) < 60, "it did not stop early"


def test_persistent_infra_is_left_out_and_reported(driver, tmp_path, monkeypatch, no_flight) -> None:
    for cause in ("A", "A", "B"):
        episodes.append_row(tmp_path, _infra(0, cause=cause))
    d = driver()
    monkeypatch.setattr(d, "_select_episodes", lambda: ["ep0"])
    monkeypatch.setattr(d, "worker_loop", lambda w: pytest.fail("a capped episode flew"))
    d._deadline = time.monotonic() + 5
    assert d.run() == 0
    for text in (d.summary_file.read_text(), d.driver_log.read_text()):
        assert "persistent infra: 1 episode(s)" in text and "A (2), B (1)" in text, text


def test_a_spent_wall_budget_stops_the_run(driver, monkeypatch, no_flight) -> None:
    d = driver()
    monkeypatch.setattr(d, "_select_episodes", lambda: ["ep0"])
    monkeypatch.setattr(d, "worker_loop", lambda w: pytest.fail("flew past the budget"))
    d.max_wall_s, d._deadline = 1, time.monotonic() - 1
    assert d.run() == 0
    assert "wall budget reached" in d.driver_log.read_text() and d.summary_file.exists()


def test_a_second_run_into_a_held_root_is_refused(driver, tmp_path, monkeypatch) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time\nfrom pathlib import Path\nfrom benchmarks.citynav import episodes\n"
         "lock = episodes.hold_root(Path(sys.argv[1]))\nprint('held', flush=True)\n"
         "time.sleep(60)\n", str(tmp_path)],
        cwd=REPO, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(rm, "build_manifest", lambda **kw: pytest.fail("flew"))
        d = driver()
        monkeypatch.setattr(d, "_select_episodes", lambda: ["ep0"])
        assert d.run() == 2 and f"pid {holder.pid}" in d.driver_log.read_text()
    finally:
        holder.kill()
        holder.wait()


def test_missing_block_assets_fail_before_flying(driver, tmp_path, monkeypatch) -> None:
    from benchmarks.citynav import paths

    tiles, rgbd = tmp_path / "tiles", tmp_path / "rgbd"
    tiles.mkdir()
    rgbd.mkdir()
    for suffix in paths.BAKE_OUTPUT_SUFFIXES:
        (tiles / f"blk{suffix}").write_text("{}")
    (rgbd / "blk.tif").write_bytes(b"")
    monkeypatch.setattr(paths, "TILES_DIR", tiles)
    monkeypatch.setattr(paths, "CITYNAV_NATIVE_RGBD_DIR", rgbd)
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: pytest.fail("flew"))
    (tmp_path / "run").mkdir()
    d = driver(tmp_path / "run")
    monkeypatch.setattr(d, "_select_episodes", lambda: ["blk::1#0"])
    assert d.run() == 2 and str(rgbd / "blk_rgb.tif") in d.driver_log.read_text()
    (rgbd / "blk.png").write_bytes(b"")  # the CityNav raster pair
    assert bf._missing_block_assets(["blk::1#0"]) == []


def test_a_ledger_the_worker_cannot_write_stops_the_run(driver, monkeypatch) -> None:
    d = driver(selection=["ep0"])

    def full(root, row):
        raise OSError(28, "No space left on device")

    def boom(worker_id, eid):
        raise RuntimeError("the attempt failed")

    monkeypatch.setattr(episodes, "append_row", full)
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "run_one", boom)
    d.worker_loop(0)
    assert "No space left on device" in d._abort
    assert "No space left on device" in d.driver_log.read_text()
    assert d._next_claimed_episode() is None


_CHILD = (
    "import time\n"
    "from benchmarks.citynav.run_benchmark.cli import _die_with_parent\n"
    "_die_with_parent()\n"
    "print('armed', flush=True)\n"
    "time.sleep(60)\n"
)
_PARENT = (
    "import os, subprocess, sys, time\n"
    "env = {**os.environ, 'CITYNAV_BENCH_PARENT_PID': str(os.getpid())}\n"
    f"c = subprocess.Popen([sys.executable, '-c', {_CHILD!r}], env=env,\n"
    "                     start_new_session=True, stdout=subprocess.PIPE, text=True)\n"
    "c.stdout.readline()\n"
    "print(c.pid, flush=True)\n"
    "time.sleep(60)\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split()[2] != "Z"


def test_a_sigkilled_driver_takes_its_attempt_with_it() -> None:
    """The attempt sits in its own session, so nothing else reaps it."""
    parent = subprocess.Popen([sys.executable, "-c", _PARENT], cwd=REPO,
                              stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    try:
        assert _alive(child)
        parent.send_signal(signal.SIGKILL)
        parent.wait()
        deadline = time.monotonic() + 5.0
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child), "the attempt outlived its driver"
    finally:
        parent.kill()
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


_INTERRUPTED = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from benchmarks.citynav import bench_fullset as bf, run_manifest as rm\n"
    "d = bf.Driver.__new__(bf.Driver)\n"
    "root = Path(sys.argv[1])\n"
    "d.out_root, d.episodes_file, d.driver_log = root, root / 'episodes.jsonl', root / 'driver.log'\n"
    "d.summary_file, d.split, d.workers, d.max_wall_s, d._deadline = root / 'summary.txt', 's', 2, 0, None\n"
    "d._done_cache, d._infra_errors, d.max_infra_attempts, d._abort = set(), {}, 3, None\n"
    "d.sample_seed, d.limit, d.allowlist_path, d.arrivals_path = 1, 0, None, None\n"
    "d._select_episodes = lambda: ['ep0']\n"
    "bf._missing_block_assets, bf.WORKER_STAGGER_S = (lambda eps: []), 0\n"
    "rm.build_manifest = lambda **kw: None\n"
    "rm.emit_or_verify = lambda *a, **kw: None\n"
    "def fly(wid):\n"
    "    print('flying', flush=True)\n"
    "    time.sleep(60)\n"
    "d.worker_loop = fly\n"
    "d.run()\n"
)


def test_an_interrupted_driver_exits_with_its_workers(tmp_path) -> None:
    """Ctrl-C releases the root, so a worker left flying would fly beside the next launch."""
    # A shell that ignores SIGINT hands that to its children; the driver is
    # launched with the default handler, as from a terminal.
    proc = subprocess.Popen([sys.executable, "-c", _INTERRUPTED, str(tmp_path)], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        flying = (line for line in iter(proc.stdout.readline, "") if line == "flying\n")
        assert next(flying) and next(flying)
        time.sleep(0.5)    # the main thread is joining its first worker
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) != 0
    finally:
        proc.kill()
        proc.wait()


def test_the_split_defaults_to_the_benchmarks_own() -> None:
    from benchmarks.citynav import paths

    assert bf._build_parser().parse_args([]).split == paths.DEFAULT_SPLIT


def test_the_out_root_and_the_usage_name_the_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(bf.paths, "BENCHMARK", "calcity")
    assert bf._build_parser().parse_args([]).out_root == "output/calcity_fullset"
    assert bf._build_parser().prog == "python -m benchmarks.calcity.bench_fullset"
    assert bf._build_parser().description == "Continuous full-split calcity benchmark at fixed concurrency."


def test_the_summary_names_an_errorless_infra_row_as_the_driver_counts_it(driver) -> None:
    d = driver()
    episodes.append_row(d.out_root, {"episode_id": "ep0", "worker_id": 0, "status": "infra"})
    # The driver's one infra predicate: a mission the serve ended counts too.
    episodes.append_row(d.out_root, {"episode_id": "ep1", "worker_id": 0, "status": "ok",
                                     "metrics": {"mission": {"ended_by": "transient_failures"}}})
    d._write_summary()
    assert "infra attempts (excluded): 2  by cause: infra (2)" in d.summary_file.read_text()


def test_the_end_line_prints_no_score_for_an_infra_attempt(driver, monkeypatch) -> None:
    d = driver(selection=["a::1#0", "b::1#1"])
    rows = {"a": {**_infra(0), "metrics": {"NE": 3.0, "SR": 1}, "elapsed_s": 1.0},
            "b": {"episode_id": "b::1#1", "status": "ok", "metrics": {"NE": 3.0, "SR": 1}, "elapsed_s": 1.0}}
    monkeypatch.setattr(d, "_run_attempt", lambda w, eid: rows[eid[0]])
    d.run_one(0, "a::1#0")
    d.run_one(0, "b::1#1")
    ends = [ln for ln in d.driver_log.read_text().splitlines() if " END " in ln]
    assert ends[0].endswith("status=infra elapsed=1s"), ends
    assert ends[1].endswith("status=ok elapsed=1s NE=3.0 SR=1"), ends


def test_the_dry_run_counts_a_persistent_infra_episode_as_left_out(driver, tmp_path, monkeypatch, capsys) -> None:
    for _ in range(3):
        episodes.append_row(tmp_path, _infra(0))
    d = driver()
    monkeypatch.setattr(d, "_select_episodes", lambda say: ["ep0", "ep1"])
    assert d.dry_run() == 0
    out = capsys.readouterr().out
    assert "pending: 1" in out and "[PEND] ep1" in out and "[PEND] ep0" not in out, out


def test_relative_paths_resolve_against_the_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    d = bf.Driver(bf._build_parser().parse_args(["--out-root", "runs/a", "--episode-allowlist", "ids.txt"]))
    assert d.out_root == tmp_path / "runs" / "a"
    assert d.allowlist_path == tmp_path / "ids.txt"


def test_a_reasoner_that_never_answers_stops_at_the_wall_budget(driver, monkeypatch) -> None:
    """The launch gate waits for the reasoner only while the wall budget lasts: past it the
    worker claims nothing and launches nothing."""
    import threading

    d = driver(selection=["ep0"])
    d._deadline = time.monotonic() + 1
    monkeypatch.setattr(d, "_reasoner_answers", lambda: False)
    monkeypatch.setattr(d, "run_one", lambda *a: pytest.fail("launched with no reasoner"))
    t = threading.Thread(target=d.worker_loop, args=(0,), daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive()
    assert not d.episodes_file.exists()


def test_a_segmenter_that_never_answers_holds_every_launch(driver, monkeypatch) -> None:
    """An episode launched with SAM down would spend reasoner calls on a blind search: the
    gate holds it, the reasoner up, until the wall budget is spent."""
    import threading

    d = driver(selection=["ep0"])
    d._deadline = time.monotonic() + 1
    monkeypatch.setattr(d, "_reasoner_answers", lambda: True)
    monkeypatch.setattr(d, "_sam_answers", lambda: False)
    monkeypatch.setattr(d, "run_one", lambda *a: pytest.fail("launched with no segmenter"))
    t = threading.Thread(target=d.worker_loop, args=(0,), daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive()
    assert not d.episodes_file.exists()


def test_the_sam_gate_asks_the_replicas_discovery_checked(monkeypatch) -> None:
    """Any replica answering ``/health`` passes (the client steps off a dead one); none checked
    at discovery passes as discovery did; every replica down fails."""
    from dataclasses import replace

    from maps.llm import endpoint_spec as E
    from maps.search.tests.fakes import TEST_SPEC

    asked: list[str] = []
    up: set[str] = set()

    class _Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _open(url, timeout):
        asked.append(url)
        if url not in up:
            raise OSError("connection refused")
        return _Answer()

    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    monkeypatch.setattr(E, "_SPEC", replace(TEST_SPEC, sam_url="http://a:8100, http://b:8100/"))
    assert not bf.Driver._sam_answers()
    up.add("http://b:8100/health")
    assert bf.Driver._sam_answers()
    assert asked == ["http://a:8100/health", "http://b:8100/health", "http://a:8100/health", "http://b:8100/health"]
    monkeypatch.setattr(E, "_SPEC", replace(TEST_SPEC, sam_url=""))
    assert bf.Driver._sam_answers() and len(asked) == 4
    monkeypatch.setattr(E, "_SPEC", None)
    assert bf.Driver._sam_answers()


@pytest.mark.parametrize("flag", ["--workers", "--max-infra-attempts", "--episode-timeout-s"])
def test_the_parser_refuses_a_count_below_one(flag, capsys) -> None:
    with pytest.raises(SystemExit) as done:
        bf._build_parser().parse_args([flag, "0"])
    assert done.value.code == 2 and flag in capsys.readouterr().err


def test_a_session_releases_each_report_at_its_arrival(driver, tmp_path, monkeypatch) -> None:
    stream = tmp_path / "stream.json"
    stream.write_text(json.dumps({"reports": [{"report": 0, "arrival_s": 0.0, "episode_id": "b"},
                                              {"report": 1, "arrival_s": 0.6, "episode_id": "a"}]}))
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a", "b", "c"])
    d = driver()
    d.arrivals_path = stream
    d._selection = d._select_episodes()
    assert d._selection == ["b", "a"], "the session flies its stream in arrival order"
    d._t0 = time.monotonic()
    assert d._next_claimed_episode() == "b"
    assert d._next_claimed_episode() == "a" and d._session_s() >= 0.6, "a report was released early"
    assert d._next_claimed_episode() is None

    monkeypatch.setattr(d, "_run_attempt", lambda w, eid: {"episode_id": eid, "status": "ok", "elapsed_s": 1.0})
    d.run_one(0, "a")
    row = next(iter_json_rows(d.episodes_file))
    assert row["arrival_s"] == 0.6 and row["claimed_s"] >= 0.6


def test_a_session_refuses_a_root_that_already_holds_a_ledger(driver, tmp_path, monkeypatch, no_flight) -> None:
    stream = tmp_path / "stream.json"
    stream.write_text(json.dumps({"reports": [{"report": 0, "arrival_s": 0.0, "episode_id": "a"}]}))
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    episodes.append_row(tmp_path, {"episode_id": "z", "status": "ok"})
    monkeypatch.setattr(rm, "emit_or_verify", lambda *a, **k: pytest.fail("a refused session touched the manifest"))
    d = driver()
    d.arrivals_path = stream
    monkeypatch.setattr(d, "worker_loop", lambda w: pytest.fail("a resumed session flew"))
    assert d.run() == 2 and "fresh --out-root" in d.driver_log.read_text()


def test_a_session_s_manifest_names_the_stream_it_flies(driver, tmp_path, monkeypatch, no_flight) -> None:
    stream = tmp_path / "stream.json"
    stream.write_text(json.dumps({"reports": [{"report": 0, "arrival_s": 0.0, "episode_id": "a"}]}))
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    built = {}
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: built.update(kw) or {})
    d = driver()
    d.arrivals_path = stream
    monkeypatch.setattr(d, "worker_loop", lambda w: None)
    d.run()
    assert built["arrivals"] == hashlib.sha256(stream.read_bytes()).hexdigest()


@pytest.mark.parametrize("flags", [["--limit", "5"], ["--sample-seed", str(bf.SAMPLE_SEED)]])
def test_arrivals_refuse_the_shuffle_flags(flags, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bf, "Driver", lambda args: pytest.fail("a session took a shuffle flag"))
    with pytest.raises(SystemExit):
        bf.main(["--arrivals", "s.json", *flags])
    assert "--arrivals selects" in capsys.readouterr().err


def test_a_baseline_arm_whose_model_has_no_dial_off_is_refused_at_start(tmp_path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    from maps.llm import endpoint_spec

    monkeypatch.setattr(bf, "CONFIG", REPO / "configs/baselines/geonav.yaml")
    monkeypatch.setattr(endpoint_spec, "discover_from_env", lambda: SimpleNamespace(model_id="gemma4-26b-a4b-nvfp4"))
    monkeypatch.setattr(endpoint_spec, "install_spec", lambda spec: None)
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("a refused arm flew"))
    assert bf.main(["--out-root", str(tmp_path / "r")]) == 2
    assert not (tmp_path / "r" / episodes.EPISODES_JSONL).exists()
    assert "gemma4-26b-a4b-nvfp4" in capsys.readouterr().err


def test_a_session_with_no_reports_reads_as_offered_nothing(driver, tmp_path, monkeypatch, no_flight) -> None:
    from benchmarks.capacity import reader

    stream = tmp_path / "stream.json"
    stream.write_text(json.dumps({"load_per_h": 0.5, "hours": 1.0, "seed": 3, "reports": []}))
    d = driver(tmp_path / "s")
    d.out_root.mkdir()
    d.arrivals_path = stream
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    assert d.run() == 0
    s = reader.session(d.out_root)
    assert (s["offered"], s["finds_per_h"]) == (0, 0.0)


def test_a_session_s_preview_names_its_stream_not_a_seed(driver, tmp_path, monkeypatch, capsys) -> None:
    stream = tmp_path / "stream.json"
    stream.write_text(json.dumps({"reports": [{"report": 0, "arrival_s": 0.0, "episode_id": "a"}]}))
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    d = driver()
    d.arrivals_path = stream
    assert d.dry_run() == 0
    assert capsys.readouterr().out.startswith(f"arrivals: {stream}   split:")


def test_a_selection_naming_episodes_off_the_split_is_refused_fatal(driver, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    d = driver()
    d.arrivals_path = tmp_path / "stream.json"
    d.arrivals_path.write_text(json.dumps({"reports": [{"report": 0, "arrival_s": 0.0, "episode_id": "z"}]}))
    with pytest.raises(SystemExit, match="^FATAL: --arrivals names 1 episodes"):
        d._select_episodes()
    d.arrivals_path, d.allowlist_path = None, tmp_path / "absent.txt"
    with pytest.raises(SystemExit, match="^FATAL: --episode-allowlist read failed"):
        d._select_episodes()


def test_a_refused_selection_exits_2_and_says_so_in_the_driver_log(driver, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    d = driver()
    d.arrivals_path = tmp_path / "stream.json"
    d.arrivals_path.write_text(json.dumps({"reports": [{"report": 0, "arrival_s": 0.0, "episode_id": "z"}]}))
    assert d.run() == 2
    assert "FATAL: --arrivals names 1 episodes" in d.driver_log.read_text()


@pytest.mark.parametrize("dry", [[], ["--dry-run"]])
def test_a_refused_config_exits_2_before_any_out_root(dry, tmp_path, monkeypatch, capsys) -> None:
    bad = tmp_path / "arm.yaml"
    bad.write_text(f"base: {REPO / 'configs/citynav.yaml'}\npolicy: nope\n")
    monkeypatch.setattr(bf, "CONFIG", bad)
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("a refused config flew"))
    assert bf.main(["--out-root", str(tmp_path / "r"), *dry]) == 2
    assert not (tmp_path / "r").exists()
    assert capsys.readouterr().err.startswith("FATAL: ")


@pytest.mark.parametrize("stream", ["{}", "not json", '{"reports": [{"episode_id": "a"}]}', None])
def test_an_unreadable_session_stream_is_refused_fatal(stream, driver, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a"])
    d = driver()
    d.arrivals_path = tmp_path / "stream.json"
    if stream is not None:
        d.arrivals_path.write_text(stream)
    with pytest.raises(SystemExit, match="^FATAL: --arrivals read failed"):
        d._select_episodes()


@pytest.mark.parametrize("ids", [["z"], ["a", "z"]])
def test_an_allowlist_naming_ids_off_the_split_is_refused_fatal(ids, driver, tmp_path, monkeypatch) -> None:
    """Every allowlisted id must be in the split: one that is not is a wrong
    split or a stale list, and a run over the rest would read as complete."""
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a", "b"])
    d = driver()
    d.allowlist_path = tmp_path / "ids.txt"
    d.allowlist_path.write_text("\n".join(ids) + "\n")
    assert d.run() == 2
    assert "FATAL: --episode-allowlist names 1 ids not in the train_seen split, e.g. z" in d.driver_log.read_text()


@pytest.mark.parametrize("refusal", ["split", "stream", "allowlist", "partial"])
def test_a_refused_dry_run_exits_2_with_no_out_root(refusal, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("a dry run flew"))
    if refusal != "split":
        monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a", "b"])
    (tmp_path / "stream.json").write_text("{}")
    (tmp_path / "ids.txt").write_text("a\nz\n")
    flags = {"split": ["--split", "nope"], "stream": ["--arrivals", str(tmp_path / "stream.json")],
             "allowlist": ["--episode-allowlist", str(tmp_path / "absent.txt")],
             "partial": ["--episode-allowlist", str(tmp_path / "ids.txt")]}[refusal]
    assert bf.main(["--out-root", str(tmp_path / "r"), "--dry-run", *flags]) == 2
    assert not (tmp_path / "r").exists()
    assert capsys.readouterr().err.startswith("FATAL: ")


def test_a_dry_run_prints_the_allowlist_count_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bf, "_enumerate_split", lambda split: ["a", "b"])
    (tmp_path / "ids.txt").write_text("a\n")
    assert bf.main(["--out-root", str(tmp_path / "r"), "--dry-run",
                    "--episode-allowlist", str(tmp_path / "ids.txt")]) == 0
    assert not (tmp_path / "r").exists()
    assert "1 ids requested -> 1 match" in capsys.readouterr().out
