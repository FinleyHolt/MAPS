"""The bench driver: one child per attempt, its row, the claims, the stops.

The children are ``fake_runner``, whose behaviour is the episode id's map.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks import sandbox
from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav import bench_fullset as bf
from benchmarks.citynav.paths import citynav_bench
from benchmarks.run_record import episodes
from benchmarks.run_record import run_manifest as rm
from benchmarks.run_record.rows import iter_json_rows
from benchmarks.run_record.tests.ledger import ledger_row

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


@pytest.fixture
def bench_data(tmp_path, monkeypatch) -> Path:
    """A CityNav data root holding what ``assert_available`` checks: the splits
    directory and the object catalog."""
    root = tmp_path / "citynav"
    (root / "data" / "citynav").mkdir(parents=True)
    (root / "data" / "cityrefer").mkdir(parents=True)
    (root / "data" / "cityrefer" / "objects.json").write_text("{}")
    monkeypatch.setenv("SEARCH_CITYNAV_DATA", str(root))
    return root


def _infra(i: int, cause: str = "no_prediction_unclassified", **extra) -> dict:
    return {"episode_id": f"ep{i}", "worker_id": 0, "status": "infra", "error": cause, **extra}


def test_each_child_outcome_lands_its_row(driver, monkeypatch) -> None:
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    monkeypatch.setattr(bf, "REASONER_PROBE_S", 0.2)
    d = driver(selection=[f"{k}::1#{i}" for i, k in enumerate(OUTCOMES)])
    d.episode_timeout_s, d.max_infra_attempts = 3, 1
    # The reasoner is down exactly while the "reasoner" episode flies.
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(
        d, "_reasoner_answers", lambda: not any(e.startswith("reasoner") for e in d._in_flight)
    )
    # SAM is down exactly while the "sam" episode flies.
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    monkeypatch.setattr(
        d, "_sam_answers", lambda: not any(e.startswith("sam") for e in d._in_flight)
    )
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


def test_a_block_past_the_attempt_cap_is_the_model_s_error_not_a_crash(driver, monkeypatch):
    """Under the cap the block's allocation raises MemoryError inside the block: the attempt
    lands its row and is scored, where an uncapped child is OOM-killed and re-queued as infra.
    The child caps itself, so a host without util-linux's ``prlimit`` flies it too."""
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    monkeypatch.setenv("PATH", "/nonexistent")
    d = driver(selection=["hog::1#0"])
    d.memory_gib = 1.0  # under the block's 1.5 GiB list, over the fake attempt's own footprint
    d.max_infra_attempts = 1
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    d.worker_loop(0)
    (row,) = iter_json_rows(d.episodes_file)
    assert row["status"] == "ok" and row["error"].startswith("line 1: MemoryError"), row


def test_an_attempt_gets_the_reasoner_key_and_none_of_the_launcher_s_other_secrets(
    driver, monkeypatch
):
    """The attempt's environment is the allowlist: a token in the launching shell never
    reaches model code, the key the attempt calls the reasoner with does, and so do Cal
    City's data-root fallback and the hub cache's second name."""
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    monkeypatch.setenv("FAKE_SECRET", "s")
    monkeypatch.setenv("SEARCH_REASONER_API_KEY", "k")
    monkeypatch.setenv("CALCITY_ROOT", "/data/calcity")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/data/hub")
    d = driver(selection=["env::1#0"])
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    d.worker_loop(0)
    (row,) = iter_json_rows(d.episodes_file)
    assert row["env"] == {
        "FAKE_SECRET": None,
        "SEARCH_REASONER_API_KEY": "k",
        "CALCITY_ROOT": "/data/calcity",
        "HUGGINGFACE_HUB_CACHE": "/data/hub",
    }, row


@pytest.mark.skipif(
    sandbox.bwrap_refusal() is not None,
    reason="bwrap is absent or this host blocks user namespaces",
)
def test_an_attempt_under_bubblewrap_lands_its_row_and_a_timeout_kills_it(driver, monkeypatch):
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    d = driver(selection=["ok::1#0", "sleep::1#1"])
    d._bwrap, d.episode_timeout_s, d.max_infra_attempts = True, 3, 1
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    d.worker_loop(0)
    rows = {r["episode_id"].split("::")[0]: r for r in iter_json_rows(d.episodes_file)}
    assert rows["ok"]["status"] == "ok" and rows["ok"]["metrics"]["SR"] == 1, rows
    assert rows["sleep"]["status"] == "timed_out", rows
    # The killed attempt's sandbox died with it.
    ps = subprocess.run(["pgrep", "-f", str(d.out_root)], capture_output=True, text=True)
    assert ps.stdout == "", ps.stdout


def _gone(pid: int) -> bool:
    """True once ``pid`` has exited and been reaped, within five seconds."""
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


_no_scope = pytest.mark.skipif(
    not sandbox.scope_works(), reason="no systemd user manager starts scopes here"
)


@_no_scope
def test_shared_memory_past_the_cap_kills_the_attempt_in_its_scope(driver, monkeypatch):
    """Shared anonymous memory escapes RLIMIT_DATA; the scope's MemoryMax counts it, so the
    hold ends the attempt (a signal, re-queued as infra) and the host keeps its memory."""
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    d = driver(selection=["shm::1#0"])
    d._scope, d.memory_gib, d.max_infra_attempts = True, 1.0, 1
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    d.worker_loop(0)
    (row,) = iter_json_rows(d.episodes_file)
    assert (row["status"], row["error"]) == ("infra", "killed_midrun"), row


@_no_scope
def test_a_daemon_an_attempt_leaves_behind_dies_with_its_scope(driver, monkeypatch):
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    d = driver(selection=["orphan::1#0"])
    d._scope, d.max_infra_attempts = True, 1
    monkeypatch.setattr(d, "_wait_for_reasoner", lambda: True)
    monkeypatch.setattr(d, "_wait_for_sam", lambda: True)
    d.worker_loop(0)
    (row,) = iter_json_rows(d.episodes_file)
    pid = int((d.out_root / row["workdir"] / "daemon.pid").read_text())
    alive = not _gone(pid)
    if alive:
        os.kill(pid, signal.SIGKILL)
    assert row["status"] == "ok" and not alive, row


def test_the_health_gate_asks_where_discovery_listed_the_models(driver, monkeypatch) -> None:
    """An endpoint listing its models away from SEARCH_REASONER_URL/models names the listing in
    SEARCH_REASONER_MODELS_URL: a gate asking the default would hold every launch on a reasoner
    discovery accepted."""
    from dataclasses import replace

    from maps.mission.tests.fakes import TEST_SPEC

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

    served = {}

    def installed(base_url):
        served["spec"] = replace(TEST_SPEC, base_url=base_url)

    def flying() -> bf.Driver:
        """A driver whose bench flies the spec last installed."""
        d = driver()
        d.bench = replace(d.bench, endpoint_spec=served["spec"])
        return d

    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    monkeypatch.setenv(
        "SEARCH_REASONER_URL", "http://never-asked/v1"
    )  # the gate asks the spec's URL
    installed("https://reasoner.example/serve/v1")
    monkeypatch.setenv("SEARCH_REASONER_MODELS_URL", "https://reasoner.example/catalog/models")
    assert flying()._reasoner_answers()
    monkeypatch.delenv("SEARCH_REASONER_MODELS_URL")
    installed("https://example.openai.azure.com/openai/v1")
    assert flying()._reasoner_answers()
    installed("http://127.0.0.1:8001/v1")
    assert flying()._reasoner_answers()
    assert asked == [
        "https://reasoner.example/catalog/models",
        "https://example.openai.azure.com/openai/v1/models",
        "http://127.0.0.1:8001/v1/models",
    ]
    installed("")  # an offline spec names no URL: a reasoner that does not answer
    assert not flying()._reasoner_answers()
    bearers: list = []
    monkeypatch.setattr(
        bf.urllib.request,
        "urlopen",
        lambda req, timeout: bearers.append(req.get_header("Authorization")) or _Answer(),
    )
    monkeypatch.setenv("SEARCH_REASONER_API_KEY", "k")
    installed("http://127.0.0.1:8001/v1")
    assert flying()._reasoner_answers() and bearers == [
        "Bearer k"
    ]  # hosted endpoints auth the listing


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
    fresh._append_result(
        _infra(0, cause="mission ended_by=transient_failures", workdir="nonexistent")
    )
    fresh._append_result({"episode_id": "ep100", "status": "ok", "error": "RuntimeError: x"})
    for i in range(1, bf.ABORT_WINDOW):
        fresh._append_result(_infra(i, cause="mission ended_by=transient_failures"))
    assert "transient_failures" in fresh._abort
    assert (
        "1 episode(s) were scored; every infra attempt is excluded" in fresh.driver_log.read_text()
    )


def test_scattered_infra_and_timeouts_never_stop_the_run(driver) -> None:
    d = driver()
    for i in range(200):
        d._append_result(
            _infra(i) if i % 8 == 0 else {"episode_id": f"ep{i}", "status": "timed_out"}
        )
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
    d.episodes_file.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in (
                ledger_row("c::3#6534", "20260725_111500", 0),
                ledger_row("c::3#6534", "20260725_110958", 1),
                ledger_row("e::1#1", "20260725_110000", 1),
                ledger_row(
                    "g::2#2", "20260725_110100", status="infra", error="reasoner_down_midrun"
                ),
            )
        )
    )
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
    monkeypatch.setattr(rm, "build_manifest", lambda **kw: {})
    monkeypatch.setattr(rm, "emit_or_verify", lambda *a, **k: None)


def test_an_aborted_run_exits_3_and_the_summary_says_why(driver, monkeypatch, no_flight) -> None:
    d = driver(selection=[f"ep{i}" for i in range(60)])
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


def test_persistent_infra_is_left_out_and_reported(
    driver, tmp_path, monkeypatch, no_flight
) -> None:
    for cause in ("A", "A", "B"):
        episodes.append_row(tmp_path, _infra(0, cause=cause))
    d = driver(selection=["ep0"])
    monkeypatch.setattr(d, "worker_loop", lambda w: pytest.fail("a capped episode flew"))
    d._deadline = time.monotonic() + 5
    assert d.run() == 0
    for text in (d.summary_file.read_text(), d.driver_log.read_text()):
        assert "persistent infra: 1 episode(s)" in text and "A (2), B (1)" in text, text


def test_a_spent_wall_budget_stops_the_run(driver, monkeypatch, no_flight) -> None:
    d = driver(selection=["ep0"])
    monkeypatch.setattr(d, "worker_loop", lambda w: pytest.fail("flew past the budget"))
    d.max_wall_s, d._deadline = 1, time.monotonic() - 1
    assert d.run() == 0
    assert "wall budget reached" in d.driver_log.read_text() and d.summary_file.exists()


def test_a_second_run_into_a_held_root_is_refused(driver, tmp_path, monkeypatch) -> None:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\nfrom pathlib import Path\nfrom benchmarks.run_record import episodes\n"
            "lock = episodes.hold_root(Path(sys.argv[1]))\nprint('held', flush=True)\n"
            "time.sleep(60)\n",
            str(tmp_path),
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(rm, "build_manifest", lambda **kw: pytest.fail("flew"))
        d = driver(selection=["ep0"])
        assert d.run() == 2 and f"pid {holder.pid}" in d.driver_log.read_text()
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.parametrize("dry", [[], ["--dry-run"]])
def test_missing_block_assets_are_refused_before_discovery_and_any_out_root(
    dry, tmp_path, monkeypatch, capsys, bench_data
) -> None:
    from benchmarks.bench import BAKE_OUTPUT_SUFFIXES
    from maps.llm import endpoint_spec

    tiles, rgbd = tmp_path / "tiles", tmp_path / "rgbd"
    tiles.mkdir()
    rgbd.mkdir()
    for suffix in BAKE_OUTPUT_SUFFIXES:
        (tiles / f"blk{suffix}").write_text("{}")
    (rgbd / "blk.tif").write_bytes(b"")
    monkeypatch.setenv("SEARCH_CITYNAV_TILES", str(tiles))
    monkeypatch.setenv("SEARCH_CITYNAV_RGBD_DIR", str(rgbd))
    monkeypatch.setattr(bf, "_enumerate_split", lambda bench, split: ["blk::1#0"])
    monkeypatch.setattr(
        endpoint_spec, "discover_from_settings", lambda settings: pytest.fail("asked a serve")
    )
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("flew"))
    monkeypatch.setattr(bf.Driver, "dry_run", lambda self: pytest.fail("previewed"))
    assert bf.main(["--out-root", str(tmp_path / "r"), *dry]) == 2
    assert not (tmp_path / "r").exists()
    assert (
        f"  Missing {rgbd / 'blk_rgb.tif'} (built by benchmarks/citynav/data_prep/rasterize_citynav_native.sh)"
        in capsys.readouterr().err.splitlines()
    )
    (rgbd / "blk.png").write_bytes(b"")  # the CityNav raster pair
    assert bf._missing_block_assets(citynav_bench(os.environ), ["blk::1#0"]) == []


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
    "import os, time\n"
    "from benchmarks.citynav.run_benchmark.cli import _die_with_parent\n"
    "_die_with_parent(os.environ)\n"
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
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT], cwd=REPO, stdout=subprocess.PIPE, text=True
    )
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
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


_INTERRUPTED = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from benchmarks.citynav import bench_fullset as bf\nfrom benchmarks.run_record import run_manifest as rm\n"
    "d = bf.Driver.__new__(bf.Driver)\n"
    "d.bench = d.config = None\n"
    "root = Path(sys.argv[1])\n"
    "d.out_root, d.episodes_file, d.driver_log = root, root / 'episodes.jsonl', root / 'driver.log'\n"
    "d.summary_file, d.split, d.workers, d.max_wall_s, d._deadline = root / 'summary.txt', 's', 2, 0, None\n"
    "d._done_cache, d._infra_errors, d.max_infra_attempts, d._abort = set(), {}, 3, None\n"
    "d.sample_seed, d.limit, d.allowlist_path, d.memory_gib = 1, 0, None, 4.0\n"
    "d._selection, d._selection_note = ['ep0'], None\n"
    "bf.WORKER_STAGGER_S = 0\n"
    "rm.build_manifest = lambda **kw: None\n"
    "rm.emit_or_verify = lambda *a, **kw: None\n"
    "def fly(wid):\n"
    "    print('flying', flush=True)\n"
    "    time.sleep(60)\n"
    "d.worker_loop = fly\n"
    "sys.exit(d.run())\n"
)


@pytest.mark.parametrize(("sig", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_an_interrupted_driver_exits_with_its_workers(sig, code, tmp_path) -> None:
    """Ctrl-C, or a SIGTERM from kill, systemd or SLURM, releases the root, so a worker left
    flying would fly beside the next launch."""
    # A shell that ignores SIGINT hands that to its children; the driver is
    # launched with the default handler, as from a terminal.
    proc = subprocess.Popen(
        [sys.executable, "-c", _INTERRUPTED, str(tmp_path)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        flying = (line for line in iter(proc.stdout.readline, "") if line == "flying\n")
        assert next(flying) and next(flying)
        time.sleep(0.5)  # the main thread is joining its first worker
        proc.send_signal(sig)
        assert proc.wait(timeout=10) == code
        assert "Traceback" not in proc.stderr.read()
        assert f"interrupted ({sig.name})" in (tmp_path / "driver.log").read_text()
    finally:
        proc.kill()
        proc.wait()


def test_after_an_interrupt_no_worker_claims_another_episode(driver, monkeypatch) -> None:
    """Workers are daemon threads that run on until the process exits: once the driver
    is interrupted, the next claim is none, so no START lands without its END."""
    d = driver(selection=["ep0", "ep1"])

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(d, "_run", interrupted)
    assert d.run() == 130
    assert d._next_claimed_episode() is None


def test_the_split_defaults_to_the_benchmarks_own() -> None:
    assert bf._build_parser(citynav_bench({})).parse_args([]).split == "val_seen"
    assert bf._build_parser(calcity_bench({})).parse_args([]).split == "calcity"


def test_the_out_root_and_the_usage_name_the_benchmark() -> None:
    parser = bf._build_parser(calcity_bench({}))
    assert parser.parse_args([]).out_root == "output/calcity_fullset"
    assert parser.prog == "python -m benchmarks.calcity.bench_fullset"
    assert parser.description == "Continuous full-split calcity benchmark at fixed concurrency."
    assert bf._build_parser(citynav_bench({})).parse_args([]).out_root == "output/citynav_fullset"


def test_the_summary_names_an_errorless_infra_row_as_the_driver_counts_it(driver) -> None:
    d = driver()
    episodes.append_row(d.out_root, {"episode_id": "ep0", "worker_id": 0, "status": "infra"})
    # The driver's one infra predicate: a mission the serve ended counts too.
    episodes.append_row(
        d.out_root,
        {
            "episode_id": "ep1",
            "worker_id": 0,
            "status": "ok",
            "metrics": {"mission": {"ended_by": "transient_failures"}},
        },
    )
    d._write_summary()
    assert "infra attempts (excluded): 2  by cause: infra (2)" in d.summary_file.read_text()


def test_the_end_line_prints_no_score_for_an_infra_attempt(driver, monkeypatch) -> None:
    d = driver(selection=["a::1#0", "b::1#1"])
    rows = {
        "a": {**_infra(0), "metrics": {"NE": 3.0, "SR": 1}, "elapsed_s": 1.0},
        "b": {
            "episode_id": "b::1#1",
            "status": "ok",
            "metrics": {"NE": 3.0, "SR": 1},
            "elapsed_s": 1.0,
        },
    }
    monkeypatch.setattr(d, "_run_attempt", lambda w, eid: rows[eid[0]])
    d.run_one(0, "a::1#0")
    d.run_one(0, "b::1#1")
    ends = [ln for ln in d.driver_log.read_text().splitlines() if " END " in ln]
    assert ends[0].endswith("status=infra elapsed=1s"), ends
    assert ends[1].endswith("status=ok elapsed=1s NE=3.0 SR=1"), ends


def test_the_dry_run_counts_a_persistent_infra_episode_as_left_out(
    driver, tmp_path, monkeypatch, capsys
) -> None:
    for _ in range(3):
        episodes.append_row(tmp_path, _infra(0))
    d = driver(selection=["ep0", "ep1"])
    assert d.dry_run() == 0
    out = capsys.readouterr().out
    assert "pending: 1" in out and "[PEND] ep1" in out and "[PEND] ep0" not in out, out


def test_relative_paths_resolve_against_the_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bench = citynav_bench({})
    d = bf.Driver(
        bf._build_parser(bench).parse_args(
            ["--out-root", "runs/a", "--episode-allowlist", "ids.txt"]
        ),
        bench,
        bf._config_path(bench.environ),
    )
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


def test_an_offline_run_launches_with_nothing_served(driver, monkeypatch) -> None:
    """``SEARCH_ENDPOINT_OFFLINE=1`` pins a spec and serves nothing (an offline run): the
    gate launches without asking either server."""
    monkeypatch.setenv("SEARCH_ENDPOINT_OFFLINE", "1")
    d = driver(selection=["ep0"])
    monkeypatch.setattr(d, "_reasoner_answers", lambda: pytest.fail("probed the reasoner"))
    monkeypatch.setattr(d, "_sam_answers", lambda: pytest.fail("probed SAM"))
    flown: list[str] = []
    monkeypatch.setattr(d, "run_one", lambda wid, eid: (flown.append(eid), d._done_cache.add(eid)))
    d.worker_loop(0)
    assert flown == ["ep0"]


def test_the_sam_gate_asks_the_replicas_discovery_checked(driver, monkeypatch) -> None:
    """Any replica answering ``/health`` passes (the client steps off a dead one); none checked
    at discovery passes as discovery did; every replica down fails."""
    from dataclasses import replace

    from maps.mission.tests.fakes import TEST_SPEC

    asked: list[str] = []
    up: set[str] = set()

    class _Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _open(req, timeout):
        url = req.full_url
        asked.append(url)
        if url not in up:
            raise OSError("connection refused")
        return _Answer()

    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    d = driver()
    d.bench = replace(
        d.bench, endpoint_spec=replace(TEST_SPEC, sam_urls=("http://a:8100", "http://b:8100"))
    )
    assert not d._sam_answers()
    up.add("http://b:8100/health")
    assert d._sam_answers()
    assert asked == [
        "http://a:8100/health",
        "http://b:8100/health",
        "http://a:8100/health",
        "http://b:8100/health",
    ]


@pytest.mark.parametrize("flag", ["--workers", "--max-infra-attempts", "--episode-timeout-s"])
def test_the_parser_refuses_a_count_below_one(flag, capsys) -> None:
    with pytest.raises(SystemExit) as done:
        bf._build_parser(citynav_bench({})).parse_args([flag, "0"])
    assert done.value.code == 2 and flag in capsys.readouterr().err


def test_the_parser_refuses_an_attempt_memory_cap_that_is_no_cap(capsys) -> None:
    with pytest.raises(SystemExit) as done:
        bf._build_parser(citynav_bench({})).parse_args(["--attempt-memory-gib", "nan"])
    assert done.value.code == 2 and "at least 1 GiB" in capsys.readouterr().err


def test_an_allowlist_that_cannot_be_read_is_refused_fatal(driver, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bf, "_enumerate_split", lambda bench, split: ["a"])
    d = driver()
    d.allowlist_path = tmp_path / "absent.txt"
    with pytest.raises(ValueError, match="^--episode-allowlist read failed"):
        d._select_episodes()


@pytest.mark.parametrize("refusal", ["split", "allowlist", "partial", "empty"])
def test_a_refused_selection_exits_2_before_discovery_and_any_out_root(
    refusal, tmp_path, monkeypatch, capsys, bench_data
) -> None:
    """A bad selection is refused as the dry run refuses it: no serve asked, no root made, no lock taken."""
    from maps.llm import endpoint_spec

    monkeypatch.setattr(
        endpoint_spec,
        "discover_from_settings",
        lambda settings: pytest.fail("a refused selection asked a serve"),
    )
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("a refused selection flew"))
    if refusal != "split":
        monkeypatch.setattr(bf, "_enumerate_split", lambda bench, split: ["a", "b"])
    (tmp_path / "ids.txt").write_text("a\nz\n")
    (tmp_path / "empty.txt").write_text("\n")
    flags = {
        "split": ["--split", "nope"],
        "allowlist": ["--episode-allowlist", str(tmp_path / "absent.txt")],
        "partial": ["--episode-allowlist", str(tmp_path / "ids.txt")],
        "empty": ["--episode-allowlist", str(tmp_path / "empty.txt")],
    }[refusal]
    assert bf.main(["--out-root", str(tmp_path / "r"), *flags]) == 2
    assert not (tmp_path / "r").exists()
    assert capsys.readouterr().err.startswith("FATAL: ")


@pytest.mark.parametrize("dry", [[], ["--dry-run"]])
def test_a_refused_config_exits_2_before_any_out_root(dry, tmp_path, monkeypatch, capsys) -> None:
    bad = tmp_path / "run.yaml"
    bad.write_text(f"base: {REPO / 'configs/citynav.yaml'}\nbogus: 1\n")
    monkeypatch.setenv("CITYNAV_BENCH_CONFIG", str(bad))
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("a refused config flew"))
    assert bf.main(["--out-root", str(tmp_path / "r"), *dry]) == 2
    assert not (tmp_path / "r").exists()
    assert capsys.readouterr().err.startswith("FATAL: ")


def test_a_config_that_is_not_yaml_is_refused_naming_its_file(
    tmp_path, monkeypatch, capsys
) -> None:
    bad = tmp_path / "run.yaml"
    bad.write_text("max_steps: [1\n")
    monkeypatch.setenv("CITYNAV_BENCH_CONFIG", str(bad))
    assert bf.main(["--out-root", str(tmp_path / "r"), "--dry-run"]) == 2
    assert capsys.readouterr().err.startswith(f"FATAL: {bad}: ")


@pytest.mark.parametrize("ids", [["z"], ["a", "z"]])
def test_an_allowlist_naming_ids_off_the_split_is_refused_fatal(
    ids, driver, tmp_path, monkeypatch
) -> None:
    """Every allowlisted id must be in the split: one that is not is a wrong
    split or a stale list, and a run over the rest would read as complete."""
    monkeypatch.setattr(bf, "_enumerate_split", lambda bench, split: ["a", "b"])
    d = driver()
    d.allowlist_path = tmp_path / "ids.txt"
    d.allowlist_path.write_text("\n".join(ids) + "\n")
    with pytest.raises(
        ValueError,
        match="^--episode-allowlist names 1 ids not in the train_seen split, e.g. z",
    ):
        d._select_episodes()


@pytest.mark.parametrize("refusal", ["split", "allowlist", "partial", "empty"])
def test_a_refused_dry_run_exits_2_with_no_out_root(
    refusal, tmp_path, monkeypatch, capsys, bench_data
) -> None:
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("a dry run flew"))
    if refusal != "split":
        monkeypatch.setattr(bf, "_enumerate_split", lambda bench, split: ["a", "b"])
    (tmp_path / "ids.txt").write_text("a\nz\n")
    (tmp_path / "empty.txt").write_text("\n")
    flags = {
        "split": ["--split", "nope"],
        "allowlist": ["--episode-allowlist", str(tmp_path / "absent.txt")],
        "partial": ["--episode-allowlist", str(tmp_path / "ids.txt")],
        "empty": ["--episode-allowlist", str(tmp_path / "empty.txt")],
    }[refusal]
    assert bf.main(["--out-root", str(tmp_path / "r"), "--dry-run", *flags]) == 2
    assert not (tmp_path / "r").exists()
    assert capsys.readouterr().err.startswith("FATAL: ")


def test_a_dry_run_prints_the_allowlist_count_and_writes_nothing(
    tmp_path, monkeypatch, capsys, bench_data
) -> None:
    monkeypatch.setattr(bf, "_enumerate_split", lambda bench, split: ["a", "b"])
    monkeypatch.setattr(bf, "_missing_block_assets", lambda bench, eps: [])
    (tmp_path / "ids.txt").write_text("a\n")
    assert (
        bf.main(
            [
                "--out-root",
                str(tmp_path / "r"),
                "--dry-run",
                "--episode-allowlist",
                str(tmp_path / "ids.txt"),
            ]
        )
        == 0
    )
    assert not (tmp_path / "r").exists()
    assert "1 ids requested -> 1 match" in capsys.readouterr().out


def test_a_base_config_that_is_not_yaml_is_refused_naming_that_file(
    tmp_path, monkeypatch, capsys
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("max_steps: [1\n")
    top = tmp_path / "run.yaml"
    top.write_text("base: base.yaml\n")
    monkeypatch.setenv("CITYNAV_BENCH_CONFIG", str(top))
    assert bf.main(["--out-root", str(tmp_path / "r"), "--dry-run"]) == 2
    assert capsys.readouterr().err.startswith(f"FATAL: {base.resolve()}: ")


def test_each_worker_s_children_share_one_prompt_cache_key(driver, monkeypatch) -> None:
    """A worker's attempts carry SEARCH_LLM_CACHE_KEY = <out-root name>/w<worker>: one key per
    worker, so a hosted reasoner's cache serves the head every episode of that worker opens with."""
    envs: list[dict] = []

    class _Done:
        pid = 0
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def _popen(cmd, **kw):
        envs.append(kw["env"])
        return _Done()

    monkeypatch.setattr(bf.subprocess, "Popen", _popen)
    d = driver()
    for wid in (0, 3, 3):
        d._run_attempt(wid, "e::1#1")
    keys = [e["SEARCH_LLM_CACHE_KEY"] for e in envs]
    assert keys == [f"{d.out_root.name}/w0", f"{d.out_root.name}/w3", f"{d.out_root.name}/w3"]


def test_a_bench_without_its_data_exits_2_with_its_recipe_before_any_out_root(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("SEARCH_CITYNAV_DATA", str(tmp_path / "absent"))
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("flew without data"))
    assert bf.main(["--out-root", str(tmp_path / "r")]) == 2
    assert not (tmp_path / "r").exists()
    err = capsys.readouterr().err
    assert err.startswith("FATAL: CityNav data not found under ") and "setup.sh" in err
    assert "Traceback" not in err


def test_calcity_without_its_catalog_exits_2_with_its_recipe_before_any_out_root(
    tmp_path, monkeypatch, capsys
) -> None:
    from benchmarks.calcity import bench_fullset as calcity_driver

    monkeypatch.setenv("SEARCH_CALCITY_DATA", str(tmp_path / "absent"))
    monkeypatch.setattr(sys, "argv", ["bench_fullset", "--out-root", str(tmp_path / "r")])
    monkeypatch.setattr(bf.Driver, "run", lambda self: pytest.fail("flew without data"))
    assert calcity_driver.main() == 2
    assert not (tmp_path / "r").exists()
    err = capsys.readouterr().err
    assert err.startswith("FATAL: Cal City data is incomplete under ")
    assert "build_objects_json" in err and "objects.json" in err


def test_the_runner_run_directly_confines_itself_before_it_flies(tmp_path, monkeypatch) -> None:
    """``python -m benchmarks.citynav.run_benchmark`` runs model code in its own process: it
    takes the attempt's memory cap and, where bubblewrap runs, writes only its runs and the
    caches. A runner the driver spawns caps itself alone (the driver sandboxes it)."""
    from benchmarks.citynav.run_benchmark import cli

    confined: list = []
    monkeypatch.setattr(sandbox, "confine", lambda *a, **kw: confined.append((*a, kw)))
    monkeypatch.setattr(sandbox, "cap_memory", lambda gib: confined.append(gib))
    monkeypatch.setenv("SEARCH_CITYNAV_DATA", str(tmp_path / "absent"))
    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--out", str(tmp_path / "out")])
    assert cli.main() == 1
    ((writable, gib, _log, readable, kw),) = confined
    assert writable[0] == tmp_path / "out" and gib == sandbox.ATTEMPT_MEMORY_GIB
    assert kw == {"flies": True} and str(tmp_path / "absent") in map(str, readable)
    confined.clear()
    monkeypatch.setattr(
        sys, "argv", ["run_benchmark", "--attempt-root", str(tmp_path), "--episode-id", "e"]
    )
    monkeypatch.setattr(cli, "_die_with_parent", lambda environ: None)
    assert cli.main() == 1
    assert confined == [sandbox.ATTEMPT_MEMORY_GIB]


def test_the_cal_city_runner_run_directly_confines_itself(tmp_path, monkeypatch) -> None:
    """``python -m benchmarks.calcity.run_benchmark`` hands the shared main its bench and
    flies model code in its own process: it confines itself like the CityNav runner."""
    from benchmarks.calcity import run_benchmark

    confined: list = []
    monkeypatch.setattr(sandbox, "confine", lambda *a, **kw: confined.append(a))
    monkeypatch.setenv("SEARCH_CALCITY_DATA", str(tmp_path / "absent"))
    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--out", str(tmp_path / "out")])
    assert run_benchmark.main() == 1
    assert len(confined) == 1 and confined[0][0][0] == tmp_path / "out", confined


def test_the_runner_without_its_data_exits_1_with_the_recipe_alone(
    tmp_path, monkeypatch, capsys
) -> None:
    from benchmarks.citynav.run_benchmark import cli

    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--out", str(tmp_path / "r")])
    bench = citynav_bench({"SEARCH_CITYNAV_DATA": str(tmp_path / "absent")})
    assert cli.main(bench, confine=False) == 1
    err = capsys.readouterr().err
    assert err.startswith("FATAL: CityNav data not found under ") and "Traceback" not in err
