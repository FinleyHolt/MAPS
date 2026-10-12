"""The bench driver: one child per attempt, its row, the claims, the stops.

The children are ``fake_runner``, whose behaviour is the episode id's map.
"""

from __future__ import annotations

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

REPO = Path(bf.__file__).resolve().parents[2]

# map -> (status, error, has a workdir)
OUTCOMES = {
    "ok": ("ok", None, True),
    "miss": ("ok", "RuntimeError: boom", True),
    "infra": ("infra", "CITYNAV_EPISODE_INFRA_ABORT: SamServerOutage", True),
    "sleep": ("timed_out", None, True),
    "kill": ("infra", "killed_midrun", True),
    "norow": ("infra", "no_prediction_unclassified", True),
    "vllm": ("infra", "vllm_down_midrun", True),
    "nodir": ("infra", "killed_midrun", False),
}


def _infra(i: int, cause: str = "no_prediction_unclassified", **extra) -> dict:
    return {"episode_id": f"ep{i}", "worker_id": 0, "status": "infra", "error": cause, **extra}


def test_each_child_outcome_lands_its_row(driver, monkeypatch) -> None:
    monkeypatch.setattr(bf, "RUNNER_MODULE", "benchmarks.citynav.tests.fake_runner")
    monkeypatch.setattr(bf, "VLLM_PROBE_S", 0.2)
    d = driver(selection=[f"{k}::1#{i}" for i, k in enumerate(OUTCOMES)])
    d.episode_timeout_s, d.max_infra_attempts = 3, 1
    # The reasoner is down exactly while the "vllm" episode flies.
    monkeypatch.setattr(d, "_wait_for_vllm", lambda: None)
    monkeypatch.setattr(d, "_vllm_healthy",
                        lambda: not any(e.startswith("vllm") for e in d._in_flight))
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
    """bedrock-mantle serves GPT-5.6 under /openai/v1 and lists it under /v1/models: a gate
    asking SEARCH_REASONER_URL/models would hold every launch on a reasoner discovery accepted."""
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
    monkeypatch.setenv("SEARCH_REASONER_URL", "https://bedrock-mantle.us-east-1.api.aws/openai/v1")
    monkeypatch.setenv("SEARCH_REASONER_MODELS_URL", "https://bedrock-mantle.us-east-1.api.aws/v1/models")
    assert driver()._vllm_healthy()
    monkeypatch.delenv("SEARCH_REASONER_MODELS_URL")
    monkeypatch.setenv("SEARCH_REASONER_URL", "http://127.0.0.1:8001/v1")
    assert driver()._vllm_healthy()
    assert asked == ["https://bedrock-mantle.us-east-1.api.aws/v1/models", "http://127.0.0.1:8001/v1/models"]


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
        ledger_row("g::2#2", "20260725_110100", status="infra", error="vllm_down_midrun"),
    )))
    d._write_summary()
    out = d.summary_file.read_text()
    assert "infra attempts (excluded): 1  by cause: vllm_down_midrun:1" in out
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
        assert "persistent infra: 1 episode(s)" in text and "A=2, B=1" in text, text


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
    monkeypatch.setattr(d, "_wait_for_vllm", lambda: None)
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


def test_the_split_defaults_to_the_benchmarks_own() -> None:
    from benchmarks.citynav import paths

    assert bf._build_parser().parse_args([]).split == paths.DEFAULT_SPLIT


def test_the_out_root_defaults_to_the_benchmarks_own(monkeypatch) -> None:
    monkeypatch.setattr(bf.paths, "BENCHMARK", "calcity")
    assert bf._build_parser().parse_args([]).out_root == "output/calcity_fullset"


def test_the_summary_names_an_errorless_infra_row_as_the_driver_counts_it(driver) -> None:
    d = driver()
    episodes.append_row(d.out_root, {"episode_id": "ep0", "worker_id": 0, "status": "infra"})
    d._write_summary()
    assert "by cause: infra:1" in d.summary_file.read_text()


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
