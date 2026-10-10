"""``_dispatch_one`` writes a verdict, and the verdict is not always ``ok``.

A ``status`` the producer only ever set to ``ok`` would pass every schema
and contract test while the reader's ``infra`` branch never turned over.
Each non-``ok`` producer path is driven here for real: the episode
raising, the runner landing no record, and the mission ending
``transient_failures``. ``run_episode`` is the only thing stubbed.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from benchmarks.citynav import episodes
from benchmarks.citynav.run_benchmark import dispatch


@pytest.fixture
def light(monkeypatch):
    """Keep the dispatch off the split file and the git subprocess."""
    monkeypatch.setattr(
        dispatch, "parse_episode",
        lambda row, row_index=0: SimpleNamespace(episode_id="b::1#0"),
    )
    monkeypatch.setattr(dispatch, "arm_env_stamp", lambda: {"SEARCH_X": "1"})
    monkeypatch.setattr(dispatch, "_cached_run_provenance", lambda flags: {"commit": "abc"})


def _dispatch(tmp_path, per_episode=None):
    return dispatch._dispatch_one(
        row={}, orig_idx=0, idx=0, total=1, compiler=None, objects_by_map={},
        cfg=SimpleNamespace(withhold=["look"]), recorder=None, episode_dir=tmp_path,
        per_episode=per_episode if per_episode is not None else [],
    )


def _rec(ended_by: str = "reply") -> dict:
    return {
        "episode_id": "b::1#0", "map_name": "b", "pred_xy": [1.0, 2.0],
        "metrics": {"NE": 3.0, "SR": 1, "steps_taken": 2, "compile_s": 0.5,
                    "mission": {"ended_by": ended_by}},
    }


@pytest.mark.parametrize("msg", ["boom", "CUDA out of memory"])
def test_a_method_exception_lands_a_scored_miss(tmp_path, light, monkeypatch, msg):
    """An exception the method raised is its own failure: the row is
    ``ok`` with the error and traceback and no metrics, which every reader
    scores SR 0. Only an infrastructure fault re-queues."""
    def boom(*a, **k):
        raise RuntimeError(msg)

    monkeypatch.setattr(dispatch, "run_episode", boom)
    row = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_OK
    assert row["error"] == f"RuntimeError: {msg}"
    assert "in boom" in row["traceback"] and msg in row["traceback"]
    assert "metrics" not in row
    assert row["withhold"] == ["look"]
    assert episodes.read_episode_json(tmp_path) == row


def _status_error(code: int) -> Exception:
    import httpx

    req = httpx.Request("POST", "http://reasoner/v1/chat/completions")
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, request=req))


def test_a_serve_fault_carries_the_infra_token(tmp_path, light, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("compile failed") from _status_error(503)

    monkeypatch.setattr(dispatch, "run_episode", boom)
    row = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"].startswith("CITYNAV_EPISODE_INFRA_ABORT: RuntimeError")


@pytest.mark.parametrize("exc", [
    _status_error(400),
    RuntimeError("CUDA out of memory"),
    RuntimeError("upstream said: Connection refused"),
])
def test_a_refused_request_or_a_quoted_phrase_is_not_infra(exc):
    from benchmarks.citynav.run_benchmark.error_taxonomy import _exc_is_transient_infra

    assert _exc_is_transient_infra(exc) is False


@pytest.mark.parametrize("code", [408, 429, 500, 503])
def test_a_status_the_serve_may_clear_is_infra(code):
    from benchmarks.citynav.run_benchmark.error_taxonomy import _exc_is_transient_infra

    assert _exc_is_transient_infra(_status_error(code)) is True


def test_no_record_is_infra_not_a_scored_zero(tmp_path, light, monkeypatch):
    monkeypatch.setattr(dispatch, "run_episode", lambda *a, **k: None)
    row = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"] == "no_prediction_unclassified"
    assert episodes.read_episode_json(tmp_path) == row


def test_a_serve_death_under_the_mission_is_infra(tmp_path, light, monkeypatch):
    monkeypatch.setattr(dispatch, "run_episode", lambda *a, **k: _rec("transient_failures"))
    row = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"] == "mission ended_by=transient_failures"
    assert row["metrics"]["SR"] == 1, "the record stays on the row for the audit"
    assert episodes.read_episode_json(tmp_path) == row


def test_a_scored_record_is_ok_and_on_disk(tmp_path, light, monkeypatch):
    monkeypatch.setattr(dispatch, "run_episode", lambda *a, **k: _rec())
    per_episode: list[dict] = []
    row = _dispatch(tmp_path, per_episode)
    assert row["status"] == episodes.STATUS_OK
    assert row["error"] is None
    assert row["search_env_flags"] == {"SEARCH_X": "1"}
    assert row["provenance"] == {"commit": "abc"}
    assert isinstance(row["elapsed_s"], float) and isinstance(row["ts"], float)
    assert per_episode == [row["metrics"]]
    assert episodes.read_episode_json(tmp_path) == row


def test_append_row_from_two_threads_writes_intact_lines(tmp_path):
    n = 200

    def writer(tag: str) -> None:
        for i in range(n):
            episodes.append_row(
                tmp_path, {"episode_id": f"{tag}{i}", "status": "ok", "pad": "x" * 500}
            )

    threads = [threading.Thread(target=writer, args=(t,)) for t in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [
        json.loads(ln)
        for ln in (tmp_path / episodes.EPISODES_JSONL).read_text().splitlines()
    ]
    assert len(rows) == 2 * n
    assert {r["episode_id"] for r in rows} == {f"{t}{i}" for t in "ab" for i in range(n)}


def test_rows_are_strict_json(tmp_path):
    """A sensor's unbounded range is ``inf`` in Python and nothing in JSON;
    the rows other people parse must not carry Python's dialect."""
    import json

    from benchmarks.citynav import episodes

    row = {"episode_id": "e", "status": "ok", "sensor": {"max_range_m": float("inf")},
           "metrics": {"NE": float("nan"), "SR": 1}, "visited_xy": [[1.0, float("-inf")]]}
    episodes.append_row(tmp_path, row)
    episodes.write_episode_json(tmp_path, row)

    def refuse(name):
        raise AssertionError(f"non-JSON constant {name} on disk")

    for text in ((tmp_path / "episodes.jsonl").read_text(), (tmp_path / "episode.json").read_text()):
        parsed = json.loads(text, parse_constant=refuse)
        assert parsed["sensor"]["max_range_m"] is None
        assert parsed["metrics"] == {"NE": None, "SR": 1}
        assert parsed["visited_xy"] == [[1.0, None]]


def test_a_cut_last_line_loses_only_its_own_row(tmp_path):
    """A worker killed mid-append leaves a cut last line; the next append
    starts a line of its own, so only the cut row is lost."""
    from benchmarks.citynav.analysis import rows

    episodes.append_row(tmp_path, {"episode_id": "A", "status": "ok"})
    cut = json.dumps({"episode_id": "B", "status": "ok"})
    with (tmp_path / episodes.EPISODES_JSONL).open("a") as fh:
        fh.write(cut[: len(cut) // 2])
    episodes.append_row(tmp_path, {"episode_id": "C", "status": "ok"})
    got = [r["episode_id"] for r in rows.iter_json_rows(tmp_path / episodes.EPISODES_JSONL)]
    assert got == ["A", "C"]
