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


def test_an_exception_lands_an_infra_row(tmp_path, light, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dispatch, "run_episode", boom)
    row, failed_oom = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"] == "RuntimeError: boom"
    assert row["withhold"] == ["look"]
    assert failed_oom is False, "an unclassified exception must not respawn the runner"
    assert episodes.read_episode_json(tmp_path) == row


def test_a_classified_infra_exception_carries_its_token(tmp_path, light, monkeypatch):
    exc = RuntimeError("CUDA out of memory")
    assert dispatch._exc_is_cuda_oom(exc), "vacuity floor: the fixture must classify"

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(dispatch, "run_episode", boom)
    row, failed_oom = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"].startswith("CITYNAV_EPISODE_OOM_ABORT: RuntimeError")
    assert failed_oom is True


def test_no_record_is_infra_not_a_scored_zero(tmp_path, light, monkeypatch):
    monkeypatch.setattr(dispatch, "run_episode", lambda *a, **k: None)
    row, _ = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"] == "no_prediction_unclassified"
    assert episodes.read_episode_json(tmp_path) == row


def test_a_serve_death_under_the_mission_is_infra(tmp_path, light, monkeypatch):
    monkeypatch.setattr(dispatch, "run_episode", lambda *a, **k: _rec("transient_failures"))
    row, _ = _dispatch(tmp_path)
    assert row["status"] == episodes.STATUS_INFRA
    assert row["error"] == "mission ended_by=transient_failures"
    assert row["metrics"]["SR"] == 1, "the record stays on the row for the audit"
    assert episodes.read_episode_json(tmp_path) == row


def test_a_scored_record_is_ok_and_on_disk(tmp_path, light, monkeypatch):
    monkeypatch.setattr(dispatch, "run_episode", lambda *a, **k: _rec())
    per_episode: list[dict] = []
    row, failed_oom = _dispatch(tmp_path, per_episode)
    assert row["status"] == episodes.STATUS_OK
    assert row["error"] is None
    assert failed_oom is False
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
