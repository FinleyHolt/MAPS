"""An episode's row, built through the real mission driver from a real ``MissionReply``: a flying episode
records its commit and final pose, a zero-flight one what it spent on the LLM and how it ended."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.adapter import parse_episode
from benchmarks.citynav.run_benchmark import episode_runner as er
from maps.search.parse_compile.reasoner_loop import MissionReply

_ROW = {"area": "b", "block": 1, "descriptions": ["the church"], "ann_ids": [0],
        "target_positions": [[5.0, 5.0, 0.0]], "trajectory": [[0.0, 0.0, 100.0]]}


def _run(monkeypatch, tmp_path, flights: list[dict], row: dict = _ROW, snap: float = 0.0,
         policy: str | None = None) -> dict:
    h = SimpleNamespace(
        episode=None,
        state=SimpleNamespace(altitude_m=100.0, pose_xy=(0.0, 0.0)),
        ground_z_m=0.0, search_loop=SimpleNamespace(set_retro_perceiver=lambda f: None),
        perception=SimpleNamespace(sahi_tiles_failed=0))

    def build(cfg, ep, assets):
        h.episode = ep
        return h

    def session(h, **kw):
        return SimpleNamespace(h=h, flights=flights,
                               run=lambda text, transcript: (MissionReply(ended_by="reply", turns_used=3), None))

    def fly(*, flights, **kw):
        # Seven actions, the last ending ``snap`` m east of the commit.
        x, y = flights[-1]["resolved_xy"]
        return [(0.0, 0.0)] * 6 + [(x + snap, y)]

    monkeypatch.setattr(er, "load_episode_assets", lambda m: SimpleNamespace())
    monkeypatch.setattr(er, "build_episode", build)
    monkeypatch.setattr(er, "MissionSession", session)
    monkeypatch.setattr(er, "synthesize_compliant_trajectory", fly)
    monkeypatch.setattr(er.flight_summary, "summarize", lambda d: {"calls": 2})
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    cfg = SimpleNamespace(spawn="dataset", start_altitude_agl_m=None, altitude_cap_m=1000.0, success_radius_m=20.0,
                          policy=policy)
    return er.run_episode(parse_episode(row), {"b_block_1": [object()]}, cfg)


def test_a_flying_episode_records_its_commit_and_final_pose(monkeypatch, tmp_path) -> None:
    flights = [{"target": "prior_7", "resolved_xy": [5.0, 6.0], "provenance": "prior"}]
    row = _run(monkeypatch, tmp_path, flights)
    m = row["metrics"]
    assert row["visited_xy"][-1] == [5.0, 6.0]
    assert row["commit_xy"] == [5.0, 6.0] and row["target_xy"] == [5.0, 5.0]
    assert m["steps_taken"] == 7
    assert m["committed_provenance"] == "prior"
    assert m["mission"] == {"ended_by": "reply"}


def test_a_zero_flight_episode_records_its_llm_spend(monkeypatch, tmp_path) -> None:
    row = _run(monkeypatch, tmp_path, [])
    m = row["metrics"]
    assert row["commit_xy"] is None
    assert m["llm"] == {"calls": 2}
    assert m["mission"] == {"ended_by": "reply"}
    assert m["steps_taken"] == 0 and m["SR"] == 0


def test_a_policy_config_flies_the_policy_through_the_same_replay_and_row(monkeypatch, tmp_path) -> None:
    """``policy:`` hands the built episode to ``baselines.seam.fly`` by name in
    place of the mission; its flights are replayed and scored, its fields ride
    the row, and no mission block is written."""
    import sys

    flights = [{"target": "stub_stop_0", "resolved_xy": [2.0, 2.0], "provenance": "point"},
               {"target": "mark_3", "resolved_xy": [5.0, 6.0], "provenance": "point"}]
    seen = {}

    def fly(h, policy):
        seen["policy"] = policy
        return SimpleNamespace(flights=flights, metrics={"policy": policy, "commit_path": "stub:tier", "stub_stops": 1})

    monkeypatch.setitem(sys.modules, "baselines.seam", SimpleNamespace(fly=fly))
    row = _run(monkeypatch, tmp_path, [], policy="stub")
    m = row["metrics"]
    assert seen["policy"] == "stub"
    assert row["commit_xy"] == [5.0, 6.0] and row["visited_xy"][-1] == [5.0, 6.0]
    assert (m["policy"], m["commit_path"], m["stub_stops"]) == ("stub", "stub:tier", 1)
    assert m["SR"] == 1 and m["steps_taken"] == 7
    assert "mission" not in m


def test_a_shape_row_scores_the_commit_not_the_snapped_final_pose(monkeypatch, tmp_path) -> None:
    """A commit inside a 2 m strip whose replay ends 3 m east of it, outside the strip, scores SR 1."""
    from benchmarks.citynav import paths

    monkeypatch.setattr(paths, "SUCCESS_CRITERION", "shape")
    strip = [[4.0, 5.0], [6.0, 5.0], [6.0, 7.0], [4.0, 7.0]]
    flights = [{"target": "prior_7", "resolved_xy": [5.0, 6.0], "provenance": "prior"}]
    row = _run(monkeypatch, tmp_path, flights, row={**_ROW, "target_mask": strip}, snap=3.0)
    assert row["visited_xy"][-1] == [8.0, 6.0]
    assert row["commit_xy"] == [5.0, 6.0]
    assert row["metrics"]["SR"] == 1


def test_the_bench_resolves_its_sensor_as_the_flight_does(monkeypatch) -> None:
    """``sensor: {}`` is the default sensor's override, never no sensor."""
    from maps.perception.geometry.sensor_spec import resolve_sensor

    loop = SimpleNamespace(start=lambda g: None)
    monkeypatch.setattr(er, "_episode_graph", lambda stem, assets: None)
    monkeypatch.setattr(er, "start_altitude_m", lambda cfg, m, z: (100.0, 0.0))
    monkeypatch.setattr(er, "make_perception", lambda: SimpleNamespace(set_concepts=None))
    monkeypatch.setattr(er, "SearchLoop", lambda *a, **kw: loop)
    assets = SimpleNamespace(grid_shape=(2, 2), cell_size_m=1.0, map_name="b_block_1",
                             bounds=SimpleNamespace(x_min=0.0, y_min=0.0))
    episode = SimpleNamespace(map_name="b_block_1", start_z=100.0, start_xy=(0.0, 0.0))
    h = er.build_episode(SimpleNamespace(sensor={}), episode, assets)
    assert h.sensor == resolve_sensor({})
