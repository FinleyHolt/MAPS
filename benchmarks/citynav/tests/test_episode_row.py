"""An episode's row, built through the real mission driver from a real ``MissionReply``: a flying episode
records its commit and final pose, a zero-flight one what it spent on the LLM and how it ended."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.run_benchmark import episode_runner as er
from maps.search.parse_compile.reasoner_loop import MissionReply

_ROW = {"area": "b", "block": 1, "descriptions": ["the church"], "ann_ids": [0],
        "target_positions": [[5.0, 5.0, 0.0]], "trajectory": [[0.0, 0.0, 100.0]]}


def _run(monkeypatch, tmp_path, flights: list[dict], row: dict = _ROW, snap: float = 0.0) -> dict:
    h = SimpleNamespace(
        episode=None,
        state=SimpleNamespace(altitude_m=100.0, pose_xy=(0.0, 0.0), visited_xy=[]),
        search_loop=SimpleNamespace(_ortho_ground_z_m=0.0, set_retro_perceiver=lambda f: None),
        perception=SimpleNamespace(sahi_tiles_failed=0))

    def build(cfg, ep, assets):
        h.episode = ep
        return h

    def session(h, **kw):
        return SimpleNamespace(h=h, flights=flights, feature_tour=[],
                               run=lambda text, transcript: (MissionReply(ended_by="reply", turns_used=3), None))

    def fly(*, state, **kw):
        # The six-action replay ends ``snap`` m east of the commit.
        x, y = state.confirmed_target_xy
        state.pose_xy = (x + snap, y)
        state.visited_xy = [(0.0, 0.0), state.pose_xy]
        return 7

    monkeypatch.setattr(er, "load_episode_assets", lambda m: SimpleNamespace())
    monkeypatch.setattr(er, "build_episode", build)
    monkeypatch.setattr(er, "MissionSession", session)
    monkeypatch.setattr(er, "synthesize_compliant_trajectory", fly)
    monkeypatch.setattr(er.flight_summary, "summarize", lambda d: {"calls": 2})
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    cfg = SimpleNamespace(spawn="dataset", start_altitude_agl_m=None, altitude_cap_m=1000.0, success_radius_m=20.0)
    return er.run_episode(row, {"b_block_1": [object()]}, cfg)


def test_a_flying_episode_records_its_commit_and_final_pose(monkeypatch, tmp_path) -> None:
    flights = [{"target": "prior_7", "resolved_xy": [5.0, 6.0], "provenance": "prior"}]
    row = _run(monkeypatch, tmp_path, flights)
    m = row["metrics"]
    assert row["visited_xy"][-1] == [5.0, 6.0]
    assert m["steps_taken"] == 7
    assert (m["committed_fid"], m["committed_provenance"]) == ("prior_7", "prior")
    assert m["mission"] == {"ended_by": "reply"}


def test_a_zero_flight_episode_records_its_llm_spend(monkeypatch, tmp_path) -> None:
    m = _run(monkeypatch, tmp_path, [])["metrics"]
    assert m["llm"] == {"calls": 2}
    assert m["mission"] == {"ended_by": "reply"}
    assert m["steps_taken"] == 0 and m["SR"] == 0


def test_a_shape_row_scores_the_commit_not_the_snapped_final_pose(monkeypatch, tmp_path) -> None:
    """A commit inside a 2 m strip whose replay ends 3 m east of it, outside the strip, scores SR 1."""
    from benchmarks.citynav import paths

    monkeypatch.setattr(paths, "SUCCESS_CRITERION", "shape")
    strip = [[4.0, 5.0], [6.0, 5.0], [6.0, 7.0], [4.0, 7.0]]
    flights = [{"target": "prior_7", "resolved_xy": [5.0, 6.0], "provenance": "prior"}]
    row = _run(monkeypatch, tmp_path, flights, row={**_ROW, "target_mask": strip}, snap=3.0)
    assert row["visited_xy"][-1] == [8.0, 6.0]
    assert row["metrics"]["SR"] == 1
