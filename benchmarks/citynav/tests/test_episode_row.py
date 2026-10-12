"""An episode's row, built through the real mission driver from a real ``MissionReply``: a flying episode
records its commit and final pose, a zero-flight one what it spent on the LLM and how it ended."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.run_benchmark import episode_runner as er
from maps.search.parse_compile.reasoner_loop import MissionReply

_ROW = {"area": "b", "block": 1, "descriptions": ["the church"], "ann_ids": [0],
        "target_positions": [[5.0, 5.0, 0.0]], "trajectory": [[0.0, 0.0, 100.0]]}


def _run(monkeypatch, tmp_path, flights: list[dict]) -> dict:
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
        state.pose_xy = state.confirmed_target_xy
        state.visited_xy = [(0.0, 0.0), state.confirmed_target_xy]
        return 7

    monkeypatch.setattr(er, "load_episode_assets", lambda m: SimpleNamespace())
    monkeypatch.setattr(er, "build_episode", build)
    monkeypatch.setattr(er, "MissionSession", session)
    monkeypatch.setattr(er, "synthesize_compliant_trajectory", fly)
    monkeypatch.setattr(er, "score_episode", lambda *a, **k: {"SR": int(k.get("flew", True))})
    monkeypatch.setattr(er.flight_summary, "summarize", lambda d: {"calls": 2})
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    cfg = SimpleNamespace(spawn="dataset", start_altitude_agl_m=None, altitude_cap_m=1000.0)
    return er.run_episode(_ROW, {"b_block_1": [object()]}, cfg)


def test_a_flying_episode_records_its_commit_and_final_pose(monkeypatch, tmp_path) -> None:
    flights = [{"target": "prior_7", "resolved_xy": [5.0, 6.0], "provenance": "prior"}]
    row = _run(monkeypatch, tmp_path, flights)
    m = row["metrics"]
    assert row["final_pos_xy"] == [5.0, 6.0]
    assert m["steps_taken"] == 7
    assert (m["committed_fid"], m["committed_provenance"], m["committed_xy"]) == ("prior_7", "prior", [5.0, 6.0])
    assert m["commit_would_score"] is True
    assert m["mission"] == {"ended_by": "reply"}


def test_a_zero_flight_episode_records_its_llm_spend(monkeypatch, tmp_path) -> None:
    m = _run(monkeypatch, tmp_path, [])["metrics"]
    assert m["llm"] == {"calls": 2}
    assert m["mission"] == {"ended_by": "reply"}
    assert m["steps_taken"] == 0 and m["committed_xy"] is None
