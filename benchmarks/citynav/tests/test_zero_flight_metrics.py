"""An episode that never flies still records what it spent on the LLM and how it ended."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.run_benchmark import episode_runner as er


def test_a_zero_flight_episode_records_its_llm_spend(monkeypatch, tmp_path) -> None:
    row = {"area": "b", "block": 1, "descriptions": ["the church"], "ann_ids": [0],
           "target_positions": [[5.0, 5.0, 0.0]], "trajectory": [[0.0, 0.0, 100.0]]}
    h = SimpleNamespace(
        state=SimpleNamespace(altitude_m=100.0, pose_xy=(0.0, 0.0), visited_xy=[]),
        search_loop=SimpleNamespace(_ortho_ground_z_m=0.0, set_retro_perceiver=lambda f: None),
        perception=SimpleNamespace(sahi_tiles_failed=0))
    monkeypatch.setattr(er, "load_episode_assets", lambda m: SimpleNamespace())
    monkeypatch.setattr(er, "build_episode", lambda cfg, ep, assets: h)
    monkeypatch.setattr(er, "MissionSession", lambda h, **kw: SimpleNamespace(flights=[]))
    monkeypatch.setattr(er, "run_mission_episode",
                        lambda s: SimpleNamespace(ended_by="reply"))
    monkeypatch.setattr(er, "score_episode", lambda *a, **k: {"SR": 0})
    monkeypatch.setattr(er.flight_summary, "summarize", lambda d: {"calls": 2})
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    cfg = SimpleNamespace(spawn="dataset", start_altitude_agl_m=None, altitude_cap_m=1000.0)

    metrics = er.run_episode(row, {"b_block_1": [object()]}, cfg)["metrics"]

    assert metrics["llm"] == {"calls": 2}
    assert metrics["mission"] == {"ended_by": "reply"}
