"""The ``prior`` withhold starts the episode graph with no prior features."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.run_benchmark import episode_runner


def test_the_prior_withhold_starts_an_empty_graph(monkeypatch):
    feats = {"prior": [{"cells": [[0, 0], [0, 1]], "name": "Mill Lane", "class": "road"}]}
    assets = SimpleNamespace(base_features=feats, cell_size_m=2.0)
    monkeypatch.setattr(episode_runner, "load_graph",
                        lambda stem: (_ for _ in ()).throw(FileNotFoundError(stem)))
    monkeypatch.delenv("SEARCH_WITHHOLD", raising=False)
    assert len(episode_runner._episode_graph("blk", assets).nodes) == 1
    monkeypatch.setenv("SEARCH_WITHHOLD", "prior")
    assert len(episode_runner._episode_graph("blk", assets).nodes) == 0


def test_the_prior_withhold_builds_no_prior_colour(monkeypatch):
    """A flight under the withhold builds no prior colour enrichment: the
    enrichment reads the bake's prior features, not the episode graph."""
    import numpy as np

    from benchmarks.citynav.episode import EpisodeState
    from benchmarks.citynav.run_benchmark import mission_engine as me
    from maps.search.parse_compile.tour_state import FeatureTour

    asked: list = []

    def _enrich(**kw):
        asked.append(kw["build_prior"])
        return {"prior": [], "prior_exc": None}

    monkeypatch.setattr(me, "_build_frame_enrich", _enrich)
    monkeypatch.setattr(me, "_write_tour_cursor", lambda *a, **k: None)
    loop = SimpleNamespace(
        activate_pending_boosts=lambda: None, update_perception=lambda **kw: [],
        snapshot=lambda: None, retire_boosts_after_round=lambda rounds: None,
        last_flight_detections=lambda: [], upsert_prior_percept=lambda *a, **k: None)
    frame = SimpleNamespace(pose_xy=(1.0, 2.0), altitude_m=900.0, yaw_deg=0.0,
                            rgb=np.zeros((4, 4, 3), np.uint8), sensor=None,
                            ortho_view_corners_xy=None)
    common = dict(
        flights=[], feature_tour=FeatureTour(), node_xy=lambda fid: ((1.0, 2.0), "prior"),
        marker_xy=None, cfg=SimpleNamespace(altitude_cap_m=1000.0),
        state=EpisodeState(pose_xy=(0.0, 0.0), altitude_m=100.0, graph=None),
        assets=SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0), cell_size_m=2.0,
                               grid_shape=(10, 10)),
        episode=SimpleNamespace(map_name="calcity", description="d"),
        sensor=None, tile_frame=None, perception=None, search_loop=loop,
        recorder=None, t_start=0.0, observe=lambda: frame)
    monkeypatch.delenv("SEARCH_WITHHOLD", raising=False)
    me.fly_one({"target_fid": "building_7"}, **common)
    monkeypatch.setenv("SEARCH_WITHHOLD", "prior")
    me.fly_one({"target_fid": "building_7"}, **common)
    assert asked == [True, False]
