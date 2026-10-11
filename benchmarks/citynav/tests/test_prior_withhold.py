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


def test_the_prior_withhold_builds_no_prior_colour(monkeypatch, stub_session):
    """A flight under the withhold builds no prior colour enrichment: the
    enrichment reads the bake's prior features, not the episode graph."""
    import numpy as np

    from benchmarks.citynav.run_benchmark import mission_engine as me

    asked: list = []

    def _enrich(**kw):
        asked.append(kw["build_prior"])
        return {"prior": [], "prior_exc": None}

    monkeypatch.setattr(me, "_build_frame_enrich", _enrich)
    session = stub_session(rgb=np.zeros((4, 4, 3), np.uint8))
    monkeypatch.delenv("SEARCH_WITHHOLD", raising=False)
    session.fly({"target_fid": "building_7"})
    monkeypatch.setenv("SEARCH_WITHHOLD", "prior")
    session.fly({"target_fid": "building_7"})
    assert asked == [True, False]
