"""The ``prior`` withhold starts the episode graph with no prior features."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.run_benchmark import episode_runner


def test_the_prior_withhold_starts_an_empty_graph(monkeypatch):
    feats = {"prior": [{"cells": [[0, 0], [0, 1]], "name": "Mill Lane", "class": "road"}]}
    assets = SimpleNamespace(base_features=feats, cell_size_m=2.0)
    monkeypatch.setattr(episode_runner, "load_episode_graph",
                        lambda stem: (_ for _ in ()).throw(FileNotFoundError(stem)))
    monkeypatch.delenv("SEARCH_WITHHOLD", raising=False)
    graph, prior = episode_runner._episode_graph("blk", assets)
    assert prior is feats and len(graph.nodes) == 1
    monkeypatch.setenv("SEARCH_WITHHOLD", "prior")
    graph, prior = episode_runner._episode_graph("blk", assets)
    assert prior == {} and len(graph.nodes) == 0
