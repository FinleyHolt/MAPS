from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import pytest

from benchmarks.citynav import bench_fullset as bf


@pytest.fixture
def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Driver holding only the fields the dispatch, attempt, result and
    summary paths read; ``__init__`` resolves a split, a config and a serve."""
    monkeypatch.setattr(bf, "WORKER_STAGGER_S", 0)

    def make(root: Path = tmp_path, selection: list[str] | None = None) -> bf.Driver:
        d = bf.Driver.__new__(bf.Driver)
        d.out_root = root
        d.episodes_file = root / "episodes.jsonl"
        d.driver_log = root / "driver.log"
        d.summary_file = root / "summary.txt"
        d._dispatch_lock = threading.Lock()
        d._done_cache = set()
        d._in_flight = set()
        d._recent = deque(maxlen=bf.ABORT_WINDOW)
        d._abort = None
        d._deadline = None
        d._selection = [f"ep{i}" for i in range(500)] if selection is None else list(selection)
        d._cursor = 0
        d.split = "train_seen"
        d.workers = 2
        d.episode_timeout_s = 3600
        d.max_wall_s = 0
        d.sample_seed = 1234
        d.limit = 0
        d.allowlist_path = None
        d.max_infra_attempts = 3
        d._infra_errors = {}
        return d

    return make


#: The stub world's fids: ``fid -> ((x, y), provenance)``.
STUB_GRAPH = {
    "building_7": ((100.0, 200.0), "prior"),
    "detected_3": ((55.5, 66.25), "detected"),
}


@pytest.fixture
def stub_session(monkeypatch: pytest.MonkeyPatch):
    """A ``MissionSession`` over a SearchLoop holding ``STUB_GRAPH``, a live
    ``observe`` frame and no render: ``make(**loop_overrides,
    pose_xy=, rgb=)``."""
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark.episode_runner import EpisodeState
    from benchmarks.citynav.run_benchmark import mission_engine as me
    from maps.search.tests.fakes import search_loop

    monkeypatch.setattr(me, "_write_tour_cursor", lambda *a, **k: None)

    def make(*, pose_xy=(10.0, 20.0), rgb=None, **loop_overrides) -> me.MissionSession:
        loop = search_loop({
            fid: SimpleNamespace(world_xy=xy, provenance="detected" if prov == "detected" else "prebaked")
            for fid, (xy, prov) in STUB_GRAPH.items()})
        for k, v in loop_overrides.items():
            setattr(loop, k, v)
        frame = SimpleNamespace(pose_xy=(130.0, 240.0), altitude_m=900.0, yaw_deg=45.0, rgb=rgb,
                                sensor=None, ortho_view_corners_xy=None)
        h = SimpleNamespace(
            state=EpisodeState(pose_xy=pose_xy, altitude_m=100.0),
            assets=SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0), cell_size_m=2.0,
                                   grid_shape=(10, 10)),
            episode=SimpleNamespace(map_name="calcity", description="the largest building"),
            sensor=None, tile_frame=None, search_loop=loop,
            perception=SimpleNamespace(detect=lambda obs: [], concept_provider=SimpleNamespace(concepts=[])))
        return me.MissionSession(h, observe=lambda: frame)

    return make
