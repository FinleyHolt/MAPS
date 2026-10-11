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
    """A ``MissionSession`` over a stubbed SearchLoop (``STUB_GRAPH``), a live
    ``observe`` frame and no render: ``make(**loop_overrides, markers=,
    pose_xy=, rgb=)``."""
    from types import SimpleNamespace

    from benchmarks.citynav.episode import EpisodeState
    from benchmarks.citynav.run_benchmark import mission_engine as me

    monkeypatch.setattr(me, "_write_tour_cursor", lambda *a, **k: None)

    def make(*, markers=(), pose_xy=(10.0, 20.0), rgb=None, **loop_overrides) -> me.MissionSession:
        loop = SimpleNamespace(
            declare_mission_concepts=lambda cleaned: {}, _lock=threading.RLock(),
            _tool_ctx_for_examiner=SimpleNamespace, _ortho_ground_z_m=None,
            _render_candidate_crops=None, _look_audit_dir=lambda: None, _boost_queue_cb=None,
            world_xy_for_node=lambda fid: STUB_GRAPH[fid][0] if fid in STUB_GRAPH else None,
            _fid_is_detected=lambda fid: STUB_GRAPH[fid][1] == "detected",
            activate_pending_boosts=lambda: None, update_perception=lambda **kw: [],
            retire_boosts_after_round=lambda rounds: None,
            last_flight_detections=lambda: [], upsert_prior_percept=lambda *a, **k: None,
        )
        for k, v in loop_overrides.items():
            setattr(loop, k, v)
        frame = SimpleNamespace(pose_xy=(130.0, 240.0), altitude_m=900.0, yaw_deg=45.0, rgb=rgb,
                                sensor=None, ortho_view_corners_xy=None)
        h = SimpleNamespace(
            cfg=SimpleNamespace(altitude_cap_m=1000.0),
            state=EpisodeState(pose_xy=pose_xy, altitude_m=100.0, graph=None),
            assets=SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0), cell_size_m=2.0,
                                   grid_shape=(10, 10)),
            episode=SimpleNamespace(map_name="calcity", description="the largest building",
                                    marker_positions=list(markers)),
            sensor=None, tile_frame=None, perception=None, search_loop=loop)
        return me.MissionSession(h, t_start=0.0, observe=lambda: frame)

    return make
