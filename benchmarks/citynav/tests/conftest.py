from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import pytest

from benchmarks.citynav import bench_fullset as bf


@pytest.fixture
def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a Driver holding only the fields the dispatch, result and
    summary paths read. ``__init__`` resolves a split, a config and a
    serve, none of which these paths touch."""
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
        d._t_start = time.monotonic()
        return d

    return make
