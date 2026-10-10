"""summary.txt scores the attempt the scheduler claimed.

``_write_summary`` deduped by episode_id keeping the last row, which is
COMPLETION order -- so on a run with replicates it reported whichever
attempt happened to finish last. On
citynav_train_seen_2026-07-15_12-45-54 that read SR_eff 2/10 where the
canonical reader read 3/10.

Driven against a stub rather than a real Driver: constructing a Driver
wants a split, a GPU and a live vLLM, none of which is what broke.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from benchmarks.citynav.bench_fullset import Driver


class _Summary:
    """The surface ``_write_summary`` touches, and nothing else."""

    _write_summary = Driver._write_summary
    _persistent_infra = Driver._persistent_infra
    _infra_attempts = Driver._infra_attempts

    def __init__(self, tmp_path: Path, *, split_size: int = 10):
        self.out_root = tmp_path
        self.episodes_file = tmp_path / "episodes.jsonl"
        self.summary_file = tmp_path / "summary.txt"
        self.split = "train_seen"
        self.workers = 4
        self.episode_timeout_s = 900
        self.max_wall_s = 0
        self.coord_root = None
        self._split_size = split_size
        self._t_start = time.monotonic()
        self._session_done = 0
        # A healthy run: the summary names the mass-instant-failure abort
        # only when one fired.
        self._abort = None
        self._selection: list[str] = []
        self._done_cache: set[str] = set()
        self._infra_errors: dict = {}
        self.max_infra_attempts = 3
        self._dispatch_lock = threading.Lock()

    def write(self, *rows: dict) -> str:
        self.episodes_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self._write_summary()
        return self.summary_file.read_text()


def _row(eid: str, stamp: str, sr: int, *, status: str = "ok", **extra) -> dict:
    slug = eid.replace("::", "__").replace("#", "_")
    row = {
        "episode_id": eid, "worker_id": 0, "status": status, "elapsed_s": 1.0,
        "workdir": f"w0/maps_{stamp}_citynav_{slug}",
        "metrics": {"episode_id": eid, "SR": sr, "OSR": sr, "NE": 10.0},
    }
    row.update(extra)
    return row


def test_summary_scores_the_first_started_attempt(tmp_path):
    """The replicate that started FIRST is the one the dispatch actually
    claimed; the one that started later is the double-run."""
    s = _Summary(tmp_path, split_size=1)
    out = s.write(
        _row("e::1#1", "20260725_111500", 0),   # started second, finished first
        _row("e::1#1", "20260725_110958", 1),   # started FIRST, finished second
    )
    assert "coverage:   1 / 1 episodes" in out
    assert "SR_eff:     1/1 = 1.0000" in out


def test_a_replicated_run_reports_one_row_per_episode(tmp_path):
    """Nine attempts over four episodes -- the measured shape of
    citynav_train_seen_2026-07-25_11-04-15."""
    s = _Summary(tmp_path, split_size=4)
    out = s.write(
        _row("a::2#6474", "20260725_110954", 0),
        _row("b::2#4994", "20260725_110955", 0),
        _row("b::2#4994", "20260725_111330", 0),
        _row("c::3#6534", "20260725_111500", 0),
        _row("d::4#13444", "20260725_111000", 1),
        _row("c::3#6534", "20260725_110958", 1),
        _row("d::4#13444", "20260725_111637", 1),
        _row("d::4#13444", "20260725_111752", 1),
        _row("d::4#13444", "20260725_111726", 0),
    )
    assert "coverage:   4 / 4 episodes" in out
    # Completion order would score c::3#6534 from its 111500 attempt (SR 0)
    # and d::4#13444 from its 111726 one (SR 0), reading 1/4.
    assert "SR_eff:     2/4 = 0.5000" in out
    assert "mean_NE:    10.00 m" in out


def test_the_summary_names_the_criterion_sr_is_scored_on(tmp_path):
    out = _Summary(tmp_path, split_size=1).write(_row("e::1#1", "20260725_110000", 1))
    assert "criterion: ring" in out and "CityNav" not in out.splitlines()[0]


def test_infra_rows_stay_audited_and_out_of_the_denominator(tmp_path):
    """An infra casualty is re-queued, so counting it would depress
    SR_eff with a false zero. It is audited on its own lines, by cause."""
    s = _Summary(tmp_path, split_size=2)
    out = s.write(
        _row("e::1#1", "20260725_110000", 1),
        _row("gone::2#2", "20260725_110100", 0, status="infra",
             error="vllm_down_midrun"),
        _row("gone::3#3", "20260725_110200", 0, status="infra",
             error="killed_midrun"),
    )
    assert "infra (re-queued, not scored): 2" in out
    assert "vllm_down_midrun:1" in out
    assert "killed_midrun:1" in out
    assert "coverage:   1 / 2 episodes" in out
    assert "SR_eff:     1/1 = 1.0000" in out


def test_a_timed_out_no_commit_miss_keeps_its_coverage_slot(tmp_path):
    """A wall-clock kill with no record is a real miss, not an infra
    failure. It must occupy its slot -- dropping it would quietly shrink
    the denominator and flatter SR_eff."""
    s = _Summary(tmp_path, split_size=2)
    out = s.write(
        _row("e::1#1", "20260725_110000", 1),
        {"episode_id": "e::2#2", "worker_id": 1, "status": "timed_out",
         "elapsed_s": 3.0, "workdir": "w1/maps_20260725_110100_citynav_e__2_2"},
    )
    assert "coverage:   2 / 2 episodes" in out
    assert "with pred:  1   no pred: 1" in out
    assert "SR_eff:     1/2 = 0.5000" in out


def test_a_truncated_last_line_does_not_break_a_live_summary(tmp_path):
    """_write_summary runs after every episode, so it routinely reads a
    file another worker is mid-append to."""
    s = _Summary(tmp_path, split_size=1)
    s.episodes_file.write_text(
        json.dumps(_row("e::1#1", "20260725_110000", 1)) + "\n"
        + '{"episode_id": "e::2#2", "status": "o'
    )
    s._write_summary()
    assert "SR_eff:     1/1 = 1.0000" in s.summary_file.read_text()


def test_a_missing_ledger_still_writes_a_summary(tmp_path):
    """The first call lands before any episode has finished."""
    s = _Summary(tmp_path, split_size=4)
    s._write_summary()
    out = s.summary_file.read_text()
    assert "coverage:   0 / 4 episodes" in out
    assert "SR_eff: -" in out
