from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.capacity import reader
from benchmarks.citynav.analysis.tests.ledger import ledger_row, write_ledger


def _session(root: Path, load: float, seed: int, done: list[tuple[float, float, int]], hours: float = 1.0) -> Path:
    """A session root: report k arrives at ``done[k][0]``, is found at ``done[k][1]`` with SR ``done[k][2]``."""
    reports = [{"report": k, "arrival_s": a, "episode_id": f"calcity::{k}#{k}"} for k, (a, _f, _sr) in enumerate(done)]
    write_ledger(root, *(ledger_row(r["episode_id"], f"20260101_0000{k:02d}", sr, claimed_s=a + 10.0,
                                    elapsed_s=f - a - 10.0, arrival_s=a)
                         for k, (r, (a, f, sr)) in enumerate(zip(reports, done))))
    (root / "arrivals.json").write_text(json.dumps({"load_per_h": load, "hours": hours, "seed": seed,
                                                    "reports": reports}))
    return root


def test_finds_per_hour_count_correct_finds_inside_the_window(tmp_path) -> None:
    s = reader.session(_session(tmp_path, 4, 1, [(0, 100, 1), (60, 400, 0), (120, 3000, 1), (200, 3700, 1)]))
    assert s["finds_per_h"] == 2, "a miss and a find past the hour count no find"
    assert s["answered"] == 4 and s["accuracy"] == 0.75
    assert s["backlog"] == 1 and s["time_to_find_median_s"] == pytest.approx((100 + 2880) / 2)


def test_saturation_is_the_lightest_load_that_falls_short() -> None:
    def at(load, finds):
        return {"load_per_h": load, "offered_per_h": load, "finds_per_h": finds, "accuracy": 0.5}
    c = reader.curve([at(10, 5), at(20, 9.5), at(40, 12), at(80, 12)])
    assert c["saturation_load_per_h"] == 40 and c["capacity_finds_per_h"] == 12
    assert reader.curve([at(10, 5), at(20, 10)])["saturation_load_per_h"] is None
    # A light stream that happened to offer few reports is not saturated by finding them all.
    assert reader.curve([{**at(10, 2), "offered_per_h": 4}])["saturation_load_per_h"] is None


def test_alone_and_assisted_pair_on_the_same_stream(tmp_path) -> None:
    alone = reader.session(_session(tmp_path / "a", 4, 1, [(0, 100, 0), (60, 400, 1)]))
    assisted = reader.session(_session(tmp_path / "b", 4, 1, [(0, 100, 1), (60, 400, 1)]))
    other = reader.session(_session(tmp_path / "c", 4, 2, [(0, 100, 1)]))
    p = reader.paired([alone], [assisted, other])
    assert p["sessions"] == 1 and p["finds_per_h_delta"] == 1
    assert (p["reports"]["SR_a"], p["reports"]["SR_b"], p["reports"]["n"]) == (0.5, 1.0, 2)

    stray = reader.session(_session(tmp_path / "d", 4, 1, [(0, 100, 1)]))
    with pytest.raises(SystemExit, match="different reports"):
        reader.paired([alone], [stray])
