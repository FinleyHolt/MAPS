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
    # The time to find is over every correct report found, a late one too.
    assert s["backlog"] == 1 and s["time_to_find_median_s"] == 2880


def _arm(root: Path, load: float, service_s: float, servers: int) -> dict:
    """A session at ``load``: reports evenly spaced over the hour, flown FIFO by
    ``servers`` missions of ``service_s`` each, every one correct."""
    free = [0.0] * servers
    done = []
    for k in range(int(load)):
        a = k * 3600.0 / load
        start = max(min(free), a)
        free[free.index(min(free))] = start + service_s
        done.append((a, start + service_s, 1))
    return reader.session(_session(root, load, 1, done))


def test_saturation_is_the_lightest_load_that_falls_short(tmp_path) -> None:
    # One server, 15-minute missions: four finds an hour at most.
    light, heavy = _arm(tmp_path / "l", 2, 900, 1), _arm(tmp_path / "h", 40, 900, 1)
    c = reader.curve([light, heavy])
    assert c["saturation_load_per_h"] == 40 and c["capacity_finds_per_h"] == 4


def test_an_idle_arm_is_not_saturated_by_the_reports_too_late_to_land(tmp_path) -> None:
    # Spare servers at every load: the reports in the last 15 minutes cannot
    # land inside the hour, which says nothing about the arm's capacity.
    c = reader.curve([_arm(tmp_path / f"l{load}", load, 900, 40) for load in (10, 20, 40)])
    assert c["saturation_load_per_h"] is None


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


def test_two_sessions_on_one_stream_in_one_arm_are_refused(tmp_path) -> None:
    one = reader.session(_session(tmp_path / "a1", 4, 1, [(0, 100, 0)]))
    two = reader.session(_session(tmp_path / "a2", 4, 1, [(0, 100, 1)]))
    assisted = reader.session(_session(tmp_path / "b", 4, 1, [(0, 100, 1)]))
    for alone, assist in (([one, two], [assisted]), ([assisted], [one, two])):
        with pytest.raises(SystemExit, match="a1.*a2"):
            reader.paired(alone, assist)


def test_a_pairing_that_pairs_nothing_prints_strict_json_and_text(tmp_path, capsys) -> None:
    b = _session(tmp_path / "b", 4, 1, [(0, 100, 1)])
    other = _session(tmp_path / "a", 4, 2, [(0, 100, 1)])
    assert reader.main(["--assisted", str(b), "--alone", str(other), "--json"]) == 0
    out = json.loads(capsys.readouterr().out, parse_constant=lambda c: pytest.fail(f"{c} in the JSON"))
    assert out["paired"]["sessions"] == 0 and out["paired"]["finds_per_h_delta_ci95"] is None
    assert reader.main(["--assisted", str(b), "--alone", str(other)]) == 0
    assert "paired over 0 sessions" in capsys.readouterr().out


def test_a_report_lost_to_the_infrastructure_is_offered_and_counted(tmp_path) -> None:
    root = _session(tmp_path, 4, 1, [(0, 100, 1), (60, 400, 1)])
    ledger = root / "episodes.jsonl"
    rows = ledger.read_text().splitlines()
    ledger.write_text(rows[0] + "\n" + json.dumps(ledger_row("calcity::1#1", "20260101_000001", status="infra"))
                      + "\n")
    (root / "run_manifest.json").write_text(json.dumps({"selection": {"episodes_n": 2}}))
    s = reader.session(root)
    assert (s["offered"], s["answered"], s["infra"], s["finds_per_h"]) == (2, 1, 1, 1)


def test_the_due_cut_reads_the_light_load_s_time_to_a_correct_find(tmp_path) -> None:
    # Light load: two correct finds at 300 s and two wrong answers at 3,000 s
    # (cap timeouts); the median time to find is 300 s, so a report is due
    # when it arrives by 3,300 s.
    light = reader.session(_session(tmp_path / "l", 4, 1, [(0, 300, 1), (100, 400, 1), (200, 3200, 0),
                                                         (300, 3300, 0)]))
    heavy = reader.session(_session(tmp_path / "h", 18, 1, [(a, a + 1000, 1) for a in range(0, 3600, 200)]))
    assert light["time_to_find_median_s"] == 300
    # 17 reports arrive by 3,300 s; the 14 that arrive by 2,600 s land inside the hour.
    assert reader.curve([light, heavy])["points"][1]["due_yield"] == round(14 / 17, 3)
