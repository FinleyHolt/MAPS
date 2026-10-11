"""Which files the canonical reader reads, and in what order.

The ledger is the run's ONE per-attempt record. A worker's ``episode.json``
beside it is the runner's own copy of a row, never a second attempt: a
reader that walked the worker dirs would count every landed row twice and
every pre-flight floor once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.citynav.analysis import rows
from benchmarks.citynav.analysis.run_report import load_run, paired, report

LEDGER = rows.EPISODES_JSONL


def _row(eid: str, stamp: str, sr: int, **extra) -> dict:
    slug = eid.replace("::", "__").replace("#", "_")
    row = {
        "episode_id": eid, "worker_id": "w0", "status": "ok", "error": None,
        "elapsed_s": 1.0, "workdir": f"w0/maps_{stamp}_citynav_{slug}",
        "metrics": {"episode_id": eid, "SR": sr, "OSR": sr, "NE": 10.0,
                    "referent_bucket": "car"},
    }
    row.update(extra)
    return row


def _write(path: Path, *dicts: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d) + "\n" for d in dicts))


# --------------------------------------------------------------------------
# Tier order
# --------------------------------------------------------------------------


def test_an_episode_json_beside_the_ledger_is_not_an_attempt(tmp_path):
    """The runner's own copy of its row is not a second attempt, and an
    episode.json with no ledger row is no attempt at all."""
    root = tmp_path / "run"
    landed = _row("e::1#1", "20260725_110000", 1)
    _write(root / LEDGER, landed)
    _write(root / landed["workdir"] / "episode.json", {**landed, "worker_id": None})
    _write(root / "w0/maps_20260725_110500_citynav_e__2_2/episode.json",
           {"episode_id": "e::2#2", "status": "ok",
            "metrics": {"episode_id": "e::2#2", "SR": 1, "NE": 0.0}})

    files, tier = rows.result_files(root)
    assert tier == "ledger"
    assert files == [root / LEDGER]
    picked, stats = rows.load_attempts(root)
    assert set(picked) == {"e::1#1"}
    assert stats.rows == 1


def test_a_root_with_nothing_readable_reports_no_tier(tmp_path):
    """No ledger, no run -- an episode.json alone is not one."""
    _write(tmp_path / "w0/maps_20260725_110000_citynav_e__1_1/episode.json",
           _row("e::1#1", "20260725_110000", 1))
    assert rows.result_files(tmp_path)[1] == "none"
    assert load_run(tmp_path) == ({}, 0)


# --------------------------------------------------------------------------
# The reader's contract
# --------------------------------------------------------------------------


def test_report_and_paired_still_run(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / LEDGER, _row("e::1#1", "20260725_110000", 1))
    _write(b / LEDGER, _row("e::1#1", "20260725_110000", 0))
    report(a)
    paired(a, b)
    out = capsys.readouterr().out
    assert "SR = 1/1 (100.0%)" in out
    assert "NE median = 10.0 m  mean = 10.0 m (n=1)" in out
    assert "source: ledger tier" in out
    assert "PAIRED on 1 shared episodes" in out


def test_the_report_prints_the_route_term(tmp_path, capsys):
    """``spl_pos`` is the one route term on a scored row, and the report
    is where it is read. Its mean rides the report, over the rows that
    carry it; a run with none says so rather than printing a zero."""
    root = tmp_path / "run"
    _write(root / LEDGER,
           _row("e::1#1", "20260725_110000", 1,
                metrics={"episode_id": "e::1#1", "SR": 1, "NE": 1.0,
                         "referent_bucket": "car", "spl_pos": 0.5}),
           _row("e::2#2", "20260725_110100", 0,
                metrics={"episode_id": "e::2#2", "SR": 0, "NE": 50.0,
                         "referent_bucket": "car", "spl_pos": 0.0}),
           _row("e::3#3", "20260725_110200", 0))
    report(root)
    out = capsys.readouterr().out
    assert "mean spl_pos = 0.2500 (n=2)" in out, out
    assert "NE median = 10.0 m  mean = 20.3 m (n=3)" in out, out
    _write(root / LEDGER, _row("e::1#1", "20260725_110000", 1,
                               metrics={"episode_id": "e::1#1", "SR": 1, "referent_bucket": "car"}))
    report(root)
    out = capsys.readouterr().out
    assert "mean spl_pos = --" in out
    assert "NE = -- (no scored row carries NE)" in out


def test_the_report_counts_timeouts_and_infra_apart(tmp_path, capsys):
    root = tmp_path / "run"
    _write(root / LEDGER,
           _row("e::1#1", "20260725_110000", 1),
           _row("e::2#2", "20260725_110100", 0, status="timed_out"),
           _row("e::3#3", "20260725_110200", 0, status="infra",
                error="vllm_down_midrun", metrics=None))
    report(root)
    out = capsys.readouterr().out
    assert "scored=2" in out
    assert "timeouts(wall-clock)=1" in out
    assert "infra(re-queued)=1" in out


def test_an_all_no_commit_run_scores_every_episode_a_miss(tmp_path, capsys):
    root = tmp_path / "run"
    _write(root / LEDGER, *[
        {"episode_id": f"e::{i}#{i}", "worker_id": "w0", "status": "ok",
         "workdir": f"w0/maps_2026072{i}_110000_citynav_e__{i}_{i}"}
        for i in range(5)
    ])
    assert load_run(root) == ({f"e::{i}#{i}": {"SR": 0} for i in range(5)}, 5)
    report(root)
    assert "SR = 0/5" in capsys.readouterr().out


def test_an_all_infra_run_refuses_and_names_what_it_read(tmp_path):
    root = tmp_path / "run"
    _write(root / LEDGER, *[
        {"episode_id": f"e::{i}#{i}", "status": "infra", "error": "serve_down"}
        for i in range(3)
    ])
    with pytest.raises(SystemExit) as excinfo:
        report(root)
    assert "infra: 3" in str(excinfo.value)
    assert "tier: ledger" in str(excinfo.value)


# --------------------------------------------------------------------------
# Which criterion the report names
# --------------------------------------------------------------------------

def _metrics(eid, sr, **extra):
    return {"episode_id": eid, "SR": sr, "NE": 10.0, "referent_bucket": "car",
            **extra}


def test_a_shape_bench_names_its_own_criterion(tmp_path, capsys):
    """Cal City scores the TARGET'S OWN SHAPE. The criterion rides the ROW, so
    the report states it instead of inferring it from how many rows resolve
    a shape.
    """
    _write(tmp_path / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 1 if i < 3 else 0,
                  metrics=_metrics(f"e::{i}#{i}", 1 if i < 3 else 0,
                                   success_criterion="shape",
                                   precise_target="shape"))
             for i in range(5)])
    report(tmp_path)
    out = capsys.readouterr().out
    assert "criterion: the target's own shape" in out, out
    assert "SR = 3/5" in out, out
    # No second success number to quote by mistake.
    assert "20 m disc SR" not in out, out


def test_a_ring_bench_names_the_disc(tmp_path, capsys):
    """The vacuity floor: a change that printed the shape label everywhere
    would pass the test above and be wrong."""
    _write(tmp_path / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 1 if i < 3 else 0,
                  metrics=_metrics(f"e::{i}#{i}", 1 if i < 3 else 0, OSR=1,
                                   success_criterion="ring"))
             for i in range(5)])
    report(tmp_path)
    out = capsys.readouterr().out
    assert "criterion: the 20 m disc" in out, out
    assert "shape" not in out, out


def test_an_unstamped_run_refuses_to_elect_a_headline(tmp_path, capsys):
    """Every run flown before the stamp. The old heuristic guessed here;
    guessing is what put the wrong number on an arm's report. Both readings
    print and neither is called the score."""
    _write(tmp_path / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 1 if i < 3 else 0) for i in range(5)])
    report(tmp_path)
    out = capsys.readouterr().out
    assert "criterion: UNRECORDED" in out, out
    assert "quote neither" in out, out


def test_paired_refuses_across_criteria(tmp_path, capsys):
    """An A/B is exactly where a criterion mismatch gets believed. Two runs
    scored on different masks answer different questions, so there is no
    delta to take."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 1,
                  metrics=_metrics(f"e::{i}#{i}", 1, success_criterion="shape"))
             for i in range(4)])
    _write(b / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 0,
                  metrics=_metrics(f"e::{i}#{i}", 0, success_criterion="ring"))
             for i in range(4)])
    with pytest.raises(SystemExit) as exc:
        paired(a, b)
    assert "INVALID pairing" in str(exc.value), exc.value


def test_paired_names_the_criterion_it_counted(tmp_path, capsys):
    """The vacuity floor for the refusal above: a matched pair must still
    pair, and must say which question moved."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 1,
                  metrics=_metrics(f"e::{i}#{i}", 1, success_criterion="shape",
                                   precise_target="shape"))
             for i in range(4)])
    _write(b / LEDGER,
           *[_row(f"e::{i}#{i}", "s", 0,
                  metrics=_metrics(f"e::{i}#{i}", 0, success_criterion="shape",
                                   precise_target="shape"))
             for i in range(4)])
    paired(a, b)
    out = capsys.readouterr().out
    assert "the target's own shape" in out, out
    assert "4/4" in out and "0/4" in out, out
