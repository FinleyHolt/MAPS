"""The rule: an episode's result is its first fully complete attempt."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from benchmarks.citynav.analysis import rows
from benchmarks.citynav.analysis.run_report import load_run
from benchmarks.citynav.analysis.tests.ledger import ledger_row, write_ledger

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "attempt_pick"


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_the_first_started_attempt_wins_not_the_last_written(tmp_path):
    """The ledger appends on COMPLETION, so file order ranks a fast
    duplicate ahead of the slow attempt that was claimed first."""
    root = write_ledger(
        tmp_path / "run",
        ledger_row("e::1#1", "20260725_111500", 0),   # started second, wrote first
        ledger_row("e::1#1", "20260725_110958", 1),   # started FIRST, wrote second
    )
    picked, _ = rows.load_attempts(root)
    assert picked["e::1#1"].stamp == "20260725_110958"
    assert picked["e::1#1"].metrics["SR"] == 1


def test_a_timed_out_attempt_is_not_complete(tmp_path):
    """It committed before the kill and carries metrics, but the driver
    never saw a clean exit: any ok attempt of the episode outranks it."""
    root = write_ledger(
        tmp_path / "run",
        ledger_row("e::3#3", "20260725_110100", 1, status="timed_out"),
        ledger_row("e::3#3", "20260725_113000", 0),
    )
    picked, stats = rows.load_attempts(root)
    assert picked["e::3#3"].stamp == "20260725_113000"
    assert picked["e::3#3"].metrics["SR"] == 0
    assert stats.timed_out == 1


def test_status_is_the_one_verdict():
    """Only ``ok`` with metrics is complete. A row with no status at all is
    not ok: the producer stamps every row, and absence must not read as a
    clean exit."""
    m = {"SR": 1}
    assert rows.is_complete({"status": "ok", "metrics": m})
    assert not rows.is_complete({"status": "timed_out", "metrics": m})
    assert not rows.is_complete({"status": "infra", "metrics": m})
    assert not rows.is_complete({"metrics": m})
    assert not rows.is_complete({"status": "ok"})


# --------------------------------------------------------------------------
# Infra supersession
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("crash_stamp", "good_stamp"),
    [
        ("20260725_112000", "20260725_113000"),  # crash first, retry after
        ("20260725_113000", "20260725_112000"),  # stamps swapped
    ],
)
def test_a_reclaimed_retry_supersedes_its_earlier_failure(
    tmp_path, crash_stamp, good_stamp
):
    """Run BOTH orderings: the retry wins either way, which proves
    supersession comes from the status and not from the ordering."""
    crash = ledger_row("e::4#4", crash_stamp, status="infra", error="killed_midrun")
    root = write_ledger(tmp_path / "run", crash, ledger_row("e::4#4", good_stamp, 1))
    picked, stats = rows.load_attempts(root)
    assert picked["e::4#4"].stamp == good_stamp
    assert picked["e::4#4"].metrics["SR"] == 1
    assert stats.infra == 1


def test_an_infra_abort_never_represents_its_episode(tmp_path):
    """A lost reasoner is not a fair run. It is audit-counted, but it must
    not occupy the episode's slot even as a last resort -- otherwise an
    infra casualty lands in the coverage denominator as a false zero."""
    root = write_ledger(tmp_path / "run", ledger_row(
        "e::5#5", "20260725_110000", status="infra", error="reasoner_down_midrun"))
    picked, stats = rows.load_attempts(root)
    assert "e::5#5" not in picked
    assert stats.infra == 1
    assert stats.no_metrics == 0


def _ended(eid: str, ended_by: str, status: str = "ok") -> dict:
    return ledger_row(eid, "20260827_100000", status=status,
                      metrics={"SR": 0, "mission": {"ended_by": ended_by}})


def test_one_rule_reads_the_infra_stamp_or_the_serve_ending():
    """The bench producer stamps ``infra``; a flight row carries the ending
    itself with ``ok``. One predicate reads both."""
    assert rows.is_infra_abort(_ended("c::1#0", "transient_failures", "infra"))
    assert rows.is_infra_abort(_ended("c::2#0", "transient_failures"))
    assert not rows.is_infra_abort(ledger_row("c::3#0", "20260827_100000", 1))
    assert rows.is_infra_abort({"episode_id": "c::4#0", "status": "infra"})


def test_the_algorithm_s_own_budget_still_scores(tmp_path):
    """turn_cap, compaction_infeasible and oversize_rejection are the agent
    spending what it was given. Excluding those would launder real misses."""
    root = write_ledger(tmp_path / "run", *(
        _ended(f"c::{i}#0", ended_by)
        for i, ended_by in enumerate(("turn_cap", "compaction_infeasible", "oversize_rejection"))))
    picked, stats = rows.load_attempts(root)
    assert len(picked) == 3, "an algorithm miss was excluded from SR"
    assert stats.infra == 0


# --------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------


def test_an_episode_with_no_complete_attempt_keeps_its_last_scored_row(tmp_path):
    """The fallback row is chosen by attempt START, like every other
    choice."""
    root = write_ledger(
        tmp_path / "run",
        ledger_row("e::6#6", "20260725_115000", 1, status="timed_out"),
        ledger_row("e::6#6", "20260725_114500", 0, status="timed_out"),
    )
    picked, stats = rows.load_attempts(root)
    assert picked["e::6#6"].stamp == "20260725_115000"
    assert not picked["e::6#6"].complete
    assert stats.timed_out == 2


def test_a_no_commit_miss_keeps_its_coverage_slot(tmp_path):
    """An ``ok`` row with no metrics is a real miss, not a crash. It never
    scores -- but dropping it would quietly shrink the denominator."""
    root = write_ledger(tmp_path / "run", ledger_row("e::7#7", "20260725_114000", elapsed_s=3.0))
    picked, stats = rows.load_attempts(root)
    assert "e::7#7" in picked
    assert picked["e::7#7"].metrics == {}
    assert stats.no_metrics == 1


def test_a_crash_row_with_no_workdir_is_counted_and_never_wins(tmp_path):
    """The driver's spawn-failure row carries no ``workdir`` at all. A
    naive ``Path(row["workdir"]).name`` explodes on it."""
    naked = {"episode_id": "e::8#8", "worker_id": "w0", "status": "infra",
             "error": "attempt spawn: OSError: [Errno 24] Too many open files"}
    assert rows.attempt_stamp(naked) == ""
    root = write_ledger(tmp_path / "run", naked, ledger_row("e::8#8", "20260725_120000", 1))
    picked, stats = rows.load_attempts(root)
    assert picked["e::8#8"].stamp == "20260725_120000"
    assert stats.rows == 2 and stats.infra == 1 and stats.no_metrics == 0


# --------------------------------------------------------------------------
# The stamp
# --------------------------------------------------------------------------


def test_a_run_root_name_is_not_an_attempt_stamp():
    """A run-root name ``<benchmark>_<split>_<Y-m-d>_<H-M-S>`` is no attempt
    stamp: matching it would give every row of a run the same stamp."""
    assert rows.attempt_stamp({"workdir": "citynav_val_seen_2026-07-25_10-41-14"}) == ""
    assert rows.attempt_stamp(
        {"workdir": "w0/maps_20260725_110000_citynav_e__1_1"}) == "20260725_110000"


# --------------------------------------------------------------------------
# Row shapes
# --------------------------------------------------------------------------


def test_metrics_is_one_shape():
    assert rows.metrics({}) == {}
    assert rows.metrics({"metrics": None}) == {}
    assert rows.metrics({"metrics": {"SR": 0}}) == {"SR": 0}


def test_a_crash_row_does_carry_an_episode_id():
    row = {"episode_id": "e::9#9", "status": "ok"}
    assert rows.episode_id(row) == "e::9#9"


def test_a_truncated_last_line_does_not_break_the_read(tmp_path):
    """A live run's last line is routinely a partial write."""
    root = tmp_path / "live"
    root.mkdir()
    (root / rows.EPISODES_JSONL).write_text(
        json.dumps(ledger_row("e::10#10", "20260725_110000", 1)) + "\n"
        + '{"episode_id": "e::11#11", "status": "o'
    )
    picked, stats = rows.load_attempts(root)
    assert set(picked) == {"e::10#10"}
    assert stats.rows == 1


# --------------------------------------------------------------------------
# The shared pin
# --------------------------------------------------------------------------


def test_every_shared_fixture_case():
    """The shared fixture tree and its expected picks: ``picks`` carry
    metrics, ``unscored`` are picked episodes with none (each a miss)."""
    assert FIXTURES.is_dir(), f"shared attempt-pick fixture missing at {FIXTURES}"
    expected = json.loads((FIXTURES / "expected.json").read_text())
    assert expected["runs"], "vacuity floor: the fixture pins at least one run"
    for run_name, want in expected["runs"].items():
        root = FIXTURES / run_name
        picked, stats = rows.load_attempts(root)
        for key in ("rows", "no_metrics", "timed_out", "infra"):
            assert getattr(stats, key) == want[key], (run_name, key)

        got = {
            eid: {
                "attempt": Path(str(a.row["workdir"])).name,
                "SR": a.metrics.get("SR"),
                "NE": a.metrics.get("NE"),
            }
            for eid, a in picked.items()
            if a.metrics
        }
        assert got == {
            eid: {"attempt": w["attempt"], "SR": w["SR"], "NE": w["NE"]}
            for eid, w in want["picks"].items()
        }, run_name

        scored = load_run(root)
        assert set(scored) == set(want["picks"]) | set(want["unscored"]), run_name
        for eid in want["unscored"]:
            assert scored[eid]["SR"] == 0 and scored[eid].get("OSR", 0) == 0, \
                f"{run_name}: {eid} keeps its slot and scores a miss"
        for eid in want["never_landed"]:
            assert eid not in picked, f"{run_name}: {eid} has no ledger row"


def test_every_reader_counts_a_timed_out_episode_with_no_record_as_a_miss(tmp_path):
    """A wall-clock timeout before any commit is a miss inside the SR
    denominator; summary.txt counts it, so every reader must."""
    from benchmarks.citynav.analysis import arm_table, run_report

    root = write_ledger(
        tmp_path / "run",
        ledger_row("e1", "20260101_000001", 1),
        ledger_row("e2", "20260101_000002", 0),
        ledger_row("e3", "20260101_000003", status="timed_out"),
        {"episode_id": "e4", "status": "infra", "error": "serve_down"},
        ledger_row("e4", "20260101_000005", 0),
    )
    scored = load_run(root)
    assert sorted(scored) == ["e1", "e2", "e3", "e4"]
    assert scored["e3"]["SR"] == 0 and scored["e3"].get("OSR", 0) == 0
    assert len(run_report.report(root)) == 4
    row = arm_table.arm_row(root)
    assert (row["n"], row["SR_n"], row["SR"]) == (4, 1, 0.25)


def _partial_run(tmp_path: Path, episodes_n: int) -> Path:
    """Three selected episodes: one scored, one only ever infra, one never landed."""
    root = write_ledger(
        tmp_path / f"selected_{episodes_n}",
        ledger_row("e1", "20260101_000001", 1),
        ledger_row("e2", "20260101_000002", status="infra", error="serve_down"),
    )
    (root / "run_manifest.json").write_text(json.dumps({"selection": {"episodes_n": episodes_n}}))
    return root


def test_every_reader_refuses_a_partial_root_naming_the_shortfall(tmp_path, capsys, monkeypatch):
    from benchmarks.citynav.analysis import ab_report, arm_table, run_report

    root = _partial_run(tmp_path, 3)
    monkeypatch.setattr(sys, "argv", ["ab_report", str(root), str(root)])
    want = "scores 1 of the 3 episodes its manifest selects (1 only ever infrastructure casualties, 1 never landed a row)"
    for read in (lambda: arm_table.main([str(root)]), ab_report.main):
        with pytest.raises(SystemExit, match=re.escape(want)) as refused:
            read()
        assert "resume" in str(refused.value) and "--max-infra-attempts" in str(refused.value)
    assert run_report.main([str(root)]) == 1  # run_report names the refused root and reads on
    refused = capsys.readouterr().err
    assert want in refused and "resume" in refused and "--max-infra-attempts" in refused
    assert run_report.main([str(root), "--allow-partial"]) == 0
    assert arm_table.main(["--allow-partial", str(root)]) == 0
    assert f"*** PARTIAL (--allow-partial): {root}: {want} ***" in capsys.readouterr().err


def test_a_sortie_root_selecting_no_episodes_reads_whole(tmp_path):
    """A sortie's manifest selects nothing up front (episodes_n 0): its
    missions are flown one at a time, so no count bounds the read."""
    picked, _ = rows.load_attempts(_partial_run(tmp_path, 0))
    assert set(picked) == {"e1"}


def test_a_metricless_pick_is_an_osr_and_spl_miss_on_a_ring_run():
    """A crash or an unrecorded timeout is a miss in OSR's and spl_pos's
    denominators as in SR's, so all three are rates over the same episodes."""
    def pick(metrics):
        return rows.Attempt("e", {}, metrics, "20260725_110000", 0, True)
    picked = {"a": pick({"SR": 1, "OSR": 1, "spl_pos": 0.5}), "b": pick({})}
    assert rows.scored_metrics(picked)["b"] == {"SR": 0, "OSR": 0, "spl_pos": 0.0}
    shape = {"a": pick({"SR": 1}), "b": pick({})}
    assert rows.scored_metrics(shape)["b"] == {"SR": 0}


def test_the_manifest_is_the_roots_own_never_a_missions_below_it(tmp_path):
    root = write_ledger(tmp_path / "sortie", ledger_row("e1", "20260101_000001", 1))
    (root / "run03").mkdir()
    (root / "run03" / "run_manifest.json").write_text(json.dumps({"selection": {"episodes_n": 5}}))
    assert rows.read_manifest(root) == {}
