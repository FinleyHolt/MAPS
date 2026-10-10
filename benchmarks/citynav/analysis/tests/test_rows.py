"""The rule: an episode's result is its first fully complete attempt.

Nothing pinned reader semantics before this file, which is exactly how
four readers of the same rows came to disagree about the same run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.citynav.analysis import rows
from benchmarks.citynav.analysis.run_report import load_run

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "attempt_pick"


def _ledger(eid: str, stamp: str, sr: int | None = None, *, status: str = "ok",
            worker: str = "w0", **extra) -> dict:
    """A ledger row for one attempt of ``eid``, ``workdir`` relative."""
    slug = eid.replace("::", "__").replace("#", "_")
    row: dict = {
        "episode_id": eid, "worker_id": worker, "status": status, "error": None,
        "elapsed_s": 1.0, "workdir": f"{worker}/maps_{stamp}_citynav_{slug}",
        "oom_retries": 0,
    }
    if sr is not None:
        row["metrics"] = {"episode_id": eid, "SR": sr, "OSR": sr, "NE": 10.0}
    row.update(extra)
    return row


def _run(tmp_path: Path, *ledger_rows: dict, name: str = "run") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / rows.EPISODES_JSONL).write_text(
        "".join(json.dumps(r) + "\n" for r in ledger_rows)
    )
    return root


def _picked_stamp(attempt: rows.Attempt) -> str:
    return attempt.stamp


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_the_first_started_attempt_wins_not_the_last_written(tmp_path):
    """The ledger appends on COMPLETION, so file order ranks a fast
    duplicate ahead of the slow attempt that was claimed first. Measured
    on citynav_train_seen_2026-07-24_13-19-41: one episode's attempts sit
    at file positions 4, 6, 7 with start stamps 132524, 133120, 133016."""
    root = _run(
        tmp_path,
        _ledger("e::1#1", "20260725_111500", 0),   # started second, wrote first
        _ledger("e::1#1", "20260725_110958", 1),   # started FIRST, wrote second
    )
    picked, _ = rows.load_attempts(root)
    assert _picked_stamp(picked["e::1#1"]) == "20260725_110958"
    assert picked["e::1#1"].metrics["SR"] == 1


def test_a_timed_out_attempt_is_not_complete(tmp_path):
    """It committed before the kill and carries metrics, but the driver
    never saw a clean exit: any ok attempt of the episode outranks it."""
    root = _run(
        tmp_path,
        _ledger("e::3#3", "20260725_110100", 1, status="timed_out"),
        _ledger("e::3#3", "20260725_113000", 0),
    )
    picked, stats = rows.load_attempts(root)
    assert _picked_stamp(picked["e::3#3"]) == "20260725_113000"
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
    assert not rows.is_complete({"status": "flying", "metrics": m})
    assert not rows.is_complete({"metrics": m})
    assert not rows.is_complete({"status": "ok"})
    assert rows.is_provisional({"status": "flying"})
    assert not rows.is_provisional({"status": "ok", "provisional": True})


# --------------------------------------------------------------------------
# Infra supersession — the property that had to survive first-by-start
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
    supersession comes from the status and not from the ordering. That
    is what let first-by-start replace last-row-wins without losing it."""
    crash = _ledger("e::4#4", crash_stamp, status="infra", error="killed_midrun")
    root = _run(tmp_path, crash, _ledger("e::4#4", good_stamp, 1))
    picked, stats = rows.load_attempts(root)
    assert _picked_stamp(picked["e::4#4"]) == good_stamp
    assert picked["e::4#4"].metrics["SR"] == 1
    assert stats.infra == 1


def test_an_infra_abort_never_represents_its_episode(tmp_path):
    """A vLLM death is not a fair run. It is audit-counted, but it must
    not occupy the episode's slot even as a last resort -- otherwise an
    infra casualty lands in the coverage denominator as a false zero."""
    root = _run(tmp_path, _ledger("e::5#5", "20260725_110000",
                                  status="infra", error="vllm_down_midrun"))
    picked, stats = rows.load_attempts(root)
    assert "e::5#5" not in picked
    assert stats.infra == 1
    assert stats.no_metrics == 1


# --------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------


def test_an_episode_with_no_complete_attempt_keeps_its_last_scored_row(tmp_path):
    """Ordered by START like everything else, so the fallback cannot
    reintroduce the file ordering this change removed."""
    root = _run(
        tmp_path,
        _ledger("e::6#6", "20260725_115000", 1, status="timed_out"),
        _ledger("e::6#6", "20260725_114500", 0, status="timed_out"),
    )
    picked, stats = rows.load_attempts(root)
    assert _picked_stamp(picked["e::6#6"]) == "20260725_115000"
    assert not picked["e::6#6"].complete
    assert stats.timed_out == 2


def test_a_no_commit_miss_keeps_its_coverage_slot(tmp_path):
    """An ``ok`` row with no metrics is a real miss, not a crash. It never
    scores -- but dropping it would quietly shrink the denominator. This
    is the shape of all fourteen rows in citynav_val_seen_2026-07-23_16-56-19."""
    root = _run(tmp_path, _ledger("e::7#7", "20260725_114000", elapsed_s=3.0))
    picked, stats = rows.load_attempts(root)
    assert "e::7#7" in picked
    assert picked["e::7#7"].metrics == {}
    assert stats.no_metrics == 1


def test_a_crash_row_with_no_workdir_is_counted_and_never_wins(tmp_path):
    """The driver's spawn-failure row carries no ``workdir`` at all. A
    naive ``Path(row["workdir"]).name`` explodes on it."""
    naked = {"episode_id": "e::8#8", "worker_id": "w0", "status": "infra",
             "error": "RuntimeError: warm runner never came up"}
    assert rows.attempt_stamp(naked) == ""
    root = _run(tmp_path, naked, _ledger("e::8#8", "20260725_120000", 1))
    picked, stats = rows.load_attempts(root)
    assert _picked_stamp(picked["e::8#8"]) == "20260725_120000"
    assert stats.rows == 2 and stats.infra == 1 and stats.no_metrics == 1


# --------------------------------------------------------------------------
# The stamp
# --------------------------------------------------------------------------


def test_a_run_root_name_is_not_an_attempt_stamp():
    """The dashboard's fleet_stamp deliberately matches the RUN-root name
    ``<benchmark>_<split>_<Y-m-d>_<H-M-S>`` as well. The attempt pattern must
    not: matching it would hand every row in a standalone run the same
    stamp and collapse the ordering entirely."""
    assert rows.attempt_stamp({"workdir": "citynav_val_seen_2026-07-25_10-41-14"}) == ""
    assert rows.attempt_stamp(
        {"workdir": "w0/maps_20260725_110000_citynav_e__1_1"}) == "20260725_110000"


def test_the_ledger_name_matches_the_producer():
    """The producer (``episodes``) must not import a reader, so the literal
    is duplicated and this keeps the two honest."""
    from benchmarks.citynav import episodes

    assert rows.EPISODES_JSONL == episodes.EPISODES_JSONL


# --------------------------------------------------------------------------
# Row shapes
# --------------------------------------------------------------------------


def test_metrics_is_one_shape():
    assert rows.metrics({}) == {}
    assert rows.metrics({"metrics": None}) == {}
    assert rows.metrics({"metrics": {"SR": 0}}) == {"SR": 0}


def test_a_crash_row_does_carry_an_episode_id():
    """The canonical reader's docstring claimed otherwise for years. Every
    no-metrics row in citynav_val_seen_2026-07-23_16-56-19 has one."""
    row = {"episode_id": "e::9#9", "status": "ok"}
    assert rows.episode_id(row) == "e::9#9"


def test_a_truncated_last_line_does_not_break_the_read(tmp_path):
    """A live run's last line is routinely a partial write."""
    root = tmp_path / "live"
    root.mkdir()
    (root / rows.EPISODES_JSONL).write_text(
        json.dumps(_ledger("e::10#10", "20260725_110000", 1)) + "\n"
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
        for key in ("tier", "files", "rows", "no_metrics", "timed_out", "infra"):
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

        scored, no_metrics = load_run(root)
        assert set(scored) == set(want["picks"]) | set(want["unscored"]), run_name
        assert no_metrics == want["no_metrics"], run_name
        for eid in want["unscored"]:
            assert scored[eid] == {"SR": 0}, \
                f"{run_name}: {eid} keeps its slot and scores a miss"
        for eid in want["never_landed"]:
            assert eid not in picked, f"{run_name}: {eid} has no ledger row"


def test_every_reader_counts_a_timed_out_episode_with_no_record_as_a_miss(tmp_path):
    """A wall-clock timeout before any commit is a miss inside the SR
    denominator; summary.txt counts it, so every reader must."""
    from benchmarks.citynav.analysis import ab_report, arm_table, run_report

    root = _run(
        tmp_path,
        _ledger("e1", "20260101_000001", 1),
        _ledger("e2", "20260101_000002", 0),
        _ledger("e3", "20260101_000003", status="timed_out"),
        {"episode_id": "e4", "status": "infra", "error": "serve_down"},
        _ledger("e4", "20260101_000005", 0),
    )
    scored, _ = load_run(root)
    assert sorted(scored) == ["e1", "e2", "e3", "e4"]
    assert scored["e3"] == {"SR": 0}
    assert len(run_report.report(root)) == 4
    row = arm_table.arm_row(root)
    assert (row["n"], row["SR_n"], row["SR"]) == (4, 1, 0.25)
    assert ab_report._sr_by_episode(root)["e3"] == 0
