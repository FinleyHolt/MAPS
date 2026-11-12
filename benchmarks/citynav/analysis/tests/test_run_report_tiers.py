"""Which files the canonical reader reads, and in what order.

The ledger is the run's ONE per-attempt record. A worker's ``episode.json``
beside it is the runner's own copy of a row, never a second attempt: a
reader that walked the worker dirs would count every landed row twice and
every pre-flight floor once.
"""

from __future__ import annotations

import json
import os

import pytest

from benchmarks.citynav.analysis import run_report
from benchmarks.citynav.analysis.run_report import load_run
from benchmarks.run_record import rows
from benchmarks.run_record.tests.ledger import ledger_row, write_ledger


def report(root):
    """The report over ``root`` as the CLI prints it: the benches placed by this process's environment."""
    return run_report.report(root, run_report.benches(os.environ))


# --------------------------------------------------------------------------
# Tier order
# --------------------------------------------------------------------------


def test_an_episode_json_beside_the_ledger_is_not_an_attempt(tmp_path):
    """The runner's own copy of its row is not a second attempt, and an
    episode.json with no ledger row is no attempt at all."""
    root = tmp_path / "run"
    landed = ledger_row("e::1#1", "20260725_110000", 1)
    write_ledger(root, landed)
    write_ledger(root / landed["workdir"], {**landed, "worker_id": None}, name="episode.json")
    write_ledger(
        root / "w0/maps_20260725_110500_citynav_e__2_2",
        {
            "episode_id": "e::2#2",
            "status": "ok",
            "metrics": {"episode_id": "e::2#2", "SR": 1, "NE": 0.0},
        },
        name="episode.json",
    )

    picked, stats = rows.load_attempts(root)
    assert set(picked) == {"e::1#1"}
    assert stats.rows == 1


def test_a_root_without_a_ledger_is_refused(tmp_path):
    """No ledger, no run -- an episode.json alone is not one, nor is a mistyped root."""
    write_ledger(
        tmp_path / "w0/maps_20260725_110000_citynav_e__1_1",
        ledger_row("e::1#1", "20260725_110000", 1),
        name="episode.json",
    )
    for root in (tmp_path, tmp_path / "typo"):
        with pytest.raises(SystemExit, match=r": not a run root \(no episodes.jsonl\)"):
            load_run(root)


# --------------------------------------------------------------------------
# The reader's contract
# --------------------------------------------------------------------------


def test_report_runs(tmp_path, capsys):
    a = tmp_path / "a"
    write_ledger(a, ledger_row("e::1#1", "20260725_110000", 1))
    report(a)
    out = capsys.readouterr().out
    assert "SR = 1/1 (100.0%)" in out
    assert "NE median = 10.0 m  mean = 10.0 m (n=1)" in out
    assert "tier" not in out


def test_the_report_prints_the_route_term(tmp_path, capsys):
    """``spl_pos`` is the one route term on a scored row, and the report
    is where it is read. Its mean rides the report, over the rows that
    carry it; a run with none (a shape-scored run) prints no route line."""
    root = tmp_path / "run"
    write_ledger(
        root,
        ledger_row("e::1#1", "20260725_110000", metrics={"SR": 1, "NE": 1.0, "spl_pos": 0.5}),
        ledger_row("e::2#2", "20260725_110100", metrics={"SR": 0, "NE": 50.0, "spl_pos": 0.0}),
        ledger_row("e::3#3", "20260725_110200", 0),
    )
    report(root)
    out = capsys.readouterr().out
    assert "mean spl_pos = 0.2500 (n=2)" in out, out
    assert "NE median = 10.0 m  mean = 20.3 m (n=3)" in out, out
    write_ledger(root, ledger_row("e::1#1", "20260725_110000", metrics={"SR": 1}))
    report(root)
    out = capsys.readouterr().out
    assert "spl_pos" not in out
    assert "NE = -- (no scored row carries NE)" in out


def test_the_report_counts_timeouts_and_infra_apart(tmp_path, capsys):
    root = tmp_path / "run"
    write_ledger(
        root,
        ledger_row("e::1#1", "20260725_110000", 1),
        ledger_row("e::2#2", "20260725_110100", 0, status="timed_out"),
        ledger_row("e::3#3", "20260725_110200", status="infra", error="reasoner_down_midrun"),
    )
    report(root)
    out = capsys.readouterr().out
    assert "scored=2" in out
    assert "timeouts(wall-clock)=1" in out
    assert "infra(excluded)=1" in out


def test_an_all_no_commit_run_scores_every_episode_a_miss(tmp_path, capsys):
    root = tmp_path / "run"
    write_ledger(
        root,
        *[
            {
                "episode_id": f"e::{i}#{i}",
                "worker_id": "w0",
                "status": "ok",
                "workdir": f"w0/maps_2026072{i}_110000_citynav_e__{i}_{i}",
            }
            for i in range(5)
        ],
    )
    assert load_run(root) == {f"e::{i}#{i}": {"SR": 0} for i in range(5)}
    report(root)
    assert "SR = 0/5" in capsys.readouterr().out


def test_an_all_infra_run_refuses_and_names_what_it_read(tmp_path):
    root = tmp_path / "run"
    write_ledger(
        root,
        *[
            {"episode_id": f"e::{i}#{i}", "status": "infra", "error": "serve_down"}
            for i in range(3)
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        report(root)
    assert "infra: 3" in str(excinfo.value)
    assert "tier" not in str(excinfo.value)


# --------------------------------------------------------------------------
# Which criterion the report names
# --------------------------------------------------------------------------


def _scored(srs, **metrics) -> list[dict]:
    """One ledger row per entry of ``srs``, each scoring that SR with ``metrics``."""
    return [
        ledger_row(f"e::{i}#{i}", "s", metrics={"SR": sr, "NE": 10.0, **metrics})
        for i, sr in enumerate(srs)
    ]


def test_a_shape_bench_names_its_own_criterion(tmp_path, capsys):
    """Cal City scores the TARGET'S OWN SHAPE. The criterion rides the ROW, so
    the report states it instead of inferring it from how many rows resolve
    a shape.
    """
    write_ledger(
        tmp_path, *_scored([1, 1, 1, 0, 0], success_criterion="shape", precise_target="shape")
    )
    report(tmp_path)
    out = capsys.readouterr().out
    assert "criterion: the target's own shape" in out, out
    assert "SR = 3/5" in out, out
    # No second success number to quote by mistake.
    assert "20 m disc SR" not in out, out


def test_a_ring_bench_names_the_disc(tmp_path, capsys):
    """The vacuity floor: a change that printed the shape label everywhere
    would pass the test above and be wrong."""
    write_ledger(tmp_path, *_scored([1, 1, 1, 0, 0], OSR=1, success_criterion="ring"))
    report(tmp_path)
    out = capsys.readouterr().out
    assert "criterion: the disc of unrecorded radius" in out, out
    assert "shape" not in out, out


def test_the_disc_label_reads_the_run_s_configured_radius(tmp_path, capsys):
    """The disc is the run's ``success_radius_m`` (manifest ``config_yaml``), not a fixed 20 m;
    unstamped rows name the same disc."""
    write_ledger(tmp_path, *_scored([1, 0], OSR=1, success_criterion="ring"))
    (tmp_path / "run_manifest.json").write_text('{"config_yaml": "success_radius_m: 30.0\\n"}')
    report(tmp_path)
    assert "criterion: the 30 m disc" in capsys.readouterr().out
    write_ledger(tmp_path, *_scored([1, 0], OSR=1))
    report(tmp_path)
    assert "the SR above is the 30 m disc" in capsys.readouterr().out


def test_an_unstamped_run_refuses_to_elect_a_headline(tmp_path, capsys):
    """A run with no stamp: both readings print and neither is called the
    score, since a guess would put the wrong number on an arm's report."""
    write_ledger(tmp_path, *_scored([1, 1, 1, 0, 0]))
    report(tmp_path)
    out = capsys.readouterr().out
    assert "criterion: UNRECORDED" in out, out
    assert "quote neither" in out, out


def test_a_root_with_no_recorded_metrics_claims_no_disc_sr(tmp_path, capsys):
    """Timeouts with no record carry no SR to call the 20 m disc."""
    write_ledger(tmp_path, ledger_row("e::1#1", "20260725_110000", status="timed_out"))
    report(tmp_path)
    out = capsys.readouterr().out
    assert "SR = 0/1" in out and "criterion: UNRECORDED" in out, out
    assert "quote neither" not in out, out


def test_failed_tiles_are_reported_beside_the_sr(tmp_path, capsys):
    write_ledger(
        tmp_path,
        *(
            ledger_row(eid, "s", metrics={"SR": 0, "NE": 10.0, "sam_tiles_failed": n})
            for eid, n in (("a", 3), ("b", 0), ("c", 2))
        ),
    )
    report(tmp_path)
    out = capsys.readouterr().out
    assert "SAHI tiles failed = 5 over 2 of 3 scored episode(s)" in out, out
    write_ledger(tmp_path, ledger_row("a", "s", metrics={"SR": 0, "sam_tiles_failed": 0}))
    report(tmp_path)
    assert "SAHI tiles failed" not in capsys.readouterr().out


def test_help_is_an_option_not_a_root(capsys):
    import pytest

    from benchmarks.citynav.analysis import run_report

    with pytest.raises(SystemExit) as done:
        run_report.main(["--help"])
    assert done.value.code == 0 and "--allow-partial" in capsys.readouterr().out


def test_a_root_with_no_scored_row_does_not_stop_the_roots_after_it(tmp_path, capsys):
    """Each root in turn: a refused root is named on stderr, the next is still reported, and the
    exit code says one was refused."""
    from benchmarks.citynav.analysis import run_report

    empty = write_ledger(
        tmp_path / "empty",
        ledger_row("e::1#1", "20260725_110000", status="infra", error="reasoner_down_midrun"),
    )
    good = write_ledger(tmp_path / "good", ledger_row("e::2#2", "20260725_110100", 1))
    assert run_report.main([str(empty), str(good)]) == 1
    out = capsys.readouterr()
    assert f"no scored rows under {empty}" in out.err
    assert f"=== {good} ===" in out.out


def test_the_provenance_line_orders_ties_by_name(tmp_path, capsys):
    root = tmp_path / "run"
    write_ledger(
        root,
        *(
            ledger_row(
                f"e::{i}#{i}",
                f"20260725_1{i}0000",
                metrics={"SR": 1, "NE": 1.0, "committed_provenance": prov},
            )
            for i, prov in enumerate(("point", "detection", "point", "detection", "landmark"))
        ),
    )
    report(root)
    assert "provenance: detection=2  point=2  landmark=1" in capsys.readouterr().out


def _calcity_corpus(tmp_path, monkeypatch, *rows: tuple[str, str | None]) -> None:
    """A Cal City split of ``(description, intel_quality)`` rows, ids ``calcity::<i>#<i>``."""
    corpus = [
        {
            "area": "calcity",
            "descriptions": [d],
            "ann_ids": [i],
            "target_positions": [[0.0, 0.0]],
            "trajectory": [[0.0, 0.0, 100.0]],
            "intel_quality": q,
        }
        for i, (d, q) in enumerate(rows)
    ]
    (tmp_path / "calcity.json").write_text(json.dumps(corpus))
    monkeypatch.setenv("SEARCH_CALCITY_SPLITS", str(tmp_path))


def test_the_by_class_lines_read_a_row_s_own_class_and_the_corpus_for_a_pick_without_one(
    tmp_path, monkeypatch, capsys
):
    """A recorded row's referent bucket and intel quality are its own keys; a pick without them
    takes its corpus row's, so a G episode that timed out with no record is a G miss, not a '?'
    one. A run whose every row carries its class needs no split on this machine."""
    _calcity_corpus(
        tmp_path,
        monkeypatch,
        ("the red car", "G"),
        ("a building", "G"),
        ("the lot", "G"),
        ("a shed", None),
    )
    root = tmp_path / "run"
    write_ledger(
        root,
        ledger_row("calcity::0#0", "20260725_110000", 1),
        ledger_row(
            "calcity::1#1",
            "20260725_110001",
            1,
            metrics={"SR": 1, "intel_quality": "P", "referent_bucket": "car"},
        ),
        ledger_row("calcity::2#2", "20260725_110002", status="timed_out"),
        ledger_row("calcity::3#3", "20260725_110003", 0),
    )
    (root / "run_manifest.json").write_text(json.dumps({"split": "calcity"}))
    report(root)
    out = capsys.readouterr().out
    assert "by bucket: car:2/2(100%)   other:0/2(0%)" in out
    assert "by intel quality: G:1/2(50%)   P:1/1(100%)   ?:0/1(0%)" in out
    alone = tmp_path / "alone"
    write_ledger(
        alone,
        ledger_row(
            "calcity::9#9",
            "20260725_110000",
            1,
            metrics={"SR": 1, "intel_quality": "M", "referent_bucket": "building"},
        ),
    )
    report(alone)
    out = capsys.readouterr().out
    assert "by bucket: building:1/1(100%)" in out and "by intel quality: M:1/1(100%)" in out


def test_a_split_that_is_not_the_run_s_corpus_prints_no_class_split(tmp_path, monkeypatch, capsys):
    """Without the run's corpus its classes are unknown: no split on this machine, or a split
    lacking one of the run's episodes (another corpus of the same name), says so instead of '?'."""
    _calcity_corpus(tmp_path, monkeypatch, ("the red car", None))
    root = tmp_path / "q"
    write_ledger(
        root,
        ledger_row("calcity::0#0", "20260725_110000", 1),
        ledger_row("calcity::7#1", "20260725_110001", 1),
    )
    report(root)
    out = capsys.readouterr().out
    assert "by bucket" not in out and "intel quality" not in out
    assert (
        "by class: -- (2 of the run's 2 episodes are not in its split None on this machine)" in out
    )
    (root / "run_manifest.json").write_text(json.dumps({"split": "calcity"}))
    report(root)
    assert (
        "by class: -- (1 of the run's 2 episodes are not in its split calcity"
        in capsys.readouterr().out
    )
    write_ledger(root, ledger_row("calcity::0#0", "20260725_110000", 1))
    report(root)
    out = capsys.readouterr().out
    assert "by bucket: car:1/1(100%)" in out and "intel quality" not in out


def test_a_root_without_a_manifest_reads_its_split_off_the_rows(tmp_path, monkeypatch, capsys):
    """A released-baseline root has no run manifest; each row records its split."""
    _calcity_corpus(tmp_path, monkeypatch, ("the red car", "G"))
    root = tmp_path / "released"
    write_ledger(root, ledger_row("calcity::0#0", "20260725_110000", 1, split="calcity"))
    report(root)
    out = capsys.readouterr().out
    assert "by bucket: car:1/1(100%)" in out and "by intel quality: G:1/1(100%)" in out


def test_a_row_that_records_its_description_is_bucketed_by_it(tmp_path, monkeypatch, capsys):
    """A revised released root carries its own wording; the split row keeps the original."""
    _calcity_corpus(tmp_path, monkeypatch, ("a building", "G"))
    root = tmp_path / "revised"
    write_ledger(
        root,
        ledger_row(
            "calcity::0#0", "20260725_110000", 1, split="calcity", description="the red car"
        ),
    )
    report(root)
    out = capsys.readouterr().out
    assert "by bucket: car:1/1(100%)" in out and "by intel quality: G:1/1(100%)" in out
