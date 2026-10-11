"""The CLI reaches its exit, and the arm-waiver warning names the waived keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.citynav.analysis.ab_report import main, wilson_ci
from benchmarks.citynav.analysis.tests.ledger import write_ledger


def _write_run(root: Path, sha: str, srs: dict[str, int]) -> None:
    prov = {
        "git_commit": sha,
        "split": "val_seen",
        "search_env_flags": {"SEARCH_CITYNAV_TILES": "tiles/citynav_2m_osmv2"},
    }
    write_ledger(root, *(
        {"episode_id": eid, "status": "ok", "metrics": {"SR": sr}, "provenance": prov}
        for eid, sr in srs.items()))


def test_cli_completes_and_warns_on_waived_arm_mismatch(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path / "a", "aaaaaaa", {"e1": 1, "e2": 0, "e3": 1})
    _write_run(tmp_path / "b", "bbbbbbb", {"e1": 1, "e2": 1, "e3": 0})
    monkeypatch.setattr(
        sys,
        "argv",
        ["ab_report", str(tmp_path / "a"), str(tmp_path / "b"),
         "--allow-provenance-mismatch"],
    )
    main()
    out = capsys.readouterr().out
    assert "A/B paired" in out
    assert "override active" in out and "git_commit" in out


def test_cli_completes_silently_comparable(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path / "a", "aaaaaaa", {"e1": 1, "e2": 0})
    _write_run(tmp_path / "b", "aaaaaaa", {"e1": 0, "e2": 1})
    monkeypatch.setattr(
        sys, "argv", ["ab_report", str(tmp_path / "a"), str(tmp_path / "b")]
    )
    main()
    out = capsys.readouterr().out
    assert "A/B paired" in out
    assert "override active" not in out


def test_wilson_ci_bounds_and_midpoint():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25  # k=0 pins the floor, hi stays real
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.75 < lo < 1.0
    lo, hi = wilson_ci(50, 100)
    assert (lo, hi) == (pytest.approx(0.4038, abs=1e-3),
                        pytest.approx(0.5962, abs=1e-3))
    assert wilson_ci(0, 0) == (pytest.approx(float("nan"), nan_ok=True),) * 2
