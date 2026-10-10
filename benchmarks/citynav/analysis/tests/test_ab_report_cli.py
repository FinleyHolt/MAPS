"""The CLI reaches its exit, and the arm-waiver warning names the waived keys.

The comparability guards were unit-pinned, but ``main()`` itself never ran
under test — so when the fatal/arm split renamed the mismatch list, the
post-report warning kept reading the retired name and every CLI invocation
died on NameError after printing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.citynav.analysis.ab_report import main, wilson_ci


def _write_run(root: Path, sha: str, srs: dict[str, int]) -> None:
    root.mkdir(parents=True)
    prov = {
        "git_commit": sha,
        "split": "val_seen",
        "search_env_flags": {"SEARCH_CITYNAV_TILES": "tiles/citynav_2m_osmv2"},
    }
    rows = [
        {"episode_id": eid, "status": "ok", "metrics": {"SR": sr}, "provenance": prov}
        for eid, sr in srs.items()
    ]
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )


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


def test_sr_by_episode_uses_canonical_first_complete_pick(tmp_path):
    """A duplicated episode reads as its FIRST complete attempt (by start
    stamp), not the last row in the file."""
    from benchmarks.citynav.analysis.ab_report import _sr_by_episode

    root = tmp_path / "run"
    root.mkdir()
    later = {"episode_id": "e1", "status": "ok", "metrics": {"SR": 0},
             "workdir": "w0/maps_20260801_110000_citynav_e1"}
    earlier = {"episode_id": "e1", "status": "ok", "metrics": {"SR": 1},
               "workdir": "w0/maps_20260801_100000_citynav_e1"}
    # File order is completion order: the re-fly (later start) landed first.
    (root / "episodes.jsonl").write_text(
        json.dumps(later) + "\n" + json.dumps(earlier) + "\n"
    )
    assert _sr_by_episode(root) == {"e1": 1}


def test_the_served_alias_is_not_the_served_build():
    """`inkling-small` named two different quantizations on the same cluster
    in the same week — `thinkingmachines/Inkling-Small-NVFP4` and
    `RedHatAI/Inkling-Small-FP8-dynamic`. On the ALIAS alone the pair reads
    as same-model, and the ledger's null verb quoted the resulting -8.7 pts
    at McNemar p=0.009 as what IDENTICAL code does. The build is an arm."""
    from benchmarks.citynav.analysis.ab_report import comparability_issues

    gt = {"split": "train_seen",
          "search_env_flags": {"SEARCH_CITYNAV_TILES": "/x/tiles/citynav_2m_osmv3"}}
    a = {**gt, "served_model": "inkling-small",
         "served_build": "thinkingmachines/Inkling-Small-NVFP4"}
    b = {**gt, "served_model": "inkling-small",
         "served_build": "RedHatAI/Inkling-Small-FP8-dynamic"}
    fatal, arm = comparability_issues(a, b)
    assert not fatal
    assert any("BUILD" in m for m in arm), arm
    # Absence-tolerant, or every column stored before the key existed would
    # stop pairing against every column stored after it.
    fatal2, arm2 = comparability_issues(a, {**gt, "served_model": "inkling-small"})
    assert not fatal2 and not arm2
