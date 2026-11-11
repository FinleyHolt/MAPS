"""The CLI reaches its exit, and the arm-waiver warning names the waived keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.citynav.analysis.ab_report import main, wilson_ci
from benchmarks.run_record.tests.ledger import write_ledger


def _write_run(root: Path, sha: str, srs: dict[str, int], criterion: str = "ring") -> None:
    prov = {
        "git_commit": sha,
        "split": "val_seen",
        "search_env_flags": {"SEARCH_CITYNAV_TILES": "tiles/citynav_2m_osmv2"},
    }
    write_ledger(root, *(
        {"episode_id": eid, "status": "ok", "metrics": {"SR": sr, "success_criterion": criterion}, "provenance": prov}
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


def test_cli_refuses_runs_scored_under_different_criteria(tmp_path, monkeypatch):
    _write_run(tmp_path / "a", "aaaaaaa", {"e1": 1, "e2": 0}, criterion="ring")
    _write_run(tmp_path / "b", "aaaaaaa", {"e1": 0, "e2": 1}, criterion="shape")
    monkeypatch.setattr(sys, "argv", ["ab_report", str(tmp_path / "a"), str(tmp_path / "b")])
    with pytest.raises(SystemExit, match="the disc of unrecorded radius.*the target's own shape"):
        main()


def test_cli_refuses_ring_runs_of_two_radii_even_with_the_waiver(tmp_path, monkeypatch):
    import json

    import yaml

    for name, radius in (("a", 20), ("b", 40)):
        _write_run(tmp_path / name, "aaaaaaa", {"e1": 1, "e2": 0})
        (tmp_path / name / "run_manifest.json").write_text(json.dumps({
            "withhold": [], "config_yaml": yaml.safe_dump({"success_radius_m": radius})}))
    monkeypatch.setattr(sys, "argv", ["ab_report", str(tmp_path / "a"), str(tmp_path / "b"),
                                      "--allow-provenance-mismatch"])
    with pytest.raises(SystemExit, match="A is scored on the 20 m disc and B on the 40 m disc"):
        main()


def test_a_withhold_arm_compares_without_a_waiver_and_names_its_treatment(tmp_path, monkeypatch, capsys):
    import json

    import yaml

    for name, withhold in (("a", []), ("b", ["look"])):
        _write_run(tmp_path / name, "aaaaaaa", {"e1": 1, "e2": 0})
        rows = [json.loads(ln) for ln in (tmp_path / name / "episodes.jsonl").read_text().splitlines()]
        write_ledger(tmp_path / name, *({**r, "provenance": {**r["provenance"], "config_sha256": name}} for r in rows))
        (tmp_path / name / "run_manifest.json").write_text(json.dumps({
            "withhold": withhold, "config_yaml": yaml.safe_dump({"max_steps": 200, "withhold": withhold})}))
    monkeypatch.setattr(sys, "argv", ["ab_report", str(tmp_path / "a"), str(tmp_path / "b")])
    main()
    out = capsys.readouterr().out
    assert "A/B paired" in out and "treatment: withhold A=[] B=['look']" in out
    (tmp_path / "b" / "run_manifest.json").write_text(json.dumps({
        "withhold": ["look"], "config_yaml": yaml.safe_dump({"max_steps": 100, "withhold": ["look"]})}))
    with pytest.raises(SystemExit, match="config apart from its withholds"):
        main()


def test_wilson_ci_bounds_and_midpoint():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25  # k=0 pins the floor, hi stays real
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.75 < lo < 1.0
    lo, hi = wilson_ci(50, 100)
    assert (lo, hi) == (pytest.approx(0.4038, abs=1e-3),
                        pytest.approx(0.5962, abs=1e-3))
    assert wilson_ci(0, 0) == (pytest.approx(float("nan"), nan_ok=True),) * 2


def test_a_baseline_arm_names_its_policy_as_the_treatment(tmp_path, monkeypatch, capsys):
    import json

    import yaml

    for name, policy in (("a", None), ("b", "visa")):
        _write_run(tmp_path / name, "aaaaaaa", {"e1": 1, "e2": 0})
        config = {"max_steps": 200, "withhold": [], **({"policy": policy} if policy else {})}
        (tmp_path / name / "run_manifest.json").write_text(json.dumps({
            "withhold": [], "config_yaml": yaml.safe_dump(config)}))
    monkeypatch.setattr(sys, "argv", ["ab_report", str(tmp_path / "a"), str(tmp_path / "b")])
    main()
    out = capsys.readouterr().out
    assert "treatment: policy A=None B=visa" in out and "withhold A=" not in out
