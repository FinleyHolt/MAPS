"""The QC's ledger checks: a clean ledger passes; a duplicate, a late answer, an off-list report,
a torn line and a clock set back each fail. The QC scores what a kit's results hold and refuses
a label the kit's plan does not know."""

import json

import pytest

from baselines.human import study_qc
from baselines.human.study_qc import checks, summary
from baselines.tests.test_human_study import _root


def _write(tmp_path, answers) -> None:
    (tmp_path / "session.json").write_text(json.dumps({"started_epoch": 1_800_000_000.0, "session_s": 2400.0,
                                                       "started_at": "2027-01-15T08:00:00Z", "order": ["a", "b", "c"]}))
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))


def test_ledger_checks(tmp_path) -> None:
    on_time, late = "2027-01-15T08:00:00Z", "2027-01-15T09:00:00Z"   # the session ends 08:40:00Z
    _write(tmp_path, [{"episode_id": "a", "answered_at": on_time}, {"episode_id": "b", "answered_at": on_time}])
    assert checks(tmp_path) == []
    _write(tmp_path, [{"episode_id": "a", "answered_at": on_time}, {"episode_id": "a", "answered_at": late},
                      {"episode_id": "z", "answered_at": late}])
    assert len(checks(tmp_path)) == 3
    (tmp_path / "session.json").unlink()
    assert "clock" in checks(tmp_path)[0]


def test_a_torn_line_and_a_clock_set_back_fail_the_checks(tmp_path) -> None:
    _write(tmp_path, [{"episode_id": "a", "answered_at": "2027-01-15T08:10:00Z"}])
    with (tmp_path / "answers.jsonl").open("a") as fh:
        fh.write('{"episode_id": "b", "answ')
    assert checks(tmp_path) == ["1 unreadable lines in answers.jsonl"]
    _write(tmp_path, [{"episode_id": "a", "answered_at": "2027-01-15T07:59:00Z"}])
    assert checks(tmp_path) == ["1 answers before the session's start or before the answer ahead of them: the clock moved back"]
    _write(tmp_path, [{"episode_id": "a", "answered_at": "2027-01-15T08:10:00Z"},
                      {"episode_id": "b", "answered_at": "2027-01-15T08:05:00Z"}])
    assert len(checks(tmp_path)) == 1


def _kit(tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    root = _root(kit).rename(kit / "maps_root")
    row = json.loads((root / "episodes.jsonl").read_text())
    (root / "episodes.jsonl").write_text(json.dumps({**row, "elapsed_s": 100.0, "metrics": {**row["metrics"], "SR": 1}}) + "\n")
    (kit / "plan").mkdir()
    (kit / "plan" / "plan.json").write_text(json.dumps({"participants": {"p04": {}}}))
    team = kit / "results" / "p04" / "team"
    team.mkdir(parents=True)
    (team / "session.json").write_text(json.dumps({"arm": "team", "agents": 1, "session_s": 3600.0, "started_epoch": 1000.0,
                                                   "started_at": "1970-01-01T00:16:40Z", "episodes_n": 1,
                                                   "order": ["block_0::1#0"]}))
    solo = kit / "results" / "p04" / "solo"
    solo.mkdir()
    (solo / "answers.jsonl").write_text(json.dumps({"episode_id": "block_0::1#0", "source": "skip"}) + "\n")
    monkeypatch.setattr(study_qc, "use_kit", lambda k: None)
    return kit


def test_an_idle_team_session_scores_maps_results_and_a_lost_clock_is_reported(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    assert study_qc.main(["P04", "--kit", str(kit)]) == 1
    out = capsys.readouterr().out
    assert "solo: CHECK FAILED: no session.json" in out and "team: 1/1 correct" in out
    scored = kit / "scored" / "p04-team"
    assert json.loads((scored / "session.json").read_text())["session_s"] == 3600.0
    s = summary(scored)
    assert (s["n"], s["correct"], s["hours"], s["sources"], s["misses"]) == (1, 1, 1.0, {"unchecked": (1, 1)}, [])


def test_a_label_off_the_plan_is_refused(tmp_path, monkeypatch) -> None:
    kit = _kit(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="p99 is not a participant on this kit's plan"):
        study_qc.main(["p99", "--kit", str(kit)])
