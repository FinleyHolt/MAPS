"""The QC's ledger checks: a clean ledger passes; a duplicate, a late answer, an off-list report,
a torn line, a clock set back and a rehearsal's length each fail, and so does a participant without
the session of their condition. The QC scores the plan's condition for the run and refuses a label the kit's
plan does not know."""

import json
import shutil

import pytest

from baselines.human import study_qc
from baselines.human.study_qc import checks, summary
from baselines.tests.test_human_study import _root


def _write(tmp_path, answers, session_s: float = 3600.0) -> None:
    (tmp_path / "session.json").write_text(json.dumps({"started_epoch": 1_800_000_000.0, "session_s": session_s,
                                                       "started_at": "2027-01-15T08:00:00Z", "order": ["a", "b", "c"]}))
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))


def test_ledger_checks(tmp_path) -> None:
    on_time, late = "2027-01-15T08:00:00Z", "2027-01-15T09:20:00Z"   # the session ends 09:00:00Z
    _write(tmp_path, [{"episode_id": "a", "answered_at": on_time}, {"episode_id": "b", "answered_at": on_time}])
    assert checks(tmp_path) == []
    _write(tmp_path, [{"episode_id": "a", "answered_at": on_time}, {"episode_id": "a", "answered_at": late},
                      {"episode_id": "z", "answered_at": late}])
    assert len(checks(tmp_path)) == 3
    _write(tmp_path, [], session_s=600.0)
    assert checks(tmp_path) == ["session length 10 min, not 60: a rehearsal?"]
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
    (kit / "plan" / "plan.json").write_text(json.dumps({"participants": {"p04": {"arm": "team"}, "p05": {"arm": "solo"}}}))
    team = kit / "results" / "p04" / "team"
    team.mkdir(parents=True)
    (team / "session.json").write_text(json.dumps({"arm": "team", "agents": 1, "session_s": 7200.0, "started_epoch": 1000.0,
                                                   "started_at": "1970-01-01T00:16:40Z", "episodes_n": 1,
                                                   "order": ["block_0::1#0"]}))
    solo = kit / "results" / "p05" / "solo"
    solo.mkdir(parents=True)
    (solo / "answers.jsonl").write_text(json.dumps({"episode_id": "block_0::1#0", "source": "skip"}) + "\n")
    monkeypatch.setattr(study_qc, "use_kit", lambda k: None)
    # A toy root: graded on its own rows, not the study's frozen targets.
    monkeypatch.setattr(study_qc, "STUDY_TARGETS", None)
    return kit


def test_an_idle_team_session_scores_maps_results(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    assert study_qc.main(["P04", "--kit", str(kit)]) == 1
    out = capsys.readouterr().out
    assert "team: CHECK FAILED: session length 120 min, not 60: a rehearsal?" in out and "team: 1/1 correct" in out
    assert "solo" not in out
    scored = kit / "scored" / "p04-team"
    assert json.loads((scored / "session.json").read_text())["session_s"] == 7200.0
    s = summary(scored)
    assert (s["n"], s["correct"], s["hours"], s["sources"], s["misses"]) == (1, 1, 2.0, {"unchecked": (1, 1)}, [])


def test_a_lost_clock_is_reported(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    assert study_qc.main(["p05", "--kit", str(kit)]) == 1
    out = capsys.readouterr().out
    assert "solo: CHECK FAILED: no session.json" in out and "p05 has no session to score" in out and "team" not in out


def test_a_participant_with_their_session_passes(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    meta = kit / "results" / "p04" / "team" / "session.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "session_s": 3600.0}))
    assert study_qc.main(["p04", "--kit", str(kit)]) == 0
    out = capsys.readouterr().out
    assert "team: 1/1 correct" in out and "FAILED" not in out and "solo" not in out


def test_a_session_in_the_condition_the_plan_did_not_deal_fails(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    meta = kit / "results" / "p04" / "team" / "session.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "session_s": 3600.0}))
    shutil.copytree(kit / "results" / "p04" / "team", kit / "results" / "p04" / "solo")
    assert study_qc.main(["p04", "--kit", str(kit)]) == 1
    assert "team: CHECK FAILED: a solo session, not this participant's condition" in capsys.readouterr().out


def test_a_rehearsal_labels_run_fails_so_collection_flags_it(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    plan = json.loads((kit / "plan" / "plan.json").read_text())
    plan["participants"]["p04"]["test"] = True
    (kit / "plan" / "plan.json").write_text(json.dumps(plan))
    meta = kit / "results" / "p04" / "team" / "session.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "session_s": 3600.0}))
    assert study_qc.main(["p04", "--kit", str(kit)]) == 1
    assert "team: CHECK FAILED: a rehearsal label, not a participant" in capsys.readouterr().out


def test_a_participant_without_their_session_fails(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    shutil.rmtree(kit / "results" / "p04" / "team")
    assert study_qc.main(["p04", "--kit", str(kit)]) == 1
    out = capsys.readouterr().out
    assert "team: CHECK FAILED: no session yet" in out and "p04 has no session to score" in out


def test_a_label_off_the_plan_is_refused(tmp_path, monkeypatch) -> None:
    kit = _kit(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="p99 is not a participant on this kit's plan"):
        study_qc.main(["p99", "--kit", str(kit)])


def test_a_run_id_or_its_one_label_scores_under_the_run_id(tmp_path, monkeypatch, capsys) -> None:
    kit = _kit(tmp_path, monkeypatch)
    (kit / "results" / "p04").rename(kit / "results" / "p04-3f9a2c1e")
    for arg in ("p04-3f9a2c1e", "p04"):
        assert study_qc.main([arg, "--kit", str(kit)]) == 1
        assert "team: 1/1 correct" in capsys.readouterr().out
        assert (kit / "scored" / "p04-3f9a2c1e-team" / "episodes.jsonl").is_file()


def test_a_label_with_two_visits_asks_for_one(tmp_path, monkeypatch) -> None:
    kit = _kit(tmp_path, monkeypatch)
    (kit / "results" / "p04").rename(kit / "results" / "p04-3f9a2c1e")
    (kit / "results" / "p04-0b1c2d3e").mkdir()
    with pytest.raises(SystemExit, match="p04 has 2 visits: p04-0b1c2d3e, p04-3f9a2c1e; name one"):
        study_qc.main(["p04", "--kit", str(kit)])
