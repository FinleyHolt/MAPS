"""The QC's ledger checks: a clean ledger passes; a duplicate, a late answer and an off-list report each fail."""

import json

from baselines.human.study_qc import checks


def _write(tmp_path, answers) -> None:
    (tmp_path / "session.json").write_text(json.dumps({"started_epoch": 1_800_000_000.0, "session_s": 2400.0,
                                                       "order": ["a", "b", "c"]}))
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))


def test_ledger_checks(tmp_path) -> None:
    on_time, late = "2027-01-15T08:00:00Z", "2027-01-15T09:00:00Z"   # the session ends 08:40:00Z
    _write(tmp_path, [{"episode_id": "a", "answered_at": on_time}, {"episode_id": "b", "answered_at": on_time}])
    assert checks(tmp_path) == []
    _write(tmp_path, [{"episode_id": "a", "answered_at": on_time}, {"episode_id": "a", "answered_at": late},
                      {"episode_id": "z", "answered_at": on_time}])
    assert len(checks(tmp_path)) == 3
    (tmp_path / "session.json").unlink()
    assert "clock" in checks(tmp_path)[0]
