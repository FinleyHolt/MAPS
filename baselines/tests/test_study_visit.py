"""The visit page's walk, read off the results on disk: a label must be on the plan, and a
relaunch keeps the one it finds; each session opens only in its turn and says its condition;
a session is over when its clock runs out or every listed report has an answer; the results
file carries the answers and no account names; the briefing renders as HTML."""

import io
import json
import re
import tarfile
import time

from baselines.human import study_server
from baselines.human.study_server import STEPS, Visit


def _kit(tmp_path, contact: str | None = None):
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "plan.json").write_text(json.dumps({"participants": {"p01": {"first": "solo"},
                                                                              "p02": {"first": "team"}}}))
    if contact:
        (tmp_path / "contact.txt").write_text(contact + "\n")
    return tmp_path


def _session(folder, age_s: float = 0.0, answers: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "session.json").write_text(json.dumps({"started_epoch": time.time() - age_s, "session_s": 2400.0,
                                                     "episodes_n": 3}))
    (folder / "answers.jsonl").write_text(answers)


def test_visit_steps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study_server, "open_session", lambda kit, participant, condition: condition)
    v = Visit(_kit(tmp_path))
    v.set_label("p99")
    assert v.participant is None and "not a participant" in v.page()
    v.set_label(" P01 ")
    assert v.participant == "p01"
    assert [v.condition(s) for s in STEPS] == ["practice", "solo", "team"]
    assert "Session 1: alone, 40 minutes" in v.page() and "Opens after practice" in v.page()
    assert v.start("first") is None and v.start("second") is None
    assert v.start("practice") == "practice"
    (tmp_path / "results" / "p01" / "practice").mkdir(parents=True)
    assert "Start session 1" in v.page() and v.start("first") == "solo"
    _session(tmp_path / "results" / "p01" / "solo")
    page = v.page()
    assert v.status("first") == "running" and "Return to session 1" in page
    assert "40 minutes left" in page and "kept running while you were away" in page
    assert v.start("practice") is None
    _session(tmp_path / "results" / "p01" / "solo", age_s=2401)
    assert v.status("first") == "done" and v.start("first") is None
    assert "five-minute break" in v.page() and "Session 2: with MAPS, 40 minutes" in v.page()
    assert v.start("second") == "team"


def test_a_session_ends_when_every_report_has_an_answer(tmp_path) -> None:
    v = Visit(_kit(tmp_path))
    v.set_label("p01")
    solo = tmp_path / "results" / "p01" / "solo"
    _session(solo, answers='{"episode_id": "a"}\n{"episode_id": "a"}\n{"episode_id": "b"}\n{"episode_id": "c"')
    assert v.status("first") == "running"
    _session(solo, answers='{"episode_id": "a"}\n{"episode_id": "b"}\n{"episode_id": "c"}\n')
    assert v.status("first") == "done"


def test_a_relaunch_keeps_the_label(tmp_path) -> None:
    (_kit(tmp_path) / "results" / "p02" / "practice").mkdir(parents=True)
    v = Visit(tmp_path)
    assert v.participant == "p02"
    v.set_label("p01")
    assert v.participant == "p02" and "set up for p02" in v.page()


def test_results_file(tmp_path) -> None:
    v = Visit(_kit(tmp_path, contact="lead@example.org"))
    assert v.results_tgz() is None
    v.set_label("p01")
    assert v.results_tgz() is None
    (tmp_path / "results" / "p01" / "practice").mkdir(parents=True)
    _session(tmp_path / "results" / "p01" / "solo", age_s=2401, answers='{"episode_id": "a"}\n')
    _session(tmp_path / "results" / "p01" / "team", age_s=2401)
    page = v.page()
    assert "Save my results file" in page and "maps-study-p01.tgz" in page and "lead@example.org" in page
    members = tarfile.open(fileobj=io.BytesIO(v.results_tgz())).getmembers()
    assert "results/p01/solo/answers.jsonl" in {m.name for m in members}
    assert all(m.uname == m.gname == "" and m.uid == m.gid == 0 for m in members)
    (tmp_path / "contact.txt").unlink()
    assert "the study lead" in Visit(tmp_path).page()


def test_briefing_markdown(tmp_path) -> None:
    v = Visit(_kit(tmp_path))
    v.set_label("p01")
    page = v.page()
    assert "<code>Lime Avenue &amp; 88th Street</code>" in page
    assert not re.search(r"</li>\s*<ul>", page)
