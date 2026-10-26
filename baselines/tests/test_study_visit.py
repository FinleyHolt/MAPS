"""The visit page's walk, read off the results on disk: a label must be on the plan, and it
can change until session 1 starts; an error shows once; each session opens only in its turn and says its condition;
a session is over when its clock runs out or every listed report has an answer; the results
file carries the answers and no account names; the briefing renders as HTML."""

import io
import json
import os
import re
import time
import zipfile

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
    page = v.page()
    assert v.participant is None and "That is not a participant label in this study. Labels look like p07" in page
    assert "p99" not in page and "placeholder='like p07'" in page
    v.set_label("  ")
    assert "Type your label, for example p07." in v.page()
    v.set_label(" P01 ")
    assert v.participant == "p01"
    assert [v.condition(s) for s in STEPS] == ["practice", "solo", "team"]
    page = v.page()
    assert "Step 3: session 1 (alone, 40 minutes)" in page and "Opens after practice" in page
    assert "Practice ends after the fourth report" in page and "shift-click measuring" in page
    assert v.start("first") is None and v.start("second") is None
    assert v.start("practice") == "practice"
    (tmp_path / "results" / "p01" / "practice").mkdir(parents=True)
    assert "Start session 1" in v.page() and "Opening..." in v.page() and v.start("first") == "solo"
    _session(tmp_path / "results" / "p01" / "solo")
    page = v.page()
    assert v.status("first") == "running" and "Return to session 1" in page
    assert "Session 1 (alone) is under way" in page and "About 40 minutes left." in page
    assert "kept running while you were away" not in page
    assert v.start("practice") is None
    _session(tmp_path / "results" / "p01" / "solo", age_s=2350)
    assert "About 1 minute left." in v.page()
    _session(tmp_path / "results" / "p01" / "solo", age_s=2401)
    assert v.status("first") == "done" and v.start("first") is None
    page = v.page()
    assert "five-minute break" in page and "Then start session 2 (with MAPS, 40 minutes)." in page
    assert "<span id=countdown>5:00</span>" in page and "<p id=break hidden>" in page
    assert v.start("second") == "team"


def test_a_session_ends_when_every_report_has_an_answer(tmp_path) -> None:
    v = Visit(_kit(tmp_path))
    v.set_label("p01")
    solo = tmp_path / "results" / "p01" / "solo"
    _session(solo, answers='{"episode_id": "a"}\n{"episode_id": "a"}\n{"episode_id": "b"}\n{"episode_id": "c"')
    assert v.status("first") == "running"
    _session(solo, answers='{"episode_id": "a"}\n{"episode_id": "b"}\n{"episode_id": "c"}\n')
    assert v.status("first") == "done"


def test_a_label_changes_until_session_1_starts(tmp_path) -> None:
    (_kit(tmp_path) / "results" / "p02" / "practice").mkdir(parents=True)
    v = Visit(tmp_path)
    page = v.page()
    assert v.participant == "p02" and "Welcome back, p02" in page and "Not your label? Change it" in page
    assert "placeholder='like p07'" in v.page(change=True)
    v.set_label("p01")
    assert v.participant == "p01" and not (tmp_path / "results" / "p02").exists()
    assert "Welcome back" not in v.page() and "Participant p01" in v.page()
    _session(tmp_path / "results" / "p01" / "solo")
    v = Visit(tmp_path)
    assert "Change it" not in v.page() and "placeholder" not in v.page(change=True)
    v.set_label("p02")
    msg = "You started as p01. A different label needs a fresh copy of the study: ask the study lead."
    assert v.participant == "p01" and msg in v.page()
    assert msg not in v.page()


def test_the_clock_kept_running_only_after_a_relaunch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study_server, "open_session", lambda kit, participant, condition: condition)
    (_kit(tmp_path) / "results" / "p01" / "practice").mkdir(parents=True)
    _session(tmp_path / "results" / "p01" / "solo")
    v = Visit(tmp_path)
    assert "Its clock kept running while you were away." in v.page()
    assert v.start("first") == "solo"
    assert "kept running" not in v.page()


def test_minutes_from_the_sessions_own_clock(tmp_path, monkeypatch) -> None:
    (_kit(tmp_path) / "results" / "p01" / "practice").mkdir(parents=True)
    _session(tmp_path / "results" / "p01" / "solo")
    monkeypatch.setenv("MAPS_STUDY_SESSION_S", "600")
    v = Visit(tmp_path)
    assert v.name("first") == "session 1 (alone, 40 minutes)"
    assert v.name("second") == "session 2 (with MAPS, 10 minutes)"


def test_results_file(tmp_path) -> None:
    v = Visit(_kit(tmp_path, contact="lead@example.org"))
    assert v.results_zip() is None
    v.set_label("p01")
    assert v.results_zip() is None
    (tmp_path / "results" / "p01" / "practice").mkdir(parents=True)
    _session(tmp_path / "results" / "p01" / "solo", age_s=2401, answers='{"episode_id": "a"}\n')
    _session(tmp_path / "results" / "p01" / "team", age_s=2401)
    page = v.page()
    assert "Save my results file" in page and "maps-study-p01.zip" in page and "lead@example.org" in page
    assert "the text window (on a Mac, press Terminate if asked)." in page
    assert ("<p id=saved hidden>Saved maps-study-p01.zip, usually in your Downloads folder. "
            "Now email it to lead@example.org.</p>") in page
    members = zipfile.ZipFile(io.BytesIO(v.results_zip())).infolist()
    assert "results/p01/solo/answers.jsonl" in {m.filename for m in members}
    assert all(m.extra == b"" for m in members)
    os.utime(tmp_path / "results" / "p01" / "solo" / "answers.jsonl", (0, 0))
    assert v.results_zip()
    (tmp_path / "contact.txt").unlink()
    assert "the study lead" in Visit(tmp_path).page()


def test_briefing_markdown(tmp_path) -> None:
    v = Visit(_kit(tmp_path))
    v.set_label("p01")
    page = v.page()
    assert "<code>Lime Avenue &amp; 88th Street</code>" in page
    assert not re.search(r"</li>\s*<ul>", page)
