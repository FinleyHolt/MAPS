"""The visit page's walk, read off the results on disk: a label must be on the plan, and it
can change until the session starts; an error shows once; the session opens only after practice and says the
participant's condition;
a session is over when its clock runs out or every listed report has an answer; the results
file carries the answers and no account names; the briefing renders as HTML."""

import json
import os
import re
import time

from baselines.human import study_server
from baselines.human.study_server import STEPS, Visit


def _kit(tmp_path, contact: str | None = None):
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "plan.json").write_text(json.dumps({"participants": {"p01": {"arm": "solo"},
                                                                              "p02": {"arm": "team"}}}))
    if contact:
        (tmp_path / "contact.txt").write_text(contact + "\n")
    return tmp_path


def _session(folder, age_s: float = 0.0, answers: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "session.json").write_text(json.dumps({"started_epoch": time.time() - age_s, "session_s": 3600.0,
                                                     "episodes_n": 3}))
    (folder / "answers.jsonl").write_text(answers)


def test_visit_steps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study_server, "open_session", lambda kit, participant, condition: condition)
    v = Visit(_kit(tmp_path))
    v.set_label("p99")
    page = v.page()
    assert v.participant is None and "That is not a participant label in this study. Labels look like p07" in page
    assert "value='p99'" in page and page.count("p99") == 1 and "placeholder='like p07'" in page
    v.set_label("  ")
    assert "Type your label, for example p07." in v.page()
    v.set_label(" P01 ")
    assert v.participant == "p01"
    assert [v.condition(s) for s in STEPS] == ["practice", "solo"]
    page = v.page()
    assert "Step 3: your session (alone, 60 minutes)" in page and "Opens after practice" in page
    assert "Practice ends after the fourth report" in page and "shift-click measuring" in page
    assert v.start("session") is None
    assert v.start("practice") == "practice"
    (tmp_path / "results" / v.run / "practice").mkdir(parents=True)
    page = v.page()
    # Back from practice, the session's button leads the page, above the briefing.
    assert page.index("Practice done. Next: your session (alone, 60 minutes)") < page.index("Start your session") < page.index("The task")
    assert "Opening..." in page and v.start("session") == "solo"
    _session(tmp_path / "results" / v.run / "solo")
    page = v.page()
    assert v.status("session") == "running" and "Return to the session" in page
    assert "Your session (alone) is under way" in page and "About 60 minutes left." in page
    assert "kept running while you were away" not in page
    assert v.start("practice") is None
    _session(tmp_path / "results" / v.run / "solo", age_s=3550)
    assert "About 1 minute left." in v.page()
    _session(tmp_path / "results" / v.run / "solo", age_s=3601)
    assert v.status("session") == "done" and v.start("session") is None and v.start("practice") is None
    page = v.page()
    assert "Save my results file" in page and "break" not in page and "Start" not in page
    assert "The task" not in page  # the briefing is done with once the session is


def test_a_session_ends_when_every_report_has_an_answer(tmp_path) -> None:
    v = Visit(_kit(tmp_path))
    v.set_label("p01")
    solo = tmp_path / "results" / v.run / "solo"
    _session(solo, answers='{"episode_id": "a"}\n{"episode_id": "a"}\n{"episode_id": "b"}\n{"episode_id": "c"')
    assert v.status("session") == "running"
    _session(solo, answers='{"episode_id": "a"}\n{"episode_id": "b"}\n{"episode_id": "c"}\n')
    assert v.status("session") == "done"


def test_a_label_changes_until_the_session_starts(tmp_path) -> None:
    (_kit(tmp_path) / "results" / "p02" / "practice").mkdir(parents=True)
    v = Visit(tmp_path)
    page = v.page()
    assert v.participant == "p02" and "Welcome back, p02" in page and "Not your label? Change it" in page
    assert "placeholder='like p07'" in v.page(change=True)
    v.set_label("p01")
    assert v.participant == "p01" and not (tmp_path / "results" / "p02").exists()
    assert "Welcome back" not in v.page() and "Participant p01" in v.page()
    _session(tmp_path / "results" / v.run / "solo")
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
    assert v.start("session") == "solo"
    assert "kept running" not in v.page()


def test_minutes_from_the_sessions_own_clock(tmp_path, monkeypatch) -> None:
    (_kit(tmp_path) / "results" / "p01" / "practice").mkdir(parents=True)
    _session(tmp_path / "results" / "p01" / "solo")
    monkeypatch.setenv("MAPS_STUDY_SESSION_S", "600")
    assert Visit(tmp_path).name("session") == "your session (alone, 60 minutes)"
    (tmp_path / "fresh").mkdir()
    v = Visit(_kit(tmp_path / "fresh"))
    v.set_label("p02")
    assert v.name("session") == "your session (with MAPS, 10 minutes)"


def test_results_file(tmp_path) -> None:
    v = Visit(_kit(tmp_path, contact="lead@example.org"))
    assert v.results_json() is None
    v.set_label("p01")
    assert v.results_json() is None
    (tmp_path / "results" / v.run / "practice").mkdir(parents=True)
    _session(tmp_path / "results" / v.run / "solo", age_s=3601, answers='{"episode_id": "a"}\ntorn\n')
    page = v.page()
    assert "Save my results file" in page and f"maps-study-{v.run}.json" in page and "lead@example.org" in page
    assert "/visit/results.json" in page and ".zip" not in page
    assert "the text window (on a Mac, press Terminate if asked)." in page
    assert "<p id=saved hidden>Your browser is saving the file.</p>" in page
    data = v.results_json()
    doc = json.loads(data)
    assert (doc["format"], doc["version"], doc["run_id"], doc["participant"]) == ("maps-capacity-results", 1, v.run, "p01")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc["saved_at"])
    assert list(doc) == sorted(doc) and list(doc["sessions"]) == ["solo"]
    solo = doc["sessions"]["solo"]
    assert solo["answers"] == [{"episode_id": "a"}] and solo["unreadable_lines"] == 1 and solo["session"]["episodes_n"] == 3
    assert str(tmp_path) not in data.decode() and os.path.basename(os.path.expanduser("~")) not in data.decode()
    again = v.results_json()
    assert json.loads(again) | {"saved_at": 0} == doc | {"saved_at": 0}
    assert again.decode().replace(json.loads(again)["saved_at"], "") == data.decode().replace(doc["saved_at"], "")
    (tmp_path / "contact.txt").unlink()
    assert "the study lead" in Visit(tmp_path).page()


def test_briefing_markdown(tmp_path) -> None:
    v = Visit(_kit(tmp_path))
    v.set_label("p01")
    page = v.page()
    assert "<code>Lime Avenue &amp; 88th Street</code>" in page and "<table>" in page
    assert not re.search(r"</li>\s*<ul>", page)
    # Only the participant's own session kind, at the session's length, inside step 2.
    assert "Your session: alone" in page and "Your session: with MAPS" not in page and "{MINUTES}" not in page
    assert page.index("read the briefing") < page.index("Your session: alone") < page.index("practice</h2>")
    v.set_label("p02")
    assert "Your session: with MAPS" in v.page() and "Your session: alone" not in v.page()


def test_a_mistyped_label_stays_in_the_box(tmp_path) -> None:
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "plan.json").write_text(json.dumps({"participants": {"p01": {"arm": "team"}}}))
    v = Visit(tmp_path)
    v.set_label("p83'")
    assert "value='p83&#x27;'" in v.page()
    assert "value=''" in v.page()  # with its error, once
    v.set_label("p01")
    assert v.typed == ""


def test_each_visit_gets_its_own_run_id_under_its_label(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(study_server, "open_session", lambda kit, run, condition: (run, condition))
    v = Visit(_kit(tmp_path))
    v.set_label("P01")
    run = v.run
    assert re.fullmatch(r"p01-[0-9a-f]{8}", run) and v.participant == "p01"
    other = Visit(tmp_path)
    other.set_label("p01")
    assert other.run != run
    v.set_label("p01")
    assert v.run == run and v.start("practice") == (run, "practice")
    page = v.page()
    assert "Participant p01" in page and run in page
    (tmp_path / "results" / run / "practice").mkdir(parents=True)
    v.set_label("p02")
    assert re.fullmatch(r"p02-[0-9a-f]{8}", v.run) and not (tmp_path / "results" / run).exists()
    assert v.condition("session") == "team"
    _session(tmp_path / "results" / v.run / "team")
    back = Visit(tmp_path)
    assert (back.run, back.participant) == (v.run, "p02") and "Welcome back, p02" in back.page()
    back.set_label("p01")
    assert back.run == v.run


def test_a_session_opens_the_labels_list_and_saves_under_the_run_id(tmp_path, monkeypatch) -> None:
    from baselines.human import audit_console, capacity

    opened = {}
    monkeypatch.setattr(audit_console, "bench_labels", lambda: {})
    monkeypatch.setattr(capacity, "Capacity", lambda root, order, out, arm, **kw: opened.update(order=order, out=out, **kw))
    (_kit(tmp_path) / "plan" / "p01").mkdir()
    (tmp_path / "plan" / "p01" / "solo.txt").write_text("a\nb\n")
    study_server.open_session(tmp_path, "p01-3f9a2c1e", "solo")
    assert opened["order"] == ["a", "b"] and opened["out"] == tmp_path / "results" / "p01-3f9a2c1e" / "solo"
    assert opened["ids"] == {"run_id": "p01-3f9a2c1e", "participant": "p01"}


def test_a_roster_lists_first_names_and_keeps_the_label_in_the_results(tmp_path) -> None:
    kit = _kit(tmp_path)
    (kit / "roster.json").write_text(json.dumps({"p01": "Zoe", "p02": "Ann", "p09": "Off the plan"}))
    v = Visit(kit)
    page = v.page()
    assert "<select name=label" in page and "Select who you are from the list" in page and "Off the plan" not in page
    assert page.index(">Ann<") < page.index(">Zoe<") and "p01" not in page.split("<select")[0]
    v.set_label("")
    assert "Choose your first name." in v.page()
    v.set_label("p02")
    page = v.page()
    assert v.participant == "p02" and "Welcome, Ann" in page and "Not you? Change it" in page
    _session(tmp_path / "results" / v.run / "team", age_s=3601)
    assert json.loads(v.results_json())["participant"] == "p02" and b"Ann" not in v.results_json()
