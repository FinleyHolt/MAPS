"""The visit page's walk, read off the results on disk: a label must be on the plan; practice
opens session 1; a session's clock decides running or done; the briefing renders as HTML."""

import json
import time

from baselines.human.study_server import Visit, md_html


def test_visit_steps(tmp_path) -> None:
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "plan.json").write_text(json.dumps({"participants": {"p01": {"first": "team"}}}))
    v = Visit(tmp_path, None)
    v.set_label("p99")
    assert v.participant is None and "not a participant" in v.page()
    v.set_label("p01")
    assert [v.condition(s) for s in ("practice", "first", "second")] == ["practice", "team", "solo"]
    assert "Opens after practice" in v.page() and v.start("second") is None
    (tmp_path / "results" / "p01" / "practice").mkdir(parents=True)
    assert "Start session 1" in v.page()
    team = tmp_path / "results" / "p01" / "team"
    team.mkdir()
    session = {"started_epoch": time.time(), "session_s": 2400.0, "episodes_n": 3}
    (team / "session.json").write_text(json.dumps(session))
    assert v.status("first") == "running" and "Return to session 1" in v.page()
    (team / "session.json").write_text(json.dumps({**session, "started_epoch": time.time() - 2401}))
    assert v.status("first") == "done" and "five-minute break" in v.page() and v.start("first") is None


def test_briefing_markdown() -> None:
    out = md_html("# Title\n\nSome **bold** and `code`\nwrapped.\n\n- one\n  - nested\n    more\n- two\n")
    assert out.splitlines() == ["<h2>Title</h2>", "<p>Some <b>bold</b> and <code>code</code> wrapped.</p>", "<ul>",
                                "<li>one</li>", "<ul>", "<li>nested more</li>", "</ul>", "<li>two</li>", "</ul>"]
