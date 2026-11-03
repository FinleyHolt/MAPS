"""The capacity session holds up to what a participant's laptop does to it: no fcntl on Windows,
a stale tab, a torn last line, a relaunch mid-report, a relaunch within the second a result
landed and a relaunch without the rehearsal length."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from baselines.human import capacity
from baselines.tests.test_capacity import _act, _session

REPO = Path(__file__).resolve().parents[2]


def _relaunch(s: capacity.Capacity, now: list[float]) -> capacity.Capacity:
    return capacity.Capacity(s.maps_root, s.order, s.out, s.arm, agents=1, session_s=3600, clock=lambda: now[0])


def test_the_console_loads_and_a_session_opens_without_fcntl(tmp_path) -> None:
    """Windows has no fcntl: what a session imports must load without it."""
    code = ("import sys, json; from pathlib import Path; sys.modules['fcntl'] = None\n"
            "import baselines.human.study_server\n"
            "from baselines.human.capacity import Capacity\n"
            "from baselines.tests.test_human_study import _root\n"
            "tmp = Path(sys.argv[1]); root = _root(tmp)\n"
            "assert Capacity(root, ['block_0::1#0'], tmp / 'out', 'solo').state()['current']['mode'] == 'search'\n")
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, cwd=REPO)


def test_an_action_for_a_report_that_already_changed_is_refused(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "solo", now)
    assert s.act({"action": "skip", "episode_id": "block_0::1#0"})["ok"]
    stale = s.act({"action": "skip", "episode_id": "block_0::1#0"})
    assert stale == {"ok": False, "error": "That report already changed. The page now shows the current one."}
    assert s.act({"action": "skip"}) == stale                 # an action that names no report
    assert len((tmp_path / "p01" / "solo" / "answers.jsonl").read_text().splitlines()) == 1


def test_a_torn_last_line_is_repaired_before_the_next_answer(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "p01" / "solo" / "answers.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"episode_id": "block_0::1#0", "sou')
    s = _session(tmp_path, monkeypatch, "solo", [1000.0])
    assert _act(s, "skip")["ok"]
    assert json.loads(ledger.read_text().splitlines()[-1])["episode_id"] == "block_0::1#0"


def test_a_relaunch_keeps_the_open_report_as_it_was_opened(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "team", now)
    assert s.state()["current"]["mode"] == "search"           # MAPS's first result lands at 100 s
    now[0] += 900
    s = _relaunch(s, now)           # relaunched mid-search
    assert s.state()["current"]["mode"] == "search"
    assert _act(s, "select", x=0.0, y=99.0)["ok"] and _act(s, "commit")["ok"]
    (a,) = [json.loads(ln) for ln in (tmp_path / "p01" / "team" / "answers.jsonl").read_text().splitlines()]
    assert (a["source"], a["maps_ready_at_open"], a["opened_at"], a["answer_s"]) == (
        "own_search", False, capacity._utc(1000.0), 900.0)
    now[0] += 30
    s = _relaunch(s, now)           # the next report opened at the answer
    assert s.opened_at == a["answered_at"] and s.state()["current"]["mode"] == "verify"


def test_a_resumed_session_keeps_its_recorded_length(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "solo", now)
    meta = tmp_path / "p01" / "solo" / "session.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "session_s": 60.0}))   # a rehearsal's length
    now[0] += 120
    s = _relaunch(s, now)           # relaunched with the full length
    assert s.state()["over"] and not _act(s, "skip")["ok"]


def test_a_relaunch_reopens_the_report_at_the_answers_precise_time(tmp_path, monkeypatch) -> None:
    now = [1000.7]
    s = _session(tmp_path, monkeypatch, "team", now)
    now[0] += 600.2                 # the second result landed at 600 s, within the answer's second
    assert s.act({"action": "skip", "episode_id": "block_0::1#0"})["ok"]
    assert s.state()["current"]["mode"] == "verify"
    (a,) = [json.loads(ln) for ln in (tmp_path / "p01" / "team" / "answers.jsonl").read_text().splitlines()]
    assert a["answered_epoch"] == now[0]
    s = _relaunch(s, now)
    assert s.state()["current"]["mode"] == "verify"
    ledger = tmp_path / "p01" / "team" / "answers.jsonl"
    ledger.write_text(json.dumps({k: v for k, v in a.items() if k != "answered_epoch"}) + "\n")
    assert _relaunch(s, now).opened_at == a["answered_at"]          # a ledger from before the instant was kept


def test_back_reopens_the_last_report_and_the_one_left_open_comes_back_as_it_was_opened(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "team", now)
    assert not _act(s, "back")["ok"]                          # nothing before the first report
    now[0] += 50
    assert _act(s, "select", x=0.0, y=99.0)["ok"] and _act(s, "commit")["ok"]
    left = (s.index, s.opened, s.verify)                      # opened at 50 s, before MAPS's second landed
    now[0] += 700
    assert _act(s, "back")["ok"]
    cur = s.state()["current"]
    assert (cur["n"], cur["mode"], cur["earlier"], cur["point"], cur["can_back"]) == (1, "search", "own_search", [0.0, 99.0], False)
    assert _act(s, "select", x=5.0, y=5.0)["ok"] and _act(s, "commit")["ok"]
    assert (s.index, s.opened, s.verify) == left              # its clock and screen, not a fresh open
    assert s.state()["current"]["earlier"] is None and s.state()["answered_n"] == 1
    a, b = [json.loads(ln) for ln in (tmp_path / "p01" / "team" / "answers.jsonl").read_text().splitlines()]
    assert (a["xy"], "revises" in a, b["xy"], b["revises"]) == ([0.0, 99.0], False, [5.0, 5.0], True)
    from baselines.human import study_qc
    assert study_qc.checks(tmp_path / "p01" / "team") == []  # an hour: the real length
    assert _act(s, "back")["ok"] and _act(s, "resume")["ok"] and (s.index, s.opened) == left[:2]
    assert not _act(s, "resume")["ok"]                        # nothing left open to go back to
    s = _relaunch(s, now)                                     # the revision does not move the open report's clock
    assert (s.index, s.opened, s.verify) == left


def test_every_act_and_what_the_console_reports_land_in_the_event_record(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "solo", now)
    e = s.current
    now[0] = 1012.0
    s.log([{"kind": "search", "t": 5.0, "query": "88th & Lime"}])
    assert not _act(s, "commit")["ok"]                         # no point yet: refused, and logged
    assert _act(s, "select", x=1.234, y=5.0)["ok"] and _act(s, "commit")["ok"]
    rows = [json.loads(line) for line in (tmp_path / "p01" / "solo" / "events.jsonl").read_text().splitlines()]
    assert rows[0] == {"kind": "search", "t": 5.0, "query": "88th & Lime", "episode_id": e, "order_index": 0, "session_s": 12.0}
    assert [(r["action"], r["ok"], r["on"]) for r in rows[1:]] == [("commit", False, e), ("select", True, e), ("commit", True, e)]
    assert rows[1]["error"] and (rows[2]["x"], rows[2]["y"]) == (1.23, 5.0)
