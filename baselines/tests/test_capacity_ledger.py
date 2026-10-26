"""The capacity session holds up to what a participant's laptop does to it: no fcntl on Windows,
a stale tab, a torn last line, a relaunch mid-report and a relaunch without the rehearsal length."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from baselines.human import capacity
from baselines.tests.test_capacity import _session

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
    assert len((tmp_path / "p01" / "solo" / "answers.jsonl").read_text().splitlines()) == 1
    assert s.act({"action": "skip"})["ok"]


def test_a_torn_last_line_is_repaired_before_the_next_answer(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "p01" / "solo" / "answers.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"episode_id": "block_0::1#0", "sou')
    s = _session(tmp_path, monkeypatch, "solo", [1000.0])
    assert s.act({"action": "skip"})["ok"]
    assert json.loads(ledger.read_text().splitlines()[-1])["episode_id"] == "block_0::1#0"
