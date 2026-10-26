"""The capacity study deals equal halves, replays MAPS on its recorded timing, and logs
every answer with its source and timeline.

The halves differ in MAPS SR by at most one episode's worth, and each group of four
participants meets each half in each condition with either condition first. A report is
verified if MAPS's result had landed when it was opened and searched by hand if not, and a
result landing meanwhile does not change that. A skip is a miss; the session refuses
answers once its time is up; every source scores through team_rows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from baselines.human import capacity, team_rows
from baselines.tests.test_human_study import _CFG, _maps_row, _root


def test_the_halves_match_on_maps_sr_and_the_plan_counterbalances() -> None:
    rows = {f"block_0::{i}#0": {"metrics": {"SR": int(i % 3 == 0)}, "elapsed_s": 60.0} for i in range(101)}
    a, b = capacity.halves(rows, seed=7)
    assert not set(a) & set(b) and len(a) + len(b) == 101 and abs(len(a) - len(b)) <= 1
    assert abs(sum(rows[e]["metrics"]["SR"] for e in a) - sum(rows[e]["metrics"]["SR"] for e in b)) <= 1
    doc = capacity.plan(rows, ["p01", "p02", "p03", "p04"], seed=7)
    ps = doc["participants"]
    assert sorted((ps[p]["solo_half"], ps[p]["first"]) for p in ps) == [("A", "solo"), ("A", "team"), ("B", "solo"), ("B", "team")]
    assert ps["p01"]["maps_ready_in_session"] == len(ps["p01"]["lists"]["team"])
    assert ps["p01"]["lists"]["solo"] != ps["p03"]["lists"]["solo"] and sorted(ps["p01"]["lists"]["solo"]) == sorted(ps["p03"]["lists"]["solo"])


def test_the_replay_runs_the_list_in_order_on_so_many_agents() -> None:
    starts, ends = capacity.release_times([100, 50, 30, 10], agents=2)
    assert starts == [0, 0, 50, 80] and ends == [100, 50, 80, 90]


def _session(tmp_path, monkeypatch, arm: str, now: list[float]):
    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: SimpleNamespace(base_features={}))
    root = _root(tmp_path)
    rows = (root / "episodes.jsonl").read_text().splitlines()
    second = {**json.loads(rows[0]), "episode_id": "block_0::2#0", "elapsed_s": 500.0}
    first = {**json.loads(rows[0]), "elapsed_s": 100.0}
    (root / "episodes.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    return capacity.Capacity(root, ["block_0::1#0", "block_0::2#0"], tmp_path / "p01" / arm, arm, agents=1,
                             session_s=3600, clock=lambda: now[0])


def _act(s, action: str, **more) -> dict:
    """An action as the console sends it: naming the report it was meant for."""
    return s.act({"action": action, "episode_id": s.current, **more})


def test_a_report_is_verified_only_if_maps_had_landed_when_it_opened(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "team", now)
    assert s.state()["current"]["mode"] == "search"          # MAPS's first result lands at 100 s
    now[0] += 150
    assert s.state()["current"]["mode"] == "search"          # landing meanwhile does not interrupt
    now[0] += 500                                            # the second lands at 600 s; still searching the first
    assert _act(s, "select", x=0.0, y=99.0)["ok"] and _act(s, "commit")["ok"]
    cur = s.state()["current"]                               # opened after its result landed
    assert cur["mode"] == "verify" and cur["review"]["maps_pick"] == 0 and "target" not in json.dumps(cur)
    assert _act(s, "confirm")["ok"] and s.state()["over"]
    a, b = [json.loads(ln) for ln in (tmp_path / "p01" / "team" / "answers.jsonl").read_text().splitlines()]
    assert (a["source"], a["maps_ready_at_open"], b["source"], b["maps_ready_at_open"]) == ("own_search", False, "confirm", True)
    assert a["opened_at"] <= a["answered_at"] and b["maps_ready_at"] is not None


def test_alone_a_skip_is_logged_and_the_session_refuses_answers_after_its_hour(tmp_path, monkeypatch) -> None:
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "solo", now)
    assert s.state()["current"]["mode"] == "search" and _act(s, "skip")["ok"]
    assert s.state()["skipped_n"] == 1 and s.state()["list"][0]["status"] == "skipped"
    now[0] += 3601
    assert not _act(s, "skip")["ok"] and s.state()["over"] and s.state()["current"] is None


def test_every_source_scores_and_a_skip_is_a_miss() -> None:
    row = _maps_row()
    base = {"arm": "team", "opened_at": "2026-10-21T04:00:00Z", "answered_at": "2026-10-21T04:01:00Z", "answer_s": 60.0}
    hit = team_rows.team_row(row, {**base, "source": "own_search", "xy": [0.0, 95.0]}, _CFG)["metrics"]
    assert hit["SR"] == 1 and hit["capacity_source"] == "own_search" and hit["policy"] == "capacity_team"
    kept = team_rows.team_row(row, {**base, "source": "confirm", "xy": [0.0, 50.0]}, _CFG)["metrics"]
    assert kept["SR"] == 0 and kept["audit_decision"] == "confirm" and kept["capacity_opened_at"] == base["opened_at"]
    fixed = team_rows.team_row(row, {**base, "source": "replace", "xy": [0.0, 99.0]}, _CFG)["metrics"]
    assert fixed["SR"] == 1 and fixed["maps_SR"] == 0
    skip = team_rows.team_row(_maps_row(SR=1), {**base, "source": "skip", "xy": None}, _CFG)["metrics"]
    assert skip["SR"] == 0 and skip["maps_SR"] == 1 and skip["capacity_source"] == "skip"


def test_results_released_within_the_session_and_never_reached_count_as_maps_scored_them() -> None:
    rows = {"a": {**_maps_row(SR=1), "episode_id": "a", "elapsed_s": 100.0},
            "b": {**_maps_row(SR=0), "episode_id": "b", "elapsed_s": 200.0},
            "c": {**_maps_row(SR=1), "episode_id": "c", "elapsed_s": 5000.0}}
    session = {"arm": "team", "agents": 1, "session_s": 3600, "started_epoch": 0.0, "order": ["a", "b", "c"]}
    (u,) = team_rows.unchecked_answers(session, {"a"}, rows)   # c lands at 5300 s, after the hour
    assert u["episode_id"] == "b" and u["source"] == "unchecked" and u["answered_at"] == u["maps_ready_at"] == "1970-01-01T00:05:00Z"
    m = team_rows.team_row(rows["b"], u, _CFG)["metrics"]
    assert m["SR"] == m["maps_SR"] == 0 and m["capacity_source"] == "unchecked" and m["capacity_opened_at"] is None
    assert team_rows.unchecked_answers({**session, "arm": "solo"}, set(), rows) == []


def test_practice_draws_only_rows_the_root_never_flew(tmp_path, monkeypatch) -> None:
    from benchmarks.citynav import adapter, paths

    root = _root(tmp_path)                      # flies block_0::1
    monkeypatch.setattr(team_rows, "select_bench", lambda r: None)
    monkeypatch.setattr(paths, "split_path", lambda s: s)
    monkeypatch.setattr(adapter, "load_split", lambda p: [{"o": 1}, {"o": 2}, {"o": 3}])
    monkeypatch.setattr(adapter, "parse_episode", lambda r, row_index: SimpleNamespace(
        episode_id=f"block_0::{r['o']}#{row_index}", map_name="block_0", description=f"find {r['o']}"))
    (root / "run_manifest.json").write_text(json.dumps({**json.loads((root / "run_manifest.json").read_text()), "split": "x"}))
    picked = capacity.practice_root(root, 5, 1, tmp_path / "practice")
    assert sorted(picked) == ["block_0::2#1", "block_0::3#2"]
    assert (tmp_path / "practice" / "practice.txt").read_text().split() == picked


def test_a_team_session_with_no_answers_scores_the_results_it_released(tmp_path) -> None:
    root = _root(tmp_path)
    row = json.loads((root / "episodes.jsonl").read_text())
    (root / "episodes.jsonl").write_text(json.dumps({**row, "elapsed_s": 100.0}) + "\n")
    session = tmp_path / "p01" / "team" / "session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"arm": "team", "agents": 1, "session_s": 3600.0, "started_epoch": 0.0,
                                   "order": ["block_0::1#0"]}))
    out = tmp_path / "scored"
    assert team_rows.main(["--maps-root", str(root), "--decisions", str(session.parent / "answers.jsonl"),
                           "--out-root", str(out)]) == 0
    (scored,) = [json.loads(ln) for ln in (out / "episodes.jsonl").read_text().splitlines()]
    assert scored["metrics"]["capacity_source"] == "unchecked"
    assert (out / "session.json").read_text() == session.read_text()


def test_a_second_session_on_the_same_root_reads_its_ledger_once_until_it_changes(tmp_path, monkeypatch) -> None:
    from baselines.human import session_plan

    reads = []
    real = session_plan.iter_json_rows
    monkeypatch.setattr(session_plan, "iter_json_rows", lambda path: reads.append(path) or real(path))
    now = [1000.0]
    _session(tmp_path, monkeypatch, "team", now)
    root = tmp_path / "root"
    capacity.Capacity(root, ["block_0::1#0"], tmp_path / "p01" / "solo", "solo", clock=lambda: now[0])
    assert len(reads) == 1
    (root / "episodes.jsonl").write_text((root / "episodes.jsonl").read_text().splitlines()[0] + "\n")
    assert list(session_plan.scored_rows(root)) == ["block_0::1#0"] and len(reads) == 2


def test_a_session_opened_again_on_the_same_root_does_not_build_the_block_prior_again(tmp_path, monkeypatch) -> None:
    from baselines.human import audit_console

    builds = []
    real = audit_console.block_landmarks
    monkeypatch.setattr(audit_console, "block_landmarks", lambda features: builds.append(1) or real(features))
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "team", now)
    now[0] += 650
    assert _act(s, "select", x=0.0, y=99.0)["ok"] and _act(s, "commit")["ok"]
    assert s.state()["current"]["mode"] == "verify" and len(builds) == 1
    again = capacity.Capacity(tmp_path / "root", ["block_0::1#0", "block_0::2#0"], tmp_path / "p01" / "team", "team",
                              agents=1, session_s=3600, clock=lambda: now[0])
    assert again.state()["current"]["mode"] == "verify" and len(builds) == 1
