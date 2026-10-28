"""The capacity study deals each participant one condition, replays MAPS on its recorded
timing, and logs every answer with its source and timeline.

The conditions are dealt half and half, and each participant works every report in their
own order, the plan reproducible from its seed. A report is
verified if MAPS's result had landed when it was opened and searched by hand if not, and a
result landing meanwhile does not change that. A skip is a miss; the session refuses
answers once its time is up; every source scores through team_rows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from baselines.human import capacity, team_rows
from baselines.tests.test_human_study import _CFG, _maps_row, _root


def test_the_plan_deals_half_alone_and_half_with_maps_each_over_every_report() -> None:
    rows = {f"block_0::{i}#0": {"metrics": {"SR": int(i % 3 == 0)}, "elapsed_s": 60.0} for i in range(101)}
    labels = [f"p{i:02d}" for i in range(1, 15)]
    doc = capacity.plan(rows, labels, seed=7)
    ps = doc["participants"]
    assert (doc["design"], doc["episodes_n"], doc["maps_sr"]) == ("between subjects", 101, round(34 / 101, 4))
    assert sorted(ps[p]["arm"] for p in labels) == ["solo"] * 7 + ["team"] * 7
    assert all(list(ps[p]["lists"]) == [ps[p]["arm"]] and sorted(ps[p]["lists"][ps[p]["arm"]]) == sorted(rows)
               for p in labels)
    team = [p for p in labels if ps[p]["arm"] == "team"]
    assert all(ps[p]["maps_ready_in_session"] == 101 and ps[p]["agents"] == capacity.AGENTS for p in team)
    assert not any("maps_ready_in_session" in ps[p] for p in labels if p not in team)
    assert ps[team[0]]["lists"]["team"] != ps[team[1]]["lists"]["team"]
    assert capacity.plan(rows, labels, seed=7) == doc and capacity.plan(rows, labels, seed=8) != doc
    assert [d["arm"] for d in capacity.plan(rows, labels[:3], seed=7)["participants"].values()].count("solo") == 2


def test_rehearsal_labels_take_their_named_condition_outside_the_dealing() -> None:
    rows = {f"block_0::{i}#0": {"metrics": {"SR": 1}, "elapsed_s": 60.0} for i in range(20)}
    labels = [f"p{i:02d}" for i in range(1, 15)]
    doc = capacity.plan(rows, labels, seed=7, tests={"t01": "team", "t02": "solo"})
    ps = doc["participants"]
    assert {p: ps[p]["arm"] for p in labels} == {p: d["arm"] for p, d in capacity.plan(rows, labels, seed=7)["participants"].items()}
    assert (ps["t01"]["arm"], ps["t01"]["test"], ps["t02"]["arm"], ps["t02"]["test"]) == ("team", True, "solo", True)
    assert "maps_ready_in_session" in ps["t01"] and not any("test" in ps[p] for p in labels)


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


def test_every_source_scores_and_a_skip_alone_or_before_maps_lands_is_a_miss() -> None:
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
    alone = team_rows.team_row(_maps_row(SR=1), {**base, "arm": "solo", "source": "skip", "xy": None,
                                                 "maps_in_session": True}, _CFG)["metrics"]
    assert alone["SR"] == 0
    kept = team_rows.team_row(_maps_row(SR=1), {**base, "source": "skip", "xy": None, "maps_in_session": True}, _CFG)
    assert kept["metrics"]["SR"] == 1 and kept["metrics"]["capacity_source"] == "skip" and kept["commit_xy"] is not None


def test_a_skip_with_maps_keeps_its_answer_only_when_it_landed_within_the_session() -> None:
    session = {"started_epoch": 0.0, "session_s": 2400}
    assert team_rows.landed_in_session({"maps_ready_at": "1970-01-01T00:39:59Z"}, session)
    assert not team_rows.landed_in_session({"maps_ready_at": "1970-01-01T00:40:01Z"}, session)
    assert not team_rows.landed_in_session({"maps_ready_at": None}, session)


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


def test_the_servers_prior_and_every_session_share_one_build_of_the_block(tmp_path, monkeypatch) -> None:
    from baselines.human import audit_console, study_server
    from benchmarks.citynav.run_benchmark import assets

    audit_console.block_prior.cache_clear()
    now = [1000.0]
    s = _session(tmp_path, monkeypatch, "team", now)
    loads = []
    monkeypatch.setattr(assets, "load_episode_assets", lambda m: loads.append(m) or SimpleNamespace(base_features={}))
    study_server.prior_doc("block_0", lambda pts: [list(p) for p in pts])
    now[0] += 650
    assert _act(s, "select", x=0.0, y=99.0)["ok"] and _act(s, "commit")["ok"]
    assert s.state()["current"]["mode"] == "verify" and loads == ["block_0"]
    again = capacity.Capacity(tmp_path / "root", ["block_0::1#0", "block_0::2#0"], tmp_path / "p01" / "team", "team",
                              agents=1, session_s=3600, clock=lambda: now[0])
    assert again.state()["current"]["mode"] == "verify" and loads == ["block_0"]


def test_the_session_and_every_answer_carry_the_visits_ids(tmp_path, monkeypatch) -> None:
    root = _root(tmp_path)
    ids = {"run_id": "p01-3f9a2c1e", "participant": "p01"}
    s = capacity.Capacity(root, ["block_0::1#0"], tmp_path / "out", "solo", ids=ids, clock=lambda: 1000.0)
    assert _act(s, "skip")["ok"]
    session = json.loads((tmp_path / "out" / "session.json").read_text())
    answer = json.loads((tmp_path / "out" / "answers.jsonl").read_text())
    assert {k: session[k] for k in ids} == ids and {k: answer[k] for k in ids} == ids
