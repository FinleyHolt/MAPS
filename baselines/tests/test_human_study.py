"""The human study's three arms stay comparable and blind.

The session plan never hands a participant the same block in both arms, nor
one target object twice in an arm. A confirmed audit is the MAPS row
unchanged, and a replacement is scored where it lands. The audit console
never sends the auditor the target or a score, and a session resumes after
its last decision.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from baselines.human import audit_console, session_plan, team_rows
from maps.perception.observation import Observation

_CFG = {"max_steps": 200, "success_radius_m": 20.0}


def _rows(n_blocks: int = 6, objects: int = 5, descriptions: int = 2) -> dict[str, dict]:
    rows = {}
    for b in range(n_blocks):
        for o in range(objects):
            for d in range(descriptions):
                eid = f"block_{b}::{o}#{d}"
                rows[eid] = {"episode_id": eid, "map_name": f"block_{b}"}
    return rows


def test_a_participant_never_meets_a_block_in_both_arms_or_a_target_twice() -> None:
    rows = _rows()
    result = session_plan.plan(rows, ["p01", "p02", "p03", "p04"], per_arm=8, seed=7)
    assert len(result["participants"]) == 4  # vacuity floor
    for arms in result["participants"].values():
        blocks = {arm: {rows[e]["map_name"] for e in arms[arm]} for arm in ("solo", "team")}
        assert blocks["solo"] and blocks["team"] and not blocks["solo"] & blocks["team"]
        for arm in ("solo", "team"):
            keys = [session_plan.target_key(e) for e in arms[arm]]
            assert len(keys) == len(set(keys)) == 8
    p = result["participants"]
    # Pool and order are counterbalanced over each four participants.
    assert [p[k]["solo_pool"] for k in p] == [0, 1, 0, 1]
    assert [p[k]["first"] for k in p] == ["solo", "solo", "team", "team"]
    assert session_plan.plan(rows, list(p), per_arm=8, seed=7) == result


def _maps_row(**metrics) -> dict:
    return {"episode_id": "block_0::1#0", "map_name": "block_0", "description": "the red car",
            "start_xy": [0.0, 0.0], "target_xy": [0.0, 100.0], "commit_xy": [0.0, 50.0],
            "visited_xy": [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0]], "status": "ok", "workdir": "w0/maps_x",
            "metrics": {"SR": 0, "NE": 50.0, "steps_taken": 10, "success_criterion": "ring", **metrics}}


def test_a_confirmed_audit_is_the_maps_row_and_a_replacement_is_scored_where_it_lands() -> None:
    row = _maps_row()
    kept = team_rows.team_row(row, {"decision": "confirm", "xy": row["commit_xy"]}, _CFG)
    assert {k: kept["metrics"][k] for k in ("SR", "NE", "steps_taken")} == {"SR": 0, "NE": 50.0, "steps_taken": 10}
    assert kept["visited_xy"] == row["visited_xy"] and kept["metrics"]["commit_path"] == "audit:confirm"
    fixed = team_rows.team_row(row, {"decision": "replace", "xy": [0.0, 99.0]}, _CFG)
    assert fixed["metrics"]["SR"] == 1 and fixed["metrics"]["maps_SR"] == 0
    assert fixed["metrics"]["steps_taken"] == 10 + team_rows.correction_leg(row, (0.0, 99.0))[0]
    with pytest.raises(ValueError, match="budget"):
        team_rows.team_row(_maps_row(steps_taken=199), {"decision": "replace", "xy": [0.0, 99.0]}, _CFG)


def _root(tmp_path):
    root = tmp_path / "root"
    attempt = root / "w0" / "maps_x"
    (attempt / "flight").mkdir(parents=True)
    row = _maps_row()
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (root / "run_manifest.json").write_text(json.dumps({"config_yaml": yaml.safe_dump(_CFG)}))
    (attempt / "conversation.jsonl").write_text(
        json.dumps({"kind": "run", "reasoning": "the red car by the gate", "said": "look()", "printed": "", "frames": []}) + "\n"
        + json.dumps({"kind": "mission_end", "ended_by": "reply"}) + "\n")
    square = lambda x, y: [[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2], [x - 2, y + 2]]
    (attempt / "detections.jsonl").write_text("".join(json.dumps(
        {"world_xy": xy, "label": "car", "confidence": 0.9, "attributes": {"color": "red 0.8"},
         "metadata": {"track_id": f"wt_{i}", "world_contour": square(*xy)}, "step": 0}) + "\n"
        for i, xy in enumerate([[0.0, 50.0], [0.0, 98.0]])))
    (attempt / "flight" / "tour_cursor.json").write_text(json.dumps({"stops_visited": [{"xy": [0.0, 50.0]}]}))
    return root


def _crop(map_name, xy, half):
    x, y = xy
    x0, y0, x1, y1 = x - half, y - half, x + half, y + half
    return Observation(pose_xy=xy, altitude_m=half, world_window=(x0, y0, x1, y1), rgb=np.zeros((120, 120, 3), np.uint8),
                       ortho_view_corners_xy=((x0, y1), (x1, y1), (x1, y0), (x0, y0)))


def test_the_audit_console_is_blind_and_resumes_after_its_last_decision(tmp_path, monkeypatch) -> None:
    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: type("A", (), {"base_features": {}})())
    root, out = _root(tmp_path), tmp_path / "p01"
    s = audit_console.Session(root, ["block_0::1#0"], out, _crop)
    state = s.state()
    assert state["maps_pick"] == 0 and len(state["candidates"]) == 2
    blob = json.dumps(state)
    for leak in ("target_xy", "100.0", '"SR"', '"NE"', "metrics"):
        assert leak not in blob, leak
    assert s.act({"action": "candidate", "id": 1})["ok"] and s.state()["selected"]["affordable"]
    assert s.act({"action": "replace", "opened": [0]})["ok"]
    (d,) = [json.loads(ln) for ln in (out / "decisions.jsonl").read_text().splitlines()]
    assert d["decision"] == "replace" and d["candidate"] == 1 and d["xy"] == [0.0, 98.0] and d["opened"] == [0]
    assert s.state()["done"] and audit_console.Session(root, ["block_0::1#0"], out, _crop).review is None
