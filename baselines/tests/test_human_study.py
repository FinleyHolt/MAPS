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


def test_on_one_block_the_pools_are_regions_a_participant_never_meets_in_both_arms() -> None:
    rows = {}
    for i in range(400):
        eid = f"calcity::{i}#{i}"
        rows[eid] = {"episode_id": eid, "map_name": "calcity", "target_xy": [float(i % 20) * 450.0, float(i // 20) * 450.0]}
    result = session_plan.plan(rows, ["p01", "p02"], per_arm=15, seed=3, region_m=1000.0)
    assert len(result["pools"][0]) > 3 and len(result["pools"][1]) > 3  # vacuity floor
    for arms in result["participants"].values():
        units = {arm: {session_plan.ground_unit(rows[e], 1000.0) for e in arms[arm]} for arm in ("solo", "team")}
        assert units["solo"] and units["team"] and not units["solo"] & units["team"]


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
    shape = _maps_row(success_criterion="shape")
    square = [(-5.0, 95.0), (5.0, 95.0), (5.0, 105.0), (-5.0, 105.0)]
    inside = team_rows.team_row(shape, {"decision": "replace", "xy": [0.0, 99.0]}, _CFG, lambda eid: square)
    assert inside["metrics"]["SR"] == 1 and inside["metrics"]["success_criterion"] == "shape"
    with pytest.raises(ValueError, match="footprint"):
        team_rows.team_row(shape, {"decision": "replace", "xy": [0.0, 99.0]}, _CFG)
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


def test_a_commit_on_a_mapped_object_outlines_that_object() -> None:
    features = {"buildings": [{"name": "", "world_contour": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                              {"name": "Shed", "world_contour": [[40, 40], [60, 40], [60, 60], [40, 60]]}]}
    contour, what = audit_console.prior_under(features, (50.0, 50.0))
    assert what == "building, Shed" and len(contour) >= 4
    road = {"prior": [{"name": "Lime Avenue", "worldkit_layer": "roads", "world_contour": [[0, 0], [100, 0], [100, 9], [0, 9]]}]}
    assert audit_console.prior_under(road, (50.0, 5.0))[1] == "road, Lime Avenue"
    assert audit_console.prior_under(features, (500.0, 500.0)) is None


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
    assert state["maps_commit"] == "object 0 MAPS detected"
    blob = json.dumps(state)
    for leak in ("target_xy", "100.0", '"SR"', '"NE"', "metrics"):
        assert leak not in blob, leak
    assert s.act({"action": "candidate", "id": 1})["ok"] and s.state()["selected"]["affordable"]
    assert s.state()["selected"]["refusal"] is None
    assert s.act({"action": "replace", "opened": [0]})["ok"]
    (d,) = [json.loads(ln) for ln in (out / "decisions.jsonl").read_text().splitlines()]
    assert d["decision"] == "replace" and d["candidate"] == 1 and d["xy"] == [0.0, 98.0] and d["opened"] == [0]
    assert s.state()["done"] and audit_console.Session(root, ["block_0::1#0"], out, _crop).review is None


def test_the_review_explains_maps_answer_with_its_reply_and_says_what_each_step_used(tmp_path, monkeypatch) -> None:
    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: type("A", (), {"base_features": {}})())
    root = _root(tmp_path)
    attempt = root / "w0" / "maps_x"
    s = audit_console.Session(root, ["block_0::1#0"], tmp_path / "p01")
    # No reply on record: the last step's reasoning stands in.
    assert s.state()["maps_reply"] == "the red car by the gate"
    (attempt / "conversation.jsonl").write_text(json.dumps(
        {"kind": "run", "reasoning": "prior_3 is the gate at px (12, 40); detected_1 sits at [0.0, 98.0]; "
                                     "as in the example h = \"prior_3123\", corner = [3411, 1468]", "said": "r = find('car', nearest=[1, 2])",
         "printed": "detected_0 [0.0, 50.0]\ndetected_1/7 prior_3 [0:20]", "frames": []}) + "\n")
    (attempt / "reply.md").write_text("**Answer: `detected_1`** by the gate, `prior_3` (0.0, 98.0; 2 m off it).")
    state = audit_console.Session(root, ["block_0::1#0"], tmp_path / "p02").state()
    assert state["maps_reply"] == "**Answer: `detected_1`** by the gate, `prior_3` (0.0, 98.0; 2 m off it)."
    assert state["maps_reason"].startswith("prior_3 is the gate at px (12, 40)")
    # A step's references are what its code and output used, never ids or points its reasoning
    # quotes (MAPS repeats its instructions' examples there).
    refs = state["turns"][0]["refs"]
    assert refs["prior"] == ["prior_3"] and sorted(refs["detected"]) == ["detected_0", "detected_1", "detected_7"]
    assert sorted(refs["xy"]) == [[0.0, 50.0], [1.0, 2.0]]
    # The reply writes its points in parentheses too; a step's reasoning there means pixels.
    assert state["reply_refs"] == {"prior": ["prior_3"], "detected": ["detected_1"], "xy": [[0.0, 98.0]]}
    # Each detection by the id MAPS gave it, at its outline; an id no detection carries is left out.
    assert sorted(state["detected"]) == ["detected_0", "detected_1"]
    assert state["detected"]["detected_1"]["label"] == "car" and state["detected"]["detected_1"]["xy"] == [0.0, 98.0]
    assert len(state["detected"]["detected_1"]["contour"]) == 4


def test_a_session_with_no_raster_selects_by_world_point(tmp_path, monkeypatch) -> None:
    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: type("A", (), {"base_features": {}})())
    s = audit_console.Session(_root(tmp_path), ["block_0::1#0"], tmp_path / "p01")
    assert s.act({"action": "select", "x": 0.5, "y": 97.0})["ok"] and s.selected == {"candidate": 1, "xy": (0.0, 98.0)}
    assert s.act({"action": "select", "x": 30.0, "y": 30.0})["ok"] and s.selected == {"candidate": None, "xy": (30.0, 30.0)}


def test_a_replacement_maps_could_not_have_flown_is_refused_in_plain_words(tmp_path, monkeypatch) -> None:
    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: type("A", (), {"base_features": {}})())
    s = audit_console.Session(_root(tmp_path), ["block_0::1#0"], tmp_path / "p01")
    assert s.act({"action": "select", "x": 0.0, "y": 1e6})["ok"]
    refusal = "MAPS could not have reached that spot. Choose a place nearer MAPS's answer, or press Confirm."
    assert s.state()["selected"]["refusal"] == refusal
    assert s.act({"action": "replace"}) == {"ok": False, "error": refusal}


def test_with_the_benchmark_labels_the_candidates_are_the_labelled_objects_near_the_run(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: SimpleNamespace(base_features={}))
    square = lambda x, y, h: [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
    labels = {"block_0": [SimpleNamespace(id=7, contour=square(0.0, 50.0, 3.0), kind="house"),
                          SimpleNamespace(id=8, contour=square(0.0, 100.0, 3.0), kind=""),
                          SimpleNamespace(id=9, contour=square(5000.0, 5000.0, 3.0))]}
    s = audit_console.Session(_root(tmp_path), ["block_0::1#0"], tmp_path / "p01", labels=labels)
    state = s.state()
    assert [c["label"] for c in state["candidates"]] == ["labelled object 7", "labelled object 8"]
    assert state["candidates"][0]["kind"] == "house" and "kind" not in state["candidates"][1]
    assert state["maps_pick"] == 0 and state["maps_commit"] == "object 0 of the benchmark labels"
    assert s.act({"action": "select", "x": 1.0, "y": 101.0})["ok"] and s.selected == {"candidate": 1, "xy": (1.0, 101.0)}
    assert s.act({"action": "replace"})["ok"]
    (d,) = [json.loads(ln) for ln in (tmp_path / "p01" / "decisions.jsonl").read_text().splitlines()]
    assert d["candidate"] == 1 and d["xy"] == [1.0, 101.0]


def test_a_long_labelled_object_holding_maps_answer_sits_at_that_answer() -> None:
    from types import SimpleNamespace

    street = SimpleNamespace(id=3, contour=[(0.0, -5.0), (1000.0, -5.0), (1000.0, 5.0), (0.0, 5.0)], kind="road")
    house = SimpleNamespace(id=4, contour=[(0.0, 20.0), (10.0, 20.0), (10.0, 30.0), (0.0, 30.0)], kind="house")
    held, apart = audit_console.labelled_candidates([street, house], [(900.0, 0.0), (5.0, 25.0)])
    assert held.xy == (900.0, 0.0)
    assert apart.xy != (900.0, 0.0) and audit_console._inside(apart.xy, apart.contour)


def test_only_the_smallest_object_holding_maps_answer_sits_at_it() -> None:
    from types import SimpleNamespace

    area = SimpleNamespace(id=5, contour=[(0.0, 0.0), (300.0, 0.0), (300.0, 100.0), (0.0, 100.0)], kind="shopping area")
    shop = SimpleNamespace(id=6, contour=[(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)], kind="store")
    around, inner = audit_console.labelled_candidates([area, shop], [(50.0, 50.0)])
    # The store is MAPS's pick and sits at its answer; picking the area around it is another answer.
    assert inner.xy == (50.0, 50.0)
    assert around.xy != (50.0, 50.0) and audit_console._inside(around.xy, around.contour)
    # Even when the area's own middle lies in the store, picking the area commits outside the store.
    middle = SimpleNamespace(id=7, contour=[(140.0, 40.0), (160.0, 40.0), (160.0, 60.0), (140.0, 60.0)], kind="store")
    around, inner = audit_console.labelled_candidates([area, middle], [(150.0, 50.0)])
    assert inner.xy == (150.0, 50.0) and not audit_console._inside(around.xy, inner.contour)


def test_a_candidate_is_as_near_as_its_outline() -> None:
    strip = audit_console.Candidate(id=0, label="runway", color="", xy=(500.0, 0.0),
                                    contour=[(0.0, -5.0), (1000.0, -5.0), (1000.0, 5.0), (0.0, 5.0)])
    assert audit_console._reach(strip, (10.0, 0.0)) == 0.0 and audit_console._reach(strip, (10.0, 25.0)) == 20.0


def test_the_map_search_lists_every_episode_blind_and_keeps_the_last_commit(tmp_path) -> None:
    from baselines.human import map_search

    root, out = _root(tmp_path), tmp_path / "p01"
    s = map_search.Search(root, ["block_0::1#0"], out)
    state = s.state()
    assert [e["instruction"] for e in state["episodes"]] == ["the red car"] and state["done_n"] == 0
    blob = json.dumps(state)
    for leak in ("target", "100.0", '"SR"', "metrics", "commit_xy"):
        assert leak not in blob, leak
    assert not s.open("block_9::0#0")["ok"]
    assert s.open("block_0::1#0")["ok"] and s.commit("block_0::1#0", (3.0, 40.0))["ok"]
    assert s.commit("block_0::1#0", (0.0, 99.0))["ok"]
    resumed = map_search.Search(root, ["block_0::1#0"], out).state()
    assert resumed["done_n"] == 1 and resumed["episodes"][0]["committed"] == [0.0, 99.0]


def test_a_map_search_commit_is_scored_where_it_lies_with_nothing_flown() -> None:
    row = _maps_row()
    hit = team_rows.team_row(row, {"decision": "commit", "xy": [0.0, 95.0], "search_s": 12.0}, _CFG)
    assert hit["metrics"]["SR"] == 1 and hit["metrics"]["steps_taken"] == 0 and hit["metrics"]["policy"] == "human_map"
    assert hit["metrics"]["maps_SR"] == 0 and hit["commit_xy"] == [0.0, 95.0] and hit["visited_xy"] == [[0.0, 95.0]]
    assert team_rows.team_row(row, {"decision": "commit", "xy": [0.0, 10.0]}, _CFG)["metrics"]["SR"] == 0


def test_the_bench_labels_say_what_each_object_is_in_plain_words(tmp_path, monkeypatch) -> None:
    from benchmarks.citynav import paths

    square = [[0, 0], [0, 5], [5, 5], [5, 0]]
    objects = [{"id": 1, "object_type": "Building", "tags": {"building": "house"}},
               {"id": 2, "object_type": "Parking", "tags": {"amenity": "parking"}},
               {"id": 3, "object_type": "HighVegetation", "tags": {}},
               {"id": 4, "object_type": "Ground", "tags": {"leisure": "pitch"}},
               {"id": 5, "object_type": "Building", "tags": {"building": "yes"}}]
    (tmp_path / "objects.json").write_text(json.dumps({"block_0": {
        str(o["id"]): {**o, "position": [0, 0, 0], "dimension": [1, 1, 1], "contour": square} for o in objects}}))
    monkeypatch.setattr(paths, "OBJECTS_JSON", tmp_path / "objects.json")
    assert [o.kind for o in audit_console.bench_labels()["block_0"]] == [
        "house", "parking lot", "trees", "sports field", "building"]


def test_both_ledgers_and_their_rows_carry_when_each_episode_was_opened_and_decided(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from baselines.human import map_search
    from benchmarks.citynav.run_benchmark import assets

    monkeypatch.setattr(assets, "load_episode_assets", lambda m: SimpleNamespace(base_features={}))
    root = _root(tmp_path)
    s = audit_console.Session(root, ["block_0::1#0"], tmp_path / "team")
    assert s.act({"action": "confirm"})["ok"]
    (d,) = [json.loads(ln) for ln in (tmp_path / "team" / "decisions.jsonl").read_text().splitlines()]
    assert d["opened_at"] <= d["at"] and d["opened_at"].endswith("Z")
    m = team_rows.team_row(_maps_row(), d, _CFG)["metrics"]
    assert (m["audit_opened_at"], m["audit_decided_at"]) == (d["opened_at"], d["at"])
    q = map_search.Search(root, ["block_0::1#0"], tmp_path / "solo")
    q.open("block_0::1#0")
    q.commit("block_0::1#0", (0.0, 99.0))
    (c,) = [json.loads(ln) for ln in (tmp_path / "solo" / "decisions.jsonl").read_text().splitlines()]
    assert c["opened_at"] <= c["at"]
    assert team_rows.team_row(_maps_row(), c, _CFG)["metrics"]["search_opened_at"] == c["opened_at"]
