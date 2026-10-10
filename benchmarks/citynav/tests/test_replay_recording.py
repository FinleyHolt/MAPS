"""The replay of a flight is pinned pose for pose.

``fixtures/replay_trajectory.json`` records the replay's whole flight for a
multi-stop tour with turns, a climb and a descent; a commit-only flight; and
a flight the action budget truncates short of its commit. Equality against a
recording fails OPEN if the recording is trivial, so every case also asserts
a pose count, and the six-action alphabet is pinned by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "replay_trajectory.json"


class _State:
    def __init__(self, commit):
        self.pose_xy = (0.0, 0.0)
        self.altitude_m = 50.0
        self.heading_deg = 0.0
        self.visited_xy = []
        self.belief_snapshot = SimpleNamespace(confirmed_target_xy=commit)

    def record_visit(self):
        self.visited_xy.append(self.pose_xy)


def _vp(xy, alt, heading, fid):
    return SimpleNamespace(xy=xy, altitude_m=alt, heading_deg=heading, fid=fid)


CASES = {
    "tour_then_commit": dict(
        stops=[
            _vp((120.0, 0.0), 60.0, 90.0, "f1"),
            _vp((120.0, 90.0), 40.0, 0.0, "f2"),
            _vp((-30.0, 60.0), 50.0, 270.0, "f3"),
        ],
        commit=(45.0, -25.0),
        max_steps=400,
    ),
    "commit_only": dict(stops=[], commit=(200.0, 150.0), max_steps=400),
    "budget_truncated": dict(
        stops=[_vp((500.0, 500.0), 80.0, 45.0, "f1")],
        commit=(900.0, 900.0),
        max_steps=40,
    ),
}


def _replay(case, tmp_path, monkeypatch):
    from benchmarks.citynav.run_benchmark import (
        load_config,
        synthesize_compliant_trajectory,
    )

    cfg = load_config("configs/citynav.yaml")
    cfg.max_steps = case["max_steps"]
    cfg.altitude_m = 50.0
    state = _State(case["commit"])
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    steps = synthesize_compliant_trajectory(
        cfg=cfg,
        state=state,
        episode=SimpleNamespace(episode_id="unit::0"),
        feature_tour=SimpleNamespace(viewpoints=case["stops"]),
        start_pose_xyh=(0.0, 0.0, 0.0),
        start_alt=50.0,
    )
    return steps, state


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_replay_flies_the_recorded_trajectory_exactly(name, tmp_path, monkeypatch):
    recorded = json.loads(FIXTURE.read_text())[name]
    steps, state = _replay(CASES[name], tmp_path, monkeypatch)

    assert steps == recorded["steps"]
    assert list(state.pose_xy) == recorded["pred_xy"] == recorded["final_pos_xy"]
    flown = [[round(p[0], 9), round(p[1], 9)] for p in state.visited_xy]
    assert flown == recorded["visited_xy"]
    assert round(state.heading_deg, 9) == recorded["heading_deg"]
    assert round(state.altitude_m, 9) == recorded["altitude_m"]


def test_the_recording_is_not_vacuous():
    """Equality against a 1-pose recording would pass on any flight model.

    The three cases together have to actually fly: a multi-leg tour, a
    commit-only hop, and one the budget cuts short at exactly its cap.
    """
    recorded = json.loads(FIXTURE.read_text())
    assert sorted(recorded) == sorted(CASES)
    assert recorded["tour_then_commit"]["steps"] > 100
    assert recorded["commit_only"]["steps"] > 30
    assert recorded["budget_truncated"]["steps"] == CASES["budget_truncated"]["max_steps"]
    for name, case in recorded.items():
        assert len(case["visited_xy"]) == case["steps"], name


def test_the_six_action_alphabet_is_the_contract():
    """Pinned by name: the leaderboard trajectory log is submittable only in
    these six primitives."""
    from benchmarks.citynav.action_snap import (
        ALTITUDE_STEP_M,
        FORWARD_STEP_M,
        TURN_STEP_DEG,
        CityNavAction,
    )

    assert {a.value for a in CityNavAction} == {
        "move_forward",
        "turn_left",
        "turn_right",
        "ascend",
        "descend",
        "stop",
    }
    assert (FORWARD_STEP_M, TURN_STEP_DEG, ALTITUDE_STEP_M) == (5.0, 30.0, 2.0)
