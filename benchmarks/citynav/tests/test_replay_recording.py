"""The replay of a flight is pinned pose for pose.

``fixtures/replay_trajectory.json`` records the replay's whole flight for a
multi-stop tour with turns, a climb and a descent; a one-flight mission; and
a mission the action budget truncates short of its commit (the last flight). Equality against a
recording fails OPEN if the recording is trivial, so every case also asserts
a pose count, and the six-action alphabet is pinned by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "replay_trajectory.json"


def _vp(xy, alt, fid):
    return {"target": fid, "resolved_xy": list(xy), "altitude_m": alt}


CASES = {
    "tour_then_commit": dict(
        stops=[
            _vp((120.0, 0.0), 60.0, "f1"),
            _vp((120.0, 90.0), 40.0, "f2"),
            _vp((-30.0, 60.0), 50.0, "f3"),
            _vp((45.0, -25.0), 50.0, "f4"),
        ],
        max_steps=400,
    ),
    "commit_only": dict(stops=[_vp((200.0, 150.0), 50.0, "f1")], max_steps=400),
    "budget_truncated": dict(
        stops=[_vp((500.0, 500.0), 80.0, "f1"), _vp((900.0, 900.0), 80.0, "f2")],
        max_steps=40,
    ),
}


def _replay(case, tmp_path, monkeypatch):
    from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory
    from benchmarks.run_record.config import load_config

    cfg = load_config("configs/citynav.yaml")
    cfg.max_steps = case["max_steps"]
    return synthesize_compliant_trajectory(
        cfg=cfg,
        flights=case["stops"],
        start_pose_xyh=(0.0, 0.0, 0.0),
        start_alt=50.0,
    )[0]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_replay_flies_the_recorded_trajectory_exactly(name, tmp_path, monkeypatch):
    recorded = json.loads(FIXTURE.read_text())[name]
    visited = _replay(CASES[name], tmp_path, monkeypatch)

    assert len(visited) == recorded["steps"]
    assert list(visited[-1]) == recorded["final_pos_xy"]
    flown = [[round(p[0], 9), round(p[1], 9)] for p in visited]
    assert flown == recorded["visited_xy"]


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


def test_the_move_alphabet_is_the_contract():
    """Pinned by name: the leaderboard trajectory log holds only upstream's
    five moves; its sixth action, stop, is where the log ends."""
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
    }
    assert (FORWARD_STEP_M, TURN_STEP_DEG, ALTITUDE_STEP_M) == (5.0, 30.0, 2.0)
