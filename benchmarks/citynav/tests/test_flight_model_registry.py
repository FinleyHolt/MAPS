"""The action-space model is config-declared, and ``discrete6`` is unchanged.

The registry exists so a quadcopter, a straight-routing aircraft and a C-12
each fly the same mission at their own cost. ``discrete6`` is not one option
among them: it is the CityNav six-action contract the trajectory log is
submitted under, so the refactor that made it swappable has to prove it did
not move a single pose.

``fixtures/discrete6_trajectory.json`` is that proof — the replay recorded on
the tree before the registry existed. Equality against a recording fails OPEN
if the recording is trivial, so every case also asserts a pose count and the
six-action alphabet is pinned by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.citynav.flight_model import (
    DiscreteFlightSimulator,
    FlightModelConfig,
    known_flight_kinds,
    resolve_flight_model,
)

FIXTURE = Path(__file__).parent / "fixtures" / "discrete6_trajectory.json"


class _State:
    def __init__(self, commit):
        self.pose_xy = (0.0, 0.0)
        self.altitude_m = 50.0
        self.heading_deg = 0.0
        self.visited_xy = []
        self.react_stats = {}
        self.belief_snapshot = SimpleNamespace(confirmed_target_xy=commit)

    def record_visit(self):
        self.visited_xy.append(self.pose_xy)


def _vp(xy, alt, heading, fid):
    return SimpleNamespace(xy=xy, altitude_m=alt, heading_deg=heading, fid=fid)


# Byte-identical to the generator that produced the fixture: a multi-stop tour
# with turns, a climb and a descent; a commit-only flight; and a flight the
# action budget truncates short of its commit.
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


def _replay(case, tmp_path, monkeypatch, flight_model: FlightModelConfig):
    from benchmarks.citynav.run_benchmark import (
        load_config,
        synthesize_compliant_trajectory,
    )

    cfg = load_config("configs/citynav.yaml")
    cfg.flight_model = flight_model
    cfg.max_steps = case["max_steps"]
    cfg.altitude_m = 50.0
    cfg.descend_to_land = False
    cfg.landing_altitude_m = 5.0
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
    traj = json.loads((tmp_path / "run" / "compliant_trajectory.json").read_text())
    return steps, state, traj


@pytest.mark.parametrize("name", sorted(CASES))
def test_discrete6_replays_the_recorded_trajectory_exactly(name, tmp_path, monkeypatch):
    recorded = json.loads(FIXTURE.read_text())[name]
    steps, state, traj = _replay(
        CASES[name], tmp_path, monkeypatch, FlightModelConfig()
    )

    assert steps == recorded["steps"]
    assert traj["pred_xy"] == recorded["pred_xy"]
    assert traj["final_pos_xy"] == recorded["final_pos_xy"]
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
    these six primitives, and no registry entry may quietly extend them."""
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


def test_discrete6_is_the_default_kind_and_resolves_to_the_discrete_simulator():
    cfg = FlightModelConfig()
    assert cfg.kind == "discrete6"
    sim = resolve_flight_model(cfg, start_pose=(0.0, 0.0, 0.0), start_altitude_m=50.0)
    assert isinstance(sim, DiscreteFlightSimulator)
    assert sim.arrival_tol_m == 2.5  # snap_to_actions(xy_tol_m=2.5)


def test_an_unknown_kind_names_the_known_ones():
    """A config typo must say what it could have said, not fail obscurely."""
    with pytest.raises(ValueError) as exc:
        FlightModelConfig(kind="helicopter")
    message = str(exc.value)
    assert "helicopter" in message
    for kind in known_flight_kinds():
        assert kind in message
    assert "discrete6" in known_flight_kinds()
