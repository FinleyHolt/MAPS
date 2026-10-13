"""A policy behind the seam lands the harness's own commit and replay.

The comparison between a baseline and the MAPS arm stands on the seam being
the only thing that differs: the replay, the commit the readers score, and
the row keys are the harness's. A stub policy that returns two stops and a
commit must therefore come back as flights whose last is the commit, a
decision record naming the policy's tier, a replay that ends at the commit,
and the policy's own numbers under its name.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

from baselines.seam import (
    BaselineHarness,
    BaselineResult,
    Landmark,
    _extract_json,
    _landmarks_from_rows,
    fly,
    landmark_distance,
    match_landmarks,
)
from benchmarks.citynav.run_benchmark.episode_runner import EpisodeState
from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory
from maps.perception.geometry.sensor_spec import resolve_sensor


def _install_stub_policy(monkeypatch, name: str, run) -> None:
    pkg = types.ModuleType(f"baselines.{name}")
    pkg.__path__ = []  # a package, so the dotted import resolves
    mod = types.ModuleType(f"baselines.{name}.policy")
    mod.run_episode = run
    monkeypatch.setitem(sys.modules, f"baselines.{name}", pkg)
    monkeypatch.setitem(sys.modules, f"baselines.{name}.policy", mod)


def _hydrated(alt: float = 50.0, ground: float = 0.0):
    """The parts of ``episode_runner.Hydrated`` the seam reads."""
    return SimpleNamespace(
        cfg=SimpleNamespace(max_steps=200),
        episode=SimpleNamespace(episode_id="unit::1#0", map_name="unit_block_0",
                                description="the car by the depot", start_xy=(0.0, 0.0), target_xy=(0.0, 100.0)),
        assets=SimpleNamespace(
            bounds=SimpleNamespace(x_min=-500.0, y_min=-500.0, x_max=500.0, y_max=500.0),
            cell_size_m=2.0, grid_shape=(500, 500), base_features={"prior": []},
        ),
        state=EpisodeState(pose_xy=(0.0, 0.0), altitude_m=alt),
        search_loop=SimpleNamespace(_ortho_ground_z_m=ground),
        sensor=None,
        perception=None,
    )


def test_a_stub_policy_lands_its_commit_and_the_replay_ends_there(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    _install_stub_policy(monkeypatch, "stub", lambda h: BaselineResult(
        stops=[((0.0, 60.0), 50.0), ((0.0, 120.0), 50.0)],
        commit_xy=(0.0, 100.0),
        commit_label="stub_object",
        commit_tier="stub:tier",
        lock_step=1,
        extra_metrics={"marks": 3},
    ))
    h = _hydrated()
    flight = fly(h, "stub")

    assert [f["resolved_xy"] for f in flight.flights] == [[0.0, 60.0], [0.0, 120.0], [0.0, 100.0]]
    assert flight.flights[-1]["target"] == "stub_object"
    assert flight.metrics["policy"] == "stub"
    assert flight.metrics["commit_path"] == "stub:tier"
    assert flight.metrics["committed_fid"] == "stub_object"
    decision = json.loads((tmp_path / "stub" / "decision.json").read_text())
    assert decision["commit_path"] == "stub:tier"
    assert decision["world_xy_resolved"] == [0.0, 100.0]
    # The policy's own numbers ride under its name.
    assert flight.metrics["stub_stops"] == 2
    assert flight.metrics["stub_marks"] == 3
    assert flight.metrics["stub_lock_step"] == 1

    # The runner's replay flies the stops, then the commit leg: the scored
    # pose is the commit, reached through the six-action space.
    state = h.state
    state.record_visit()
    steps = synthesize_compliant_trajectory(
        cfg=h.cfg, state=state, flights=flight.flights, start_pose_xyh=(0.0, 0.0, 0.0), start_alt=50.0,
    )
    assert 0 < steps <= h.cfg.max_steps
    assert len(state.visited_xy) == steps
    assert abs(state.pose_xy[1] - 100.0) <= 5.0


def test_a_policy_that_commits_nothing_commits_its_last_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    _install_stub_policy(monkeypatch, "stub3", lambda h: BaselineResult(
        stops=[((0.0, 60.0), 70.0)], commit_xy=None, commit_label="last", commit_tier="stub:exhausted"))
    flights = fly(_hydrated(), "stub3").flights
    assert flights[-1]["resolved_xy"] == [0.0, 60.0] and flights[-1]["altitude_m"] == 70.0


def test_the_runner_start_altitude_is_the_altitude_the_policy_sees(tmp_path, monkeypatch):
    """The runner owns the start (the dataset z, or ``start_altitude_agl_m``);
    the harness hands that altitude to the policy unchanged."""
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    seen = {}

    def run(h):
        seen["alt"] = h.start_alt_m
        seen["half"] = h.view_half_m()
        return BaselineResult(stops=[], commit_xy=(0.0, 100.0), commit_label="x", commit_tier="stub:tier")

    _install_stub_policy(monkeypatch, "stub2", run)
    fly(_hydrated(alt=80.0, ground=30.0), "stub2")
    assert seen["alt"] == 80.0
    assert abs(seen["half"] - 50.0) < 1e-6


def test_the_view_half_follows_the_sensor_not_a_fixed_square(tmp_path):
    """CityNav's 90-degree sensor gives upstream's square (half = height above
    ground); Cal City's 16-degree Oryx gives its own, much narrower footprint."""
    h = BaselineHarness.__new__(BaselineHarness)
    h.start_alt_m, h.ground_z_m = 150.0, 50.0
    h.sensor = resolve_sensor("nadir_1008_90")
    assert abs(h.view_half_m() - 100.0) < 1e-6
    h.sensor = resolve_sensor("oryx_orx32_25mm_nadir")
    assert 10.0 < h.view_half_short_m() < h.view_half_m() < 20.0


def _lm(name, contour):
    return Landmark(name=name, contour=contour, centroid=(sum(p[0] for p in contour) / len(contour), sum(p[1] for p in contour) / len(contour)), tags={})


def test_landmarks_match_on_their_distinctive_words():
    """Whole-name similarity sent "Helmore building" to Mumford Building on the
    first audit; the shared generic word carried the match."""
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    prior = [_lm("Mumford Building", sq), _lm("Helmore Building", sq), _lm("Willmore Road", sq), _lm("Science Centre", sq)]
    got = [lm.name for lm in match_landmarks(["Helmore building", "Willmore road", "Music Therapy Centre", "OCR building"], prior)]
    assert got == ["Helmore Building", "Willmore Road"]


def test_a_tie_on_the_distinctive_word_is_broken_by_the_whole_name():
    """The second audit: "Grafton Street" went to Grafton Centre and
    "Staffordshire Street" to Staffordshire Gardens, both alphabetically first
    on a tied token; a one-letter shared token matched F.Hinds; an all-generic
    landmark name ("Hall") matched "small building"."""
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    prior = [_lm(n, sq) for n in ["Grafton Centre", "Grafton Street", "Staffordshire Gardens", "Staffordshire Street",
                                  "F.Hinds", "Cambridge Magistrates Courts", "Hall", "Petty Cury"]]
    got = [lm.name for lm in match_landmarks(
        ["Grafton Street", "Staffordshire Street", "Great Court F", "tennis courts", "small building", "Petty Cury street"], prior)]
    assert got == ["Grafton Street", "Staffordshire Street", "Petty Cury"]


def test_landmark_distance_reads_the_area_not_the_centroid():
    street = _lm("Long Street", [(0, 0), (1000, 0), (1000, 6), (0, 6)])
    assert landmark_distance(street, (900.0, 3.0)) == 0.0
    assert abs(landmark_distance(street, (900.0, 20.0)) - 14.0) < 1e-6


def test_the_first_json_object_is_read_out_of_prose_or_a_fence():
    assert _extract_json('thinking... ```json\n{"verdict": "move", "n": 2}\n``` trailing') == {"verdict": "move", "n": 2}
    assert _extract_json('Here is my answer: {"a": [1, 2], "b": {"c": "x"}} and a tail {') == {"a": [1, 2], "b": {"c": "x"}}
    assert _extract_json("no json here") is None


def test_a_street_is_widened_to_its_carriageway_from_its_tags():
    """A 6 m ribbon never contained a parked car; CityRefer's road landmarks
    were the paved surface. Width tag first, lanes next, then the class."""
    line = [[0.0, 0.0], [100.0, 0.0]]
    rows = [
        {"name": "Tagged Road", "world_contour": line, "highway": "residential", "width": "12", "category": "roads"},
        {"name": "Lanes Road", "world_contour": line, "highway": "residential", "lanes": "4", "category": "roads"},
        {"name": "Plain Road", "world_contour": line, "highway": "residential", "category": "roads"},
        {"name": "Foot Path", "world_contour": line, "highway": "footway", "category": "roads"},
    ]
    by = {lm.name: lm for lm in _landmarks_from_rows(rows)}
    assert landmark_distance(by["Tagged Road"], (50.0, 5.9)) == 0.0
    assert landmark_distance(by["Lanes Road"], (50.0, 6.4)) == 0.0
    assert landmark_distance(by["Plain Road"], (50.0, 4.9)) == 0.0
    assert landmark_distance(by["Plain Road"], (50.0, 8.0)) > 2.0
    assert landmark_distance(by["Foot Path"], (50.0, 3.0)) > 0.5


def test_an_answer_that_is_not_json_is_reshaped_with_the_dial_off(monkeypatch):
    """The first call rides the arm's own dial at the paper's temperature; an
    answer that is not the schema's JSON is reshaped by a constrained call
    with the dial off."""
    import maps.llm.http as http
    from maps.llm import endpoint_spec

    sent: list[dict] = []
    replies = iter(["The answer is the left building.", '{"pick": 3}'])

    def fake(base_url, api_key, **kw):
        sent.append(kw)
        return {"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}

    monkeypatch.setattr(http, "_chat_completions_raw", fake)
    monkeypatch.setattr(endpoint_spec, "_SPEC", endpoint_spec.EndpointSpec(
        base_url="http://serve.example:8001/v1", model_id="inkling-small", max_model_len=262144,
        tokenizer_name="qwen", max_output_tokens=40000))
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    h = BaselineHarness.__new__(BaselineHarness)
    h.policy, h.cfg = "visa", SimpleNamespace()
    h.llm_calls = h.reasoning_tokens = h.extraction_calls = h.reasoning_truncated = 0
    h.calls, h.phase_s = [], {"examine": 0.0}
    schema = {"type": "object", "properties": {"pick": {"type": "integer"}}, "required": ["pick"]}

    assert h.llm_json([{"role": "user", "content": "which?"}], schema, "pick") == {"pick": 3}
    assert "reasoning_effort" not in sent[0] and "response_format" not in sent[0]
    assert sent[0]["temperature"] == 0.6 and sent[0]["max_tokens"] == 40000
    assert sent[1]["reasoning_effort"] == "none"
    assert sent[1]["response_format"]["type"] == "json_schema"
    assert h.extraction_calls == 1 and h.llm_calls == 2


def _window_harness(bounds):
    h = BaselineHarness.__new__(BaselineHarness)
    h.episode = SimpleNamespace(start_xy=(100.0, 100.0))
    h.assets = SimpleNamespace(bounds=SimpleNamespace(**bounds))
    return h


def test_a_citynav_block_is_inside_its_search_window():
    """Every CityNav block is narrower than the window's minimum half-side,
    so the window clipped to the block is the block: the ports plan and map
    over exactly what they did before the window existed."""
    block = {"x_min": 0.0, "y_min": 0.0, "x_max": 414.0, "y_max": 414.0}
    w = _window_harness(block).search_window([])
    assert (w.x_min, w.y_min, w.x_max, w.y_max) == (0.0, 0.0, 414.0, 414.0)


def test_a_city_sized_block_is_held_to_the_window_maximum():
    """A 13 km Cal City block sized GeoNav's map past 16 GB; the window caps
    the extent a policy allocates over, landmarks or none."""
    from baselines.seam import WINDOW_MAX_HALF_M, WINDOW_MIN_HALF_M

    city = {"x_min": -6000.0, "y_min": -5000.0, "x_max": 7456.0, "y_max": 5210.0}
    h = _window_harness(city)
    w = h.search_window([])
    assert w.x_max - w.x_min == 2 * WINDOW_MIN_HALF_M
    far = _lm("Far Road", [(6000.0, 100.0), (6010.0, 100.0), (6010.0, 110.0)])
    w = h.search_window([far])
    assert w.x_max - w.x_min == 2 * WINDOW_MAX_HALF_M
    near = _lm("Near Road", [(700.0, 100.0), (710.0, 100.0), (710.0, 110.0)])
    w = h.search_window([near])
    assert w.x_max - 100.0 >= 610.0
