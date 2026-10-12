"""The harness's leg: coverage before the leg, arrival, a short pass, a lost leg, and what the flight is served."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.flight import mission as M
from benchmarks.flight.tests import world


@pytest.fixture
def make_mission(built, make_twin, make_live, tmp_path):
    def make(start_xy=(600.0, 100.0)):
        twin = make_twin(start_xy)
        live = make_live(twin)
        workdir = tmp_path / "sortie" / "run"
        workdir.mkdir(parents=True)
        m = M.Mission(episode_id="calcity::t", split="calcity", config="configs/calcity.yaml", workdir=workdir,
                      live=live, recorder=None, close_log=lambda: None, run_no=1)
        m.map_name, m._tiles_dir = "calcity", built[0]
        return m, twin
    return make


def _hydrated(sensor, cells=None):
    """The parts of a hydrated mission the leg reads."""
    return SimpleNamespace(
        sensor=sensor, state=SimpleNamespace(pose_xy=(600.0, 100.0), altitude_m=world.AGL_M),
        search_loop=SimpleNamespace(_ortho_ground_z_m=0.0, node_cells_xy=lambda token: cells))


def _session(h, fly):
    """The parts of a mission session a leg reads: the hydrated stack, the
    target resolution, and the flight the leg hands its observation to."""
    from benchmarks.citynav.run_benchmark.mission_engine import resolve_flight_target

    return SimpleNamespace(h=h, fly=fly, resolve=lambda token: resolve_flight_target(
        token, node_xy=lambda fid: None, pose_xy=h.state.pose_xy, marker_xy=None))


def _legs(m):
    return [r["said"] for r in m.transcript if r.get("kind") == "leg"]


def test_a_disc_area_is_half_the_observation_span_about_the_point(built):
    area = M.leg_area(_hydrated(built[3]), "", (600.0, 600.0))
    assert len(area) == 19 and area[0] == (600.0, 600.0)
    r = M.leg_span_m(_hydrated(built[3])) / 2.0
    assert max(abs(complex(x - 600.0, y - 600.0)) for x, y in area) == pytest.approx(r)


def test_a_flight_over_unflown_ground_posts_the_leg_and_serves_the_pass_that_held_it(built, make_mission):
    m, twin = make_mission()
    h = _hydrated(built[3])
    served = []

    def inner(item):
        obs = m._held[0] if m._held else None
        served.append(obs)
        return {"stop": item["target_fid"]}

    m._fly_live(_session(h, inner), {"target_fid": "[600, 600]"})
    assert [leg.point for leg in twin.legs] == [(600.0, 600.0)]
    assert twin.legs[0].coverage_floor == M.COVERED_FRACTION and len(twin.legs[0].area) == 19
    assert served[0] is not None and served[0].ortho_view_corners_xy is not None
    assert "arrived" in _legs(m)[-1]
    assert list((m.workdir / M.OBSERVATIONS_DIR).glob("*.jpg"))


def test_ground_the_pass_holds_already_is_answered_without_a_leg(built, make_mission):
    m, twin = make_mission()
    twin.post(M.Leg(target="t", point=(600.0, 700.0), step=0))
    twin.sleep(10.0)
    posted = len(twin.legs)
    m._fly_live(_session(_hydrated(built[3]), lambda item: {}), {"target_fid": "[600, 600]"})
    assert len(twin.legs) == posted
    assert "no leg is flown" in _legs(m)[-1]


def test_an_unaskable_ring_is_asked_four_times_then_the_leg_is_flown(built, make_mission):
    m, twin = make_mission()
    index, asks = twin.index, []
    twin.index = lambda since=None: asks.append(since) or None
    t0 = twin.now()
    cov, refusal = m._coverage_before_leg([(0.0, 0.0)] * 3)
    assert cov is None and len(asks) == M.COVERAGE_TRIES and refusal
    assert twin.now() - t0 == pytest.approx((M.COVERAGE_TRIES - 1) * M.COVERAGE_RETRY_S)
    twin.index = index


def test_a_pass_that_misses_a_side_is_short_and_is_asked_again_toward_it(built, make_mission):
    """The target's footprint runs 400 m east of its point, wider than the
    frame: the pass over the point holds its west part and leaves, the leg is
    asked again moved east, and the second pass completes it."""
    m, twin = make_mission()
    footprint = [(600.0 + dx, 600.0 + dy) for dx in range(0, 401, 50) for dy in (-30, 0, 30)]
    m._fly_live(_session(_hydrated(built[3], cells=footprint), lambda item: {}), {"target_fid": "[600, 600]"})
    first, again = twin.legs[0], twin.legs[1]
    assert (first.revision, again.revision) == (0, 1)
    assert again.point[0] > 700.0 and abs(again.point[1] - 600.0) < 1.0
    said = _legs(m)
    assert any("the E side missing" in s and "re-fly 1 of 2" in s for s in said), said
    assert "arrived" in said[-1]


def test_a_leg_whose_area_the_aircraft_never_reaches_is_lost(built, make_mission):
    m, twin = make_mission()
    twin.post(M.Leg(target="t", point=(600.0, 100.0), step=0))
    twin.route = []
    twin.heading_deg = 180.0           # flying away from the area
    area_ll = m._area_lonlat([(600.0, 1100.0), (610.0, 1100.0), (600.0, 1110.0)])
    assert m._wait_arrival((600.0, 1100.0), area=area_ll, timeout_s=5.0) == "lost"


def test_a_report_banks_one_ledger_row_and_a_second_report_replaces_it(built, make_mission):
    m, _twin = make_mission()
    m.started_at = 0.0
    m.status, m.ended_by = M.STATE_AWAITING, "reply"
    m.bank()
    m.bank()
    rows = [json.loads(line) for line in (m.workdir.parent / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "calcity::t" and rows[0]["status"] == "ok" and rows[0]["workdir"] == "run"


def test_an_unknown_episode_is_refused_by_name(monkeypatch):
    import pytest

    from benchmarks.citynav import adapter, paths
    from benchmarks.flight import mission

    monkeypatch.setattr(paths, "split_path", lambda split: split)
    monkeypatch.setattr(adapter, "load_split", lambda path: [])
    with pytest.raises(ValueError, match="calcity holds no episode calcity::9#9"):
        mission.resolve_episode("calcity::9#9", "calcity")


def test_the_flight_refuses_a_config_that_withholds_a_part(tmp_path):
    import pytest

    from benchmarks.flight import mission

    repo = __import__("pathlib").Path(__file__).resolve().parents[3]
    arm = tmp_path / "arm.yaml"
    arm.write_text(f"base: {repo / 'configs' / 'calcity.yaml'}\nwithhold: [images]\n")
    with pytest.raises(ValueError, match="bench ablation"):
        mission.hydrate(episode_id="calcity::0#0", split="calcity", config=str(arm))


def test_a_leg_arrives_on_the_recording_while_the_aircraft_s_own_pose_cannot_be_read(built, make_mission):
    m, twin = make_mission()
    twin.post(M.Leg(target="t", point=(600.0, 700.0), step=0))
    m._live.ownship = lambda: None
    area_ll = m._area_lonlat([(600.0, 600.0), (610.0, 600.0), (600.0, 610.0)])
    assert m._wait_arrival((600.0, 600.0), area=area_ll, timeout_s=60.0) == "arrived"
