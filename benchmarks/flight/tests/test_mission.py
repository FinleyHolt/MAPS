"""The harness's leg: coverage before the leg, arrival, a short pass, a lost leg, and what the flight is served."""

from __future__ import annotations

import json

import pytest

from benchmarks.run_record.episode_workdir import OBSERVATIONS_DIR, TRACK_JSONL
from benchmarks.flight import mission as M
from benchmarks.flight.tests import world


@pytest.fixture
def make_mission(built, bench, make_twin, make_live, tmp_path):
    def make(start_xy=(600.0, 100.0)):
        twin = make_twin(start_xy)
        live = make_live(twin)
        workdir = tmp_path / "sortie" / "run"
        workdir.mkdir(parents=True)
        m = M.Mission(bench=bench, episode_id="calcity::t", split="calcity", config="configs/calcity.yaml", workdir=workdir,
                      live=live, recorder=None, close_log=lambda: None, run_no=1,
                      agl_m=world.AGL_M)
        m.map_name, m._tiles_dir = "calcity", built[0]
        return m, twin
    return make


def _legs(m):
    return [r["said"] for r in m.transcript if r.get("kind") == "leg"]


def test_a_disc_area_is_half_the_observation_span_about_the_point(built):
    area = M.leg_area(world.hydrated(built[3]), "", (600.0, 600.0))
    assert len(area) == 19 and area[0] == (600.0, 600.0)
    r = M.leg_span_m(world.hydrated(built[3])) / 2.0
    assert max(abs(complex(x - 600.0, y - 600.0)) for x, y in area) == pytest.approx(r)


def test_a_flight_over_unflown_ground_posts_the_leg_and_serves_the_pass_that_held_it(built, make_mission):
    m, twin = make_mission()
    h = world.hydrated(built[3])
    served = []

    def inner(item):
        obs = m._held[0] if m._held else None
        served.append(obs)
        return {"stop": item["target_fid"]}

    m._fly_live(world.session(h, inner), {"target_fid": "[600, 600]"})
    assert [leg.point for leg in twin.legs] == [(600.0, 600.0)]
    assert len(twin.legs[0].area) == 19
    assert served[0] is not None and served[0].ortho_view_corners_xy is not None
    assert "arrived" in _legs(m)[-1]
    assert list((m.workdir / OBSERVATIONS_DIR).glob("*.jpg"))


def test_ground_the_pass_holds_already_is_answered_without_a_leg(built, make_mission):
    m, twin = make_mission()
    twin.post(M.Leg(target="t", point=(600.0, 700.0), step=0))
    twin.sleep(10.0)
    posted = len(twin.legs)
    m._fly_live(world.session(world.hydrated(built[3]), lambda item: {}), {"target_fid": "[600, 600]"})
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
    m._fly_live(world.session(world.hydrated(built[3], cells=footprint), lambda item: {}), {"target_fid": "[600, 600]"})
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
    assert m._wait_arrival((600.0, 1100.0), area=area_ll, timeout_s=5.0)[0] == "lost"


def test_a_report_banks_one_ledger_row_and_a_second_report_replaces_it(built, make_mission):
    m, _twin = make_mission()
    m.started_at = 0.0
    m.status, m.ended_by = M.STATE_AWAITING, "reply"
    m.bank()
    m.bank()
    rows = [json.loads(line) for line in (m.workdir.parent / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "calcity::t" and rows[0]["status"] == "ok" and rows[0]["workdir"] == "run"


def test_a_track_fix_is_where_the_aircraft_was_at_its_stamp_under_a_timing_offset(make_mission):
    """The start's UTC is read through the boresight's timing offset, so every
    fix of the track is read through it too: a fix stamped ``t_unix`` is the
    pose whose UTC that is."""
    import dataclasses
    import datetime as dt

    m, twin = make_mission()
    m.live.boresight = dataclasses.replace(m.live.boresight, time_offset_s=10.0)
    twin.sleep(30.0)
    m._t0, m.started_at_utc = twin.now(), m._utc_now()
    twin.sleep(20.0)
    m._record_track()
    first = json.loads((m.workdir.parent / TRACK_JSONL).read_text().splitlines()[0])
    pose = m.live.pose_now(m._t0)
    assert first["t_unix"] == pytest.approx(dt.datetime.fromisoformat(pose.utc).timestamp(), abs=1e-3)
    x, y = M.lonlat_to_local(m._tiles_dir, m.map_name, pose.lon, pose.lat)
    assert (first["x"], first["y"]) == pytest.approx((x, y), abs=0.01)


def test_a_start_without_a_fix_takes_its_utc_at_the_first_bank_with_one(make_mission):
    """No fix at the start leaves no UTC start; the first bank with a fix
    reads it back through the vehicle seconds since, so the row and the
    track carry it."""
    import datetime as dt

    m, twin = make_mission()
    m.started_at, m._t0, m.started_at_utc = 0.0, twin.now(), None
    pose = m.live.pose_now(m._t0)
    twin.sleep(20.0)
    m.status, m.ended_by = M.STATE_AWAITING, "reply"
    m.bank()
    row = json.loads((m.workdir.parent / "episodes.jsonl").read_text().splitlines()[0])
    assert row["started_at_utc"] == pytest.approx(dt.datetime.fromisoformat(pose.utc).timestamp(), abs=1e-3)
    assert (m.workdir.parent / TRACK_JSONL).read_text().strip()


def test_an_unknown_episode_is_refused_by_name(monkeypatch, bench):
    import pytest

    from benchmarks.citynav import adapter
    from benchmarks.flight import mission

    # The engine reads its split through the adapter: an empty one holds no episode.
    monkeypatch.setattr(adapter, "load_split", lambda path: [])
    with pytest.raises(ValueError, match="calcity holds no episode calcity::9#9"):
        mission.resolve_episode(bench, "calcity::9#9", "calcity")


def test_ground_held_by_frames_that_cannot_be_fetched_is_flown_again_and_served_from_the_new_pass(
        built, make_mission):
    """The ring's index holds the area, but none of its frames comes over the
    link: the leg is flown, arrival waits for frames recorded after it was
    posted, and the observation is fused from those."""
    m, twin = make_mission()
    twin.post(M.Leg(target="t", point=(600.0, 700.0), step=0))
    twin.sleep(10.0)
    n_old = len(twin.rows)
    fetch = twin.frame
    twin.frame = lambda seq: None if seq < n_old else fetch(seq)
    served = []
    m._fly_live(world.session(world.hydrated(built[3]), lambda item: served.append(m._held) or {}),
                {"target_fid": "[600, 600]"})
    assert len(twin.legs) == 2
    assert served[0] is not None, _legs(m)
    assert min(served[0][1]["mosaic"]["seqs"]) >= n_old
    assert "is held by the pass (19/19 points), but no frame of it could be served" in _legs(m)[0], _legs(m)
    assert "arrived" in _legs(m)[-1]


def test_a_leg_arrives_on_the_recording_while_the_aircraft_s_own_pose_cannot_be_read(built, make_mission):
    m, twin = make_mission()
    twin.post(M.Leg(target="t", point=(600.0, 700.0), step=0))
    m.live.ownship = lambda: None
    area_ll = m._area_lonlat([(600.0, 600.0), (610.0, 600.0), (600.0, 610.0)])
    assert m._wait_arrival((600.0, 600.0), area=area_ll, timeout_s=60.0)[0] == "arrived"


@pytest.mark.parametrize("stop_in_wait", [False, True])
def test_a_stopped_mission_posts_no_leg_fetches_no_frame_and_flies_nothing(built, make_mission, stop_in_wait):
    """Stop before the flight, and Stop while its leg is out: after it nothing
    is posted, no frame is fetched, and the flight is not perceived."""
    m, twin = make_mission()
    fetched, flown = [], []
    frame = twin.frame
    twin.frame = lambda seq: fetched.append(seq) or frame(seq)
    if stop_in_wait:
        wait = m._wait_arrival
        m._wait_arrival = lambda *a, **k: (m.stop(), wait(*a, **k))[1]
    else:
        m.stop()
    out = m._fly_live(world.session(world.hydrated(built[3]), lambda item: flown.append(item) or {}),
                      {"target_fid": "[600, 600]"})
    assert out == {"error": "the mission is stopped"}
    assert len(twin.legs) == int(stop_in_wait) and fetched == [] and flown == []


def test_the_summary_line_counts_the_turns_the_row_counts(make_mission):
    """A conversation that failed mid-turn has turns in flight: the printed line and the row agree."""
    m, _twin = make_mission()
    m.turns_used, m._turn_diag["turns_used"] = 2, 3
    assert f"{m.row()['turns_used']} turns" in M.summary_line(m) and m.row()["turns_used"] == 5


def test_a_lost_leg_with_no_pose_read_says_so_rather_than_an_infinite_distance(make_mission, caplog):
    m, _twin = make_mission()
    m.live.ownship = lambda: None
    area_ll = m._area_lonlat([(600.0, 1100.0), (610.0, 1100.0), (600.0, 1110.0)])
    assert m._wait_arrival((600.0, 1100.0), area=area_ll, timeout_s=2.0)[0] == "lost"
    assert "no pose read: the leg is lost" in caplog.text and "inf" not in caplog.text
    said = m._outcome_words("lost", "t at (600, 1100)", None, 0.0)
    assert "(no pose read)" in said and "inf" not in said


def test_a_stop_while_a_short_pass_is_read_posts_no_refly(built, make_mission):
    """A Stop that lands as a short pass is read posts no re-fly leg: after
    Stop the mission posts nothing more."""
    m, twin = make_mission()
    waits = []

    def wait(*a, **k):
        waits.append(1)
        if len(waits) == 1:
            m.stop()
            return "short", {}
        return "cancel", None
    m._wait_arrival = wait
    out = m._fly_live(world.session(world.hydrated(built[3]), lambda item: {}), {"target_fid": "[600, 600]"})
    assert out == {"error": "the mission is stopped"}
    assert [leg.revision for leg in twin.legs] == [0]


def test_a_short_pass_is_said_as_the_points_it_held():
    """17 of 19 points is said as 17/19, never rounded up to a share that reads as held."""
    said = M.Mission._held_words({"coverage": {"held": 17, "points": 19}, "mosaic": {"asked": 2},
                                  "seq": 5, "off_nadir_deg": 3.1})
    assert "held 17/19 points of it in 2 frame(s)" in said and "90%" not in said


def test_a_stop_while_the_ring_is_asked_says_no_leg_is_flown(built, make_mission):
    """A Stop that lands while the recording is asked again posts no leg and
    writes no row saying the leg is flown."""
    m, twin = make_mission()
    twin.index = lambda since=None: m.stop()
    out = m._fly_live(world.session(world.hydrated(built[3]), lambda item: {}), {"target_fid": "[600, 600]"})
    assert out == {"error": "the mission is stopped"}
    assert twin.legs == [] and not any("the leg is flown" in s for s in _legs(m)), _legs(m)


def test_a_lost_leg_after_a_short_pass_says_the_closest_the_leg_came(make_mission):
    """The pass came within 100 m and was short; the re-fly never came nearer
    than 740 m and was lost: the leg row says the leg's closest, 100 m."""
    m, _twin = make_mission()
    offs = iter([100.0] + [400.0] * 6)
    m.live.coverage = lambda area, after=None: {"fraction": 0.5, "held": 1, "points": 2, "best": [None, None]}
    m.live.ownship = lambda: {"x": 600.0 + next(offs, 740.0), "y": 600.0, "agl_m": 900.0}
    m._closest_m = float("inf")
    assert m._wait_arrival((600.0, 600.0), area=[], timeout_s=60.0, reach_m=200.0)[0] == "short"
    assert m._wait_arrival((600.0, 600.0), area=[], timeout_s=3.0)[0] == "lost"
    assert "(closest 100 m)" in m._outcome_words("lost", "t at (600, 600)", None, 0.0)


def test_the_mission_stamps_the_arm_s_reasoning_effort_as_its_requests_carry_it(make_mission, monkeypatch):
    m, _twin = make_mission()
    m.bench = m.bench.with_environ({"SEARCH_SAMPLING_REASONING_EFFORT": " high "})
    assert m._mission_stamp()["reasoning_effort"] == "high"
    m.bench = m.bench.with_environ({"SEARCH_SAMPLING_REASONING_EFFORT": "  "})
    assert m._mission_stamp()["reasoning_effort"] is None


def test_a_short_pass_before_the_leg_is_logged_as_the_frames_that_held_it(built, make_mission, caplog):
    """The pass before the leg held 3 of the area's points in 2 frames out of
    the 40 the ring scored: the log line counts the 2 that held them."""
    from benchmarks.flight.live_source import coverage_record

    m, _twin = make_mission()
    best = [(1.0, 7), (2.0, 7), (1.5, 9)] + [None] * 16
    m._coverage_before_leg = lambda area: (coverage_record(best, 40), None)
    m._post_and_wait = lambda *a: None
    caplog.set_level("INFO", logger=M.log.name)
    m._fly_live(world.session(world.hydrated(built[3]), lambda item: {}), {"target_fid": "[600, 600]"})
    assert "the pass holds 3/19 points of [600, 600] at (600, 600) in 2 frames" in caplog.text, caplog.text
