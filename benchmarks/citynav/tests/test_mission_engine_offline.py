"""Offline contracts for the mission engine's pure parts.

No SAM, no render, no LLM: :func:`resolve_flight_target` (fid / xy token /
unknown) against a stubbed graph resolver.
"""

from benchmarks.citynav.run_benchmark.mission_engine import resolve_flight_target
from maps.settings import Settings

_GRAPH = {
    "building_7": ((100.0, 200.0), "prior"),
    "detected_3": ((55.5, 66.25), "detected"),
}


def _node_xy(fid):
    return _GRAPH.get(fid)


def _resolve(token):
    return resolve_flight_target(token, node_xy=_node_xy)


def test_prior_fid_resolves_to_centroid():
    out = _resolve("building_7")
    assert out == {"xy": (100.0, 200.0), "provenance": "prior"}


def test_detected_fid_resolves_with_detected_provenance():
    out = _resolve("detected_3")
    assert out == {"xy": (55.5, 66.25), "provenance": "detected"}


def test_unknown_fid_is_an_error_not_a_flight():
    out = _resolve("building_999")
    assert "xy" not in out
    assert "building_999" in out["error"]
    assert "not on the surface" in out["error"]


def test_xy_token_is_a_point():
    out = _resolve("[1530, 1378]")
    assert out == {"xy": (1530.0, 1378.0), "provenance": "point"}


def test_bare_xy_token_matches_the_one_grammar():
    # maps.xy_token accepts the bracket-less form too.
    out = _resolve("1530.5, -12")
    assert out == {"xy": (1530.5, -12.0), "provenance": "point"}


# ---------------------------------------------------------------------------
# declare_mission_concepts: each declaration replaces the director's
# watchlist with the declared union (latest wins) and pushes it.
# ---------------------------------------------------------------------------


def _bare_loop():
    from maps.mission.tests.fakes import episode_world

    pushes: list[list] = []
    return episode_world(sam_setter=pushes.append), pushes


def test_declare_arms_the_director_on_first_call():
    loop, pushes = _bare_loop()
    assert loop.declare_mission_concepts(["car"]) is None
    d = loop.state.segmenter_tasking
    assert d is not None
    assert d.watchlist == ["car"]
    assert pushes[-1] == ["car"]


def test_redeclare_is_latest_wins_on_the_same_director():
    loop, pushes = _bare_loop()
    loop.declare_mission_concepts(["car"])
    director_first = loop.state.segmenter_tasking
    loop.declare_mission_concepts(["building"])
    d = loop.state.segmenter_tasking
    assert d is director_first, "a redeclare updates the live director, never rebuilds"
    assert d.watchlist == ["building"]
    assert pushes[-1] == ["building"]


def test_the_pose_echo_follows_a_flight_and_ignores_an_error():
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark.mission_engine import echo_pose

    ctx = SimpleNamespace(pose_xy=None)
    echo_pose(ctx, {"pose_xy": [10.0, 20.0], "alt_m": 120.0})
    assert ctx.pose_xy == (10.0, 20.0)
    ctx2 = SimpleNamespace(pose_xy=None)
    echo_pose(ctx2, {"error": "unknown fid"})
    assert ctx2.pose_xy is None


def test_the_mission_deadline_is_read_per_call_and_optional(monkeypatch):
    """Unset means NO clock, which is the flight mission's case and the
    default."""
    from benchmarks.run_record.episodes import DEADLINE_ENV
    from benchmarks.citynav.run_benchmark.mission_engine import _mission_deadline_epoch

    assert _mission_deadline_epoch({}) is None
    assert _mission_deadline_epoch({DEADLINE_ENV: "1788050346.5"}) == 1788050346.5


def test_a_live_flight_is_served_at_the_target_it_asked_for(stub_session):
    """Under the flight mission's ``observe`` the observation is the aircraft's
    frame, perceived from the pose it was taken at, and the stop the model
    is served is the target it named: the report, the flight record and the
    engine's pose all carry the target's own point, and no note says where
    the frame was. A token that does not resolve is an error, as on the
    bench: there is nothing to arrive at."""
    folded: list = []
    session = stub_session(update_perception=lambda **kw: folded.append(kw) or [])
    report = session.fly({"target_fid": "building_7"})
    assert len(folded) == 1, "the frame is folded once"
    assert report["pose_xy"] == [100.0, 200.0] and "note" not in report
    assert report["answer_if_you_reply_now"] == "building_7"
    flights = session.flights
    assert flights[-1]["resolved_xy"] == [100.0, 200.0]
    assert session.h.state.pose_xy == (100.0, 200.0)
    assert session.ctx.pose_xy == (100.0, 200.0), "the pose echo follows the flight"
    assert "error" in session.fly({"target_fid": "nowhere"}) and len(flights) == 1


def test_an_end_commit_withhold_flight_report_names_no_answer(stub_session):
    session = stub_session(withhold=["end_commit"], update_perception=lambda **kw: [])
    report = session.fly({"target_fid": "building_7"})
    assert report["target"] == "building_7" and "answer_if_you_reply_now" not in report
    assert session.flights[-1]["target"] == "building_7", "the commit is still the last flight"


def test_a_live_flight_with_no_frame_says_the_recording_served_none(stub_session):
    session = stub_session()
    session.observe = lambda: None
    assert session.fly({"target_fid": "building_7"}) == {
        "error": "no observation: the aircraft's recording served no frame of this flight's ground"}
    assert session.flights == []


def test_a_failing_boost_director_never_kills_a_flight(stub_session):
    def boom(*a, **k):
        raise RuntimeError("director down")

    session = stub_session(start_flight=boom, record_flown_stop=boom)
    report = session.fly({"target_fid": "detected_3"})
    assert "error" not in report and report["target"] == "detected_3"
    assert session.flights[-1]["provenance"] == "detected"


def test_the_sensor_block_states_the_footprint_at_the_height_the_mission_flies(monkeypatch):
    """The served footprint is the sensor's own at the mission's start height
    above ground (after the altitude cap), not the config's nominal one."""
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark import mission_engine as me
    from maps.perception.geometry.sensor_spec import resolve_sensor

    served = {}
    monkeypatch.setattr(me, "endpoint_spec", lambda: SimpleNamespace(base_url="http://unused"))
    monkeypatch.setattr(me, "run_mission", lambda **kw: served.update(kw))
    session = me.MissionSession.__new__(me.MissionSession)
    session.run_kwargs = {}
    session.h = SimpleNamespace(sensor=resolve_sensor("oryx_orx32_25mm_nadir"),
                                cfg=SimpleNamespace(start_altitude_agl_m=914.4),
                                state=SimpleNamespace(altitude_m=600.0),
                                ground_z_m=100.0, bench=SimpleNamespace(environ={}),
                                settings=Settings(), workdir=None, routing_key=None)
    session.run("find it")
    assert "at the 1,640 ft AGL this run flies" in served["sensor_description"], served["sensor_description"]
