"""Offline contracts for the mission engine's pure parts.

No SAM, no render, no LLM: :func:`resolve_flight_target` (fid / xy token /
view-position sentinel / unknown) against a stubbed graph resolver.
"""

from benchmarks.citynav.run_benchmark.mission_engine import resolve_flight_target

_GRAPH = {
    "building_7": ((100.0, 200.0), "prior"),
    "detected_3": ((55.5, 66.25), "detected"),
}


def _node_xy(fid):
    return _GRAPH.get(fid)


_POSE = (10.0, 20.0)
_MARKER = (300.0, 400.0)


def _resolve(token, marker_xy=_MARKER):
    return resolve_flight_target(
        token, node_xy=_node_xy, pose_xy=_POSE, marker_xy=marker_xy
    )


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
    # maps.util.xy_token accepts the bracket-less form too.
    out = _resolve("1530.5, -12")
    assert out == {"xy": (1530.5, -12.0), "provenance": "point"}


def test_current_position_resolves_to_the_pose():
    out = _resolve("current_position")
    assert out == {"xy": _POSE, "provenance": "point"}


def test_marked_region_resolves_to_the_marker():
    out = _resolve("marked_region")
    assert out == {"xy": _MARKER, "provenance": "point"}


def test_marked_region_without_a_marker_is_an_error():
    out = _resolve("marked_region", marker_xy=None)
    assert "xy" not in out
    assert "marked region" in out["error"]


# ---------------------------------------------------------------------------
# declare_mission_concepts: the mission's act.sam_concepts channel arms the
# director on the first declaration and DIFFS it afterwards (latest-wins),
# re-arming the segmenter setter and the roles map each time.
# ---------------------------------------------------------------------------


def _bare_loop():
    from maps.search.tests.fakes import search_loop

    pushes: list[list] = []
    return search_loop(sam_setter=pushes.append), pushes


def test_declare_arms_the_director_on_first_call():
    loop, pushes = _bare_loop()
    receipt = loop.declare_mission_concepts([{"class": "car", "role": "target"}])
    assert receipt == {"declared": ["car (target)"]}
    d = loop.state.perception_director
    assert d is not None
    assert d.watchlist == ["car"]
    assert pushes[-1] == ["car"]


def test_redeclare_is_latest_wins_via_diff_not_rebuild():
    loop, pushes = _bare_loop()
    loop.declare_mission_concepts([{"class": "car", "role": "target"}])
    director_first = loop.state.perception_director
    loop.declare_mission_concepts([{"class": "building", "role": "anchor"}])
    d = loop.state.perception_director
    assert d is director_first, "a redeclare diffs the live director, never rebuilds"
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
    """Unset means NO clock, which is the field station's case and the
    default."""
    from benchmarks.citynav.run_benchmark.mission_engine import (
        MISSION_DEADLINE_ENV, _mission_deadline_epoch)

    monkeypatch.delenv(MISSION_DEADLINE_ENV, raising=False)
    assert _mission_deadline_epoch() is None

    monkeypatch.setenv(MISSION_DEADLINE_ENV, "1788050346.5")
    assert _mission_deadline_epoch() == 1788050346.5


def test_a_live_flight_is_served_at_the_target_it_asked_for(stub_session):
    """Under the station's ``observe`` the observation is the aircraft's
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
    assert flights[-1]["pose_xy"] == flights[-1]["resolved_xy"] == [100.0, 200.0]
    assert session.h.state.pose_xy == (100.0, 200.0)
    assert session.ctx.pose_xy == (100.0, 200.0), "the pose echo follows the flight"
    assert "error" in session.fly({"target_fid": "nowhere"}) and len(flights) == 1


def test_a_view_position_flight_flies_the_pose_or_the_marked_region(stub_session):
    session = stub_session(markers=[(300.0, 400.0, 0.0), (500.0, 600.0, 0.0)])
    session.fly({"target_fid": "current_position"})
    assert session.flights[-1]["resolved_xy"] == [10.0, 20.0]
    assert session.flights[-1]["provenance"] == "point"
    session.fly({"target_fid": "marked_region"})
    assert session.flights[-1]["resolved_xy"] == [400.0, 500.0]
    unmarked = stub_session()
    assert "error" in unmarked.fly({"target_fid": "marked_region"}) and not unmarked.flights


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
                                search_loop=SimpleNamespace(_ortho_ground_z_m=100.0))
    session.run("find it")
    assert "at the 1,640 ft AGL this run flies" in served["sensor_description"], served["sensor_description"]
