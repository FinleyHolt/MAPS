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
    import threading

    from maps.search import search_loop as sl

    loop = object.__new__(sl.SearchLoop)
    loop._lock = threading.RLock()
    loop._state = sl._State(runtime_graph=None)
    pushes: list[list] = []
    loop._sam_concept_setter = pushes.append
    loop._sam_watchlist_setter = None
    return loop, pushes


def test_declare_arms_the_director_on_first_call():
    loop, pushes = _bare_loop()
    receipt = loop.declare_mission_concepts([{"class": "car", "role": "target"}])
    assert receipt == {"declared": ["car (target)"]}
    d = loop.state.perception_director
    assert d is not None
    assert d.watchlist == ["car"]
    assert pushes[-1] == ["car"]
    assert "car" in loop.state.sam_class_to_roles


def test_redeclare_is_latest_wins_via_diff_not_rebuild():
    loop, pushes = _bare_loop()
    loop.declare_mission_concepts([{"class": "car", "role": "target"}])
    director_first = loop.state.perception_director
    loop.declare_mission_concepts([{"class": "building", "role": "anchor"}])
    d = loop.state.perception_director
    assert d is director_first, "a redeclare diffs the live director, never rebuilds"
    assert d.watchlist == ["building"]
    assert pushes[-1] == ["building"]
    assert "building" in loop.state.sam_class_to_roles
    assert "car" not in loop.state.sam_class_to_roles


def test_chase_live_graph_follows_the_replaced_runtime_graph():
    # The ingest wire replaces state.runtime_graph per fold; the ToolContext
    # must chase the new object or every analyze reads the pre-flight world
    # (the first-smoke regression: 114 folded, runtime finds len 0).
    import threading
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark.mission_engine import chase_live_graph

    old_graph, new_graph = object(), object()
    sl = SimpleNamespace(_lock=threading.Lock(),
                         state=SimpleNamespace(runtime_graph=new_graph))
    ctx = SimpleNamespace(graph=old_graph, robot_pose=None)
    chase_live_graph(ctx, sl, {"pose_xy": [10.0, 20.0], "alt_m": 120.0},
                     fallback_alt=100.0, heading_deg=0.0)
    assert ctx.graph is new_graph
    assert ctx.robot_pose is not None and ctx.robot_pose.x == 10.0
    # An errored flight changes nothing.
    ctx2 = SimpleNamespace(graph=old_graph, robot_pose=None)
    chase_live_graph(ctx2, sl, {"error": "unknown fid"},
                     fallback_alt=100.0, heading_deg=0.0)
    assert ctx2.graph is old_graph and ctx2.robot_pose is None


def test_chase_after_boost_follows_the_replaced_runtime_graph():
    # A retro boost folds through the SAME ingest wire as a flight and
    # replaces state.runtime_graph; an un-chased ToolContext keeps reading
    # the pre-boost world (pilot-2: 19/28 productive boosts dead-ended --
    # "n_new: 479" then "Live now: 0 detection fid(s)").
    import threading
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark.mission_engine import chase_after_boost

    old_graph, new_graph = object(), object()
    sl = SimpleNamespace(_lock=threading.Lock(),
                         state=SimpleNamespace(runtime_graph=new_graph))
    ctx = SimpleNamespace(graph=old_graph, robot_pose=None)
    chase_after_boost(ctx, sl, {"retro": [{"n_new": 2}]})
    assert ctx.graph is new_graph
    assert ctx.robot_pose is None  # a boost moves nothing
    # an errored or shapeless boost changes nothing
    for res in ({"error": "not wired"}, None, "queued"):
        ctx2 = SimpleNamespace(graph=old_graph, robot_pose=None)
        chase_after_boost(ctx2, sl, res)
        assert ctx2.graph is old_graph


def test_the_mission_deadline_is_read_per_call_and_optional(monkeypatch):
    """The warm runner flies many episodes in one process, each with its own
    deadline, so this is read at CALL time -- an import-time read would hand
    episode 40 episode 1's clock. Unset means NO clock, which is the field
    station's case and the default."""
    from benchmarks.citynav.run_benchmark.mission_engine import (
        MISSION_DEADLINE_ENV, _mission_deadline_epoch)

    monkeypatch.delenv(MISSION_DEADLINE_ENV, raising=False)
    assert _mission_deadline_epoch() is None

    monkeypatch.setenv(MISSION_DEADLINE_ENV, "1788050346.5")
    assert _mission_deadline_epoch() == 1788050346.5

    # a wiring bug must not silently become "no clock" without saying so,
    # and must not refuse to fly either
    monkeypatch.setenv(MISSION_DEADLINE_ENV, "soon")
    assert _mission_deadline_epoch() is None


def test_a_live_flight_is_served_at_the_target_it_asked_for(monkeypatch):
    """Under the station's ``observe`` the observation is the aircraft's
    frame, perceived from the pose it was taken at, and the stop the model
    is served is the target it named: the report, the flight record and the
    engine's pose all carry the target's own point, and no note says where
    the frame was. A token that does not resolve is an error, as on the
    bench: there is nothing to arrive at."""
    from types import SimpleNamespace

    from benchmarks.citynav.episode import EpisodeState
    from benchmarks.citynav.run_benchmark import mission_engine as me
    from maps.search.parse_compile.tour_state import FeatureTour

    monkeypatch.setattr(me, "_write_tour_cursor", lambda *a, **k: None)
    folded: list = []
    loop = SimpleNamespace(
        activate_pending_boosts=lambda: None,
        update_perception=lambda **kw: folded.append(kw["pose_xy"]) or [],
        snapshot=lambda: None,
        retire_boosts_after_round=lambda rounds: None,
        last_flight_detections=lambda: [],
    )
    frame = SimpleNamespace(pose_xy=(130.0, 240.0), altitude_m=900.0, yaw_deg=45.0, rgb=None,
                            sensor=None, ortho_view_corners_xy=None)
    state = EpisodeState(pose_xy=_POSE, altitude_m=100.0, graph=None)
    flights: list = []
    common = dict(
        flights=flights, feature_tour=FeatureTour(), node_xy=_node_xy, marker_xy=None,
        cfg=SimpleNamespace(altitude_cap_m=1000.0), state=state,
        assets=SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0), cell_size_m=2.0,
                               grid_shape=(10, 10)),
        episode=SimpleNamespace(map_name="calcity", description="the largest building"),
        sensor=None, tile_frame=None, perception=None, search_loop=loop,
        recorder=None, osm_ctx=None, t_start=0.0, observe=lambda: frame,
    )
    report = me.fly_one({"target_fid": "building_7"}, **common)
    assert folded == [(130.0, 240.0)], "the fold reads the frame's own pose"
    assert report["pose_xy"] == [100.0, 200.0] and "note" not in report
    assert report["answer_if_you_reply_now"] == "building_7"
    assert flights[-1]["pose_xy"] == flights[-1]["resolved_xy"] == [100.0, 200.0]
    assert state.pose_xy == (100.0, 200.0)
    assert "error" in me.fly_one({"target_fid": "nowhere"}, **common) and len(flights) == 1
