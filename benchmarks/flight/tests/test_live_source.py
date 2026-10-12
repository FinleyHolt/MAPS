"""The live observation: coverage off the ring's index, the fused observation, and registration against the map."""

from __future__ import annotations

import math

import numpy as np

from benchmarks.flight import map_render
from benchmarks.flight.live_source import frame_pose, register_frame
from benchmarks.flight.tile_georef import local_to_lonlat
from benchmarks.flight.vehicle import Leg


def _disc(tiles, x, y, r=60.0, n=12):
    pts = [(x, y)] + [(x + r * math.cos(2 * math.pi * k / n), y + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    return [local_to_lonlat(tiles, "calcity", px, py) for px, py in pts]


def _fly_over(twin, point, seconds):
    twin.post(Leg(target="t", point=point, step=0))
    twin.sleep(seconds)


def test_coverage_holds_the_ground_a_pass_flew_and_not_the_ground_it_did_not(built, make_twin, make_live):
    tiles = built[0]
    twin = make_twin()
    live = make_live(twin)
    _fly_over(twin, (600.0, 600.0), 8.0)          # 660 m at 82 m/s: over the point and on
    flown = live.coverage(_disc(tiles, 600.0, 600.0))
    assert flown["fraction"] == 1.0 and all(b is not None for b in flown["best"])
    elsewhere = live.coverage(_disc(tiles, 200.0, 900.0))
    assert elsewhere["fraction"] == 0.0


def test_a_ring_that_cannot_be_asked_is_none_with_its_reason(built, make_twin, make_live):
    twin = make_twin()
    live = make_live(twin)
    twin.index = lambda since=None: None
    twin.last_error = "timed out"
    assert live.coverage(_disc(built[0], 600.0, 600.0)) is None
    assert "timed out" in live.last_refusal


def test_the_fused_observation_is_the_ground_north_up_where_the_map_has_it(built, make_twin, make_live):
    tiles, _raster, ortho, _sensor = built
    twin = make_twin()
    live = make_live(twin)
    _fly_over(twin, (600.0, 600.0), 8.0)
    obs = live.observe_area(_disc(tiles, 600.0, 600.0))
    assert obs is not None and obs.ortho_view_corners_xy is not None
    x0, y0, x1, y1 = obs.world_window
    assert x0 < 540 and x1 > 660 and y0 < 540 and y1 > 660
    assert live.last["coverage"]["fraction"] == 1.0 and live.last["mosaic"]["placed"] >= 1
    # Where the frames landed, their pixels are the map's own at that ground.
    truth = map_render.window_render(obs.world_window, obs.rgb.shape[:2], ortho).astype(np.float32)
    held = obs.alpha > 0
    err = np.abs(obs.rgb.astype(np.float32) - truth)[held].mean()
    assert err < 12.0, err


def test_registration_reads_a_pose_error_as_the_shift_that_undoes_it(built, make_twin, make_live):
    """A frame rendered at the true pose and registered at a pose 12 m east
    and 7 m south of it measures the shift back onto the truth."""
    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 6.0)
    jpeg, t_host, _log, seq, tag = twin.frame(len(twin.rows) - 1)
    from benchmarks.flight.nav_track import tag_pose

    posed = live._posed(jpeg, t_host, pose=tag_pose(tag, t_host))
    posed.pose_xy = (posed.pose_xy[0] + 12.0, posed.pose_xy[1] - 7.0)
    record, _measured = register_frame(live.register, posed.rgb, frame_pose(posed), posed.sensor,
                                       live.ground_msl(posed.pose_xy))
    assert record["applied"] and record["source"] == "measured", record
    assert abs(record["shift_e_m"] + 12.0) < 1.5 and abs(record["shift_n_m"] - 7.0) < 1.5, record


def test_a_frame_off_the_map_is_refused_by_name(built, make_twin, make_live):
    twin = make_twin()
    live = make_live(twin, register=True)
    jpeg, t_host, _log, _seq, tag = twin.frame(0)
    from benchmarks.flight.nav_track import tag_pose

    posed = live._posed(jpeg, t_host, pose=tag_pose(tag, t_host))
    posed.pose_xy = (-5000.0, -5000.0)
    record, _measured = register_frame(live.register, posed.rgb, frame_pose(posed), posed.sensor,
                                       live.ground_msl(posed.pose_xy))
    assert not record["applied"] and record["why"] == "the frame reaches off the map"


def test_a_featureless_map_is_no_match_and_the_frame_is_refused_by_its_response(built, make_twin, make_live):
    from dataclasses import replace

    from benchmarks.flight import frame_offset
    from benchmarks.flight.nav_track import tag_pose

    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 6.0)
    jpeg, t_host, _log, _seq, tag = twin.frame(len(twin.rows) - 1)
    posed = live._posed(jpeg, t_host, pose=tag_pose(tag, t_host))
    blank = np.full_like(posed.rgb, 128)
    assert frame_offset.offset(posed.rgb, blank) == (0.0, 0.0, 0.0)
    flat = replace(live.register, render=lambda s, *a, **k: np.full(
        (s.intrinsics.image_height, s.intrinsics.image_width, 3), 128, np.uint8))
    record, _measured = register_frame(flat, posed.rgb, frame_pose(posed), posed.sensor,
                                       live.ground_msl(posed.pose_xy))
    assert not record["applied"] and record["why"].startswith("the frame does not register: response 0.00"), record


def test_the_corrector_graph_places_a_leg_and_gates_a_ring_frame_by_its_nearest_node(built, make_twin, make_live):
    """Frames registered as they arrive are solved together; a ring frame
    whose own measurement sits further than the cap from the graph's node
    nearest it in time is refused and takes that node."""
    from benchmarks.flight.nav_track import tag_pose
    from benchmarks.flight.pose_graph import INNOVATION_CAP_M

    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 4.0)
    live.correct(twin.frame(0))
    assert live.graph.nodes == {}                       # a lone anchor folds into nothing
    for seq in range(1, len(twin.rows)):
        live.correct(twin.frame(seq))
    assert live.graph.nodes and live.graph.estimate(twin.rows[-1]["t_host"]) is not None
    seq = len(twin.rows) - 1
    jpeg, t_host, _log, _seq, tag = twin.frame(seq)
    posed = live._posed(jpeg, t_host, pose=tag_pose(tag, t_host))
    posed.pose_xy = (posed.pose_xy[0] + INNOVATION_CAP_M + 20.0, posed.pose_xy[1])
    record = live._register_ring(seq, posed, t_host)
    assert record["source"] == "nearest" and abs(record["shift_e_m"]) < 2.0, record
    assert record["why"].startswith("disagrees with the graph's nearest node"), record


def test_a_simulated_vehicle_corrects_every_frame_up_to_its_clock_in_order(built, make_twin, monkeypatch):
    """The twin's clock jumps when the harness sleeps, so its corrector runs
    off the clock: every frame up to it, in order, before sleep returns."""
    from benchmarks.flight.live_source import LiveSource, open_live

    seen: list[int] = []
    monkeypatch.setattr(LiveSource, "correct", lambda self, got: seen.append(got[3]))
    tiles, raster, _ortho, sensor = built
    twin = make_twin()
    live = open_live(vehicle=twin, tiles_dir=tiles, map_name="calcity", sensor=sensor, register_raster=raster)
    try:
        twin.sleep(2.0)
        assert seen == [r["seq"] for r in twin.rows] and seen
        twin.sleep(1.0)
        assert seen == [r["seq"] for r in twin.rows]
    finally:
        live.close()


def test_a_ring_frame_reads_the_graph_under_the_corrector_s_lock(built, make_twin, make_live):
    from benchmarks.flight.nav_track import tag_pose

    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 2.0)
    for seq in range(len(twin.rows)):
        live.correct(twin.frame(seq))
    seq = len(twin.rows) - 1
    jpeg, t_host, _log, _seq, tag = twin.frame(seq)
    posed = live._posed(jpeg, t_host, pose=tag_pose(tag, t_host))
    estimates = []
    real = live.graph.estimate
    live.graph.estimate = lambda t: estimates.append(live._correcting.locked()) or real(t)
    live._register_ring(seq, posed, t_host)
    assert estimates == [True]


def test_the_mosaic_marks_imaged_only_the_ground_a_frame_lands_on():
    """A frame turned 45 degrees covers half its bounding square; the rest is
    unobserved and the alpha says so."""
    import math

    import numpy as np

    from benchmarks.flight.live_source import composite

    r = 50.0 * math.sqrt(2.0)
    corners = np.asarray([[[0.0, r], [r, 0.0], [0.0, -r], [-r, 0.0]]])   # a diamond, 100 m on a side
    _rgb, alpha, _extent, placed = composite(corners, [np.full((100, 100, 3), 200, np.uint8)], gsd_m=1.0)
    assert placed == 1
    share = float((alpha > 0).mean())
    assert abs(share - 0.5) < 0.03, share
