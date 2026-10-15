"""The live observation: coverage off the ring's index, the fused observation, and registration against the map."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.flight import map_render
from benchmarks.flight.live_source import register_frame
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


def test_coverage_of_each_reads_the_ring_as_it_stands_and_not_frames_it_has_evicted(built, make_twin, make_live):
    tiles = built[0]
    twin = make_twin()
    live = make_live(twin)
    _fly_over(twin, (600.0, 600.0), 8.0)
    area = _disc(tiles, 600.0, 600.0)
    (held,) = live.coverage_of_each([area])
    assert held["fraction"] == 1.0
    twin.rows.clear()                             # the ring has evicted every frame of that pass
    (gone,) = live.coverage_of_each([area])
    assert gone["fraction"] == 0.0 == live.coverage(area)["fraction"]


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
    assert live.last["coverage"]["held"] == live.last["coverage"]["points"] and live.last["mosaic"]["placed"] >= 1
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

    posed = live._posed(jpeg, t_host, pose=tag_pose(tag))
    posed.cam = replace(posed.cam, pose_xy=(posed.cam.pose_xy[0] + 12.0, posed.cam.pose_xy[1] - 7.0))
    record, _measured = register_frame(live.register, posed.rgb, posed.cam, posed.sensor,
                                       live.ground_msl(posed.cam.pose_xy))
    assert record["applied"] and record["source"] == "measured", record
    assert abs(record["shift_e_m"] + 12.0) < 1.5 and abs(record["shift_n_m"] - 7.0) < 1.5, record


def test_a_frame_off_the_map_is_refused_by_name(built, make_twin, make_live):
    twin = make_twin()
    live = make_live(twin, register=True)
    jpeg, t_host, _log, _seq, tag = twin.frame(0)
    from benchmarks.flight.nav_track import tag_pose

    posed = live._posed(jpeg, t_host, pose=tag_pose(tag))
    posed.cam = replace(posed.cam, pose_xy=(-5000.0, -5000.0))
    record, _measured = register_frame(live.register, posed.rgb, posed.cam, posed.sensor,
                                       live.ground_msl(posed.cam.pose_xy))
    assert not record["applied"] and record["why"] == "the frame reaches off the map"


def test_a_featureless_map_is_no_match_and_the_frame_is_refused_by_its_response(built, make_twin, make_live):
    from benchmarks.flight import frame_offset
    from benchmarks.flight.nav_track import tag_pose

    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 6.0)
    jpeg, t_host, _log, _seq, tag = twin.frame(len(twin.rows) - 1)
    posed = live._posed(jpeg, t_host, pose=tag_pose(tag))
    blank = np.full_like(posed.rgb, 128)
    assert frame_offset.offset(posed.rgb, blank) == (0.0, 0.0, 0.0)
    flat = replace(live.register, render=lambda s, *a, **k: np.full(
        (s.intrinsics.image_height, s.intrinsics.image_width, 3), 128, np.uint8))
    record, _measured = register_frame(flat, posed.rgb, posed.cam, posed.sensor,
                                       live.ground_msl(posed.cam.pose_xy))
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
    posed = live._posed(jpeg, t_host, pose=tag_pose(tag))
    posed.cam = replace(posed.cam, pose_xy=(posed.cam.pose_xy[0] + INNOVATION_CAP_M + 20.0, posed.cam.pose_xy[1]))
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


def test_a_ring_frame_is_posed_on_the_corrector_s_time_base(built, make_twin, make_live):
    """With a boresight timing offset the corrector poses each frame at its
    stamp plus the offset; a ring frame posed the same way registers where
    the graph's nodes do and keeps its own measurement."""

    twin = make_twin()
    live = make_live(twin, register=True)
    live.boresight = replace(live.boresight, time_offset_s=0.6)
    _fly_over(twin, (600.0, 600.0), 4.0)
    for seq in range(len(twin.rows)):
        live.correct(twin.frame(seq))
    assert live._fused([(0.0, -(len(twin.rows) // 2))], ranked=1, about="a frame mid-pass") is not None
    (record,) = live.last["mosaic"]["registrations"]
    assert record["applied"] and record["source"] == "measured", record


def test_a_ring_frame_reads_the_graph_under_the_corrector_s_lock(built, make_twin, make_live):
    from benchmarks.flight.nav_track import tag_pose

    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 2.0)
    for seq in range(len(twin.rows)):
        live.correct(twin.frame(seq))
    seq = len(twin.rows) - 1
    jpeg, t_host, _log, _seq, tag = twin.frame(seq)
    posed = live._posed(jpeg, t_host, pose=tag_pose(tag))
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


def test_a_truncated_ring_frame_is_refused_by_name_and_never_raised(built, make_twin, make_live):
    """A JPEG torn over the link neither kills the corrector nor raises out of
    the observation: the frame is refused with its reason. The corrector's own
    refusal never overwrites the reason the mission's last ask left."""
    twin = make_twin()
    live = make_live(twin, register=True)
    _fly_over(twin, (600.0, 600.0), 8.0)
    fetch = twin.frame
    twin.frame = lambda seq: (lambda g: None if g is None else (g[0][: len(g[0]) // 3],) + g[1:])(fetch(seq))
    live.last_refusal = "the mission's own"
    live.correct(twin.frame(0))
    assert live.last_refusal == "the mission's own"
    assert live.observe_area(_disc(built[0], 600.0, 600.0)) is None
    assert "the frame does not decode" in live.last_refusal


@pytest.mark.parametrize("off", [-0.1, 0.1])
def test_the_window_render_slices_the_pixels_whose_centres_the_window_holds(off):
    """The engine's raster convention: column i's centre is x_min + i/ppm,
    row r's is y_min + (rows - 1 - r)/ppm. A window around the centres
    x, y in {2, 3, 4} on a 1 m raster renders those nine pixels."""
    from benchmarks.flight.tile_georef import Bounds

    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    a, b = 1.5 + off, 4.5 + off
    got = map_render.window_render((a, a, b, b), (3, 3), (rgb, Bounds(0.0, 0.0, 8.0, 8.0), 1.0))
    assert np.array_equal(got, rgb[3:6, 2:5])


def test_the_terrain_reads_the_cell_a_point_lies_in_counted_from_the_north_west_corner():
    """Cells are 10 m from (x_min, y_max); the grid's south edge need not sit on y_min."""
    grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = map_render.Terrain(grid=grid, x_min=0.0, y_max=20.4, cell_m=10.0)
    x = np.array([0.1, 9.9, 10.1, 29.9, 4.0, 15.0])
    y = np.array([20.3, 10.5, 10.3, 0.5, 10.6, -5.0])
    assert t.elevation(x, y).tolist() == [1.0, 1.0, 5.0, 6.0, 1.0, 5.0]


def test_a_raster_directory_without_the_pair_loads_none_and_the_live_source_names_the_pair(built, make_twin,
                                                                                           tmp_path):
    import pytest

    from benchmarks.flight.live_source import open_live

    (tmp_path / "calcity_rgb.tif").write_bytes(b"")           # no calcity.tif beside it
    assert map_render.load_raster(tmp_path, "calcity") is None
    with pytest.raises(ValueError, match="calcity_rgb.tif .or calcity.png. beside calcity.tif"):
        open_live(vehicle=make_twin(), tiles_dir=built[0], map_name="calcity", sensor=built[3],
                  register_raster=tmp_path)


def test_the_simulated_clock_asks_a_frame_that_cannot_be_fetched_once():
    """Under the simulated clock each recorded frame is asked once: a frame
    that cannot be fetched is passed over, not asked again at every sleep."""
    import threading
    import types

    from benchmarks.flight.live_source import LiveSource

    class Ring:
        last_error = ""

        def __init__(self):
            self.rows, self.asked = [], 0

        def index(self, since=None):
            return [r for r in self.rows if since is None or r["log_time"] >= since]

        def frame(self, seq):
            self.asked += 1
            return None if self.rows[seq]["off"] else (b"", 0.0, self.rows[seq]["log_time"], seq, {})

    ring = Ring()
    src = types.SimpleNamespace(vehicle=ring, correct=lambda got: None, _clocked_since=None, _asking=threading.Lock())
    src._ask = types.MethodType(LiveSource._ask, src)
    src._correct_since = types.MethodType(LiveSource._correct_since, src)
    step = types.MethodType(LiveSource._correct_through_now, src)
    for i in range(105):
        ring.rows.append({"seq": i, "log_time": i * 100, "off": i >= 5})
        step()
    assert ring.asked == 105


def test_a_wall_clock_vehicle_registers_the_newest_frame_each_period_until_closed(built, make_live, monkeypatch):
    """A vehicle without a simulated clock (no ``on_sleep``) is corrected on a
    thread: each period it takes the newest frame the ring holds, asks it
    again until it is served, and stops when the source closes."""
    import time

    from benchmarks.flight import live_source

    monkeypatch.setattr(live_source, "CORRECTOR_PERIOD_S", 0.01)

    class Ring:
        last_error = ""

        def __init__(self):
            self.rows, self.served, self.asked = [], set(), []

        def index(self, since=None):
            return [r for r in self.rows if since is None or r["log_time"] >= since]

        def frame(self, seq):
            self.asked.append(seq)
            return (b"", 0.0, self.rows[seq]["log_time"], seq, {}) if seq in self.served else None

    def until(cond):
        deadline = time.monotonic() + 5.0
        while not cond():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    ring = Ring()
    live = make_live(ring, register=True)
    seen: list[int] = []
    live.correct = lambda got: seen.append(got[3])
    ring.rows += [{"seq": i, "log_time": i * 100} for i in range(3)]
    live.start_corrector()
    try:
        until(lambda: ring.asked.count(2) >= 3)
        assert set(ring.asked) == {2} and seen == []
        ring.served.add(2)
        until(lambda: seen == [2])
        ring.rows += [{"seq": i, "log_time": i * 100} for i in (3, 4)]
        ring.served.update((3, 4))
        until(lambda: seen == [2, 4])
    finally:
        live.close()
    time.sleep(0.05)                    # a period in flight at the close ends
    asked = len(ring.asked)
    time.sleep(0.05)
    assert len(ring.asked) == asked and seen == [2, 4]


def test_a_frame_not_fetched_is_quoted_with_its_own_reason_while_the_corrector_fetches(make_live):
    """The wall-clock corrector's fetch lands between the fused observation's
    failed fetch and the reason it quotes: the reason is still the fetch's own."""
    import threading

    threads = []

    class Ring:
        last_error = ""

        def index(self, since=None):
            return [{"seq": 1, "log_time": 1}]

        def frame(self, seq):
            if threading.current_thread().name == "corrector":
                self.last_error = ""          # a fetch that went through
                return None
            self.last_error = "link down"
            threads.append(threading.Thread(target=live._correct_since, args=(None,),
                                            kwargs={"newest_only": True}, name="corrector"))
            threads[0].start()
            threads[0].join(0.3)
            return None

    live = make_live(Ring())
    assert live._fused([(0.0, -5)], ranked=1, about="the area") is None
    threads[0].join()
    assert "ring frame 5 not fetched: link down" in live.last_refusal


def test_the_observation_is_keyed_to_its_newest_frame_by_log_time_not_seq(built, make_twin, make_live):
    """A camera restart starts its sequence again: the frame the observation
    is keyed to is the newest by its stamp, whatever its sequence number."""
    twin = make_twin()
    live = make_live(twin)
    _fly_over(twin, (600.0, 600.0), 2.0)
    last = len(twin.rows) - 1
    twin.rows[0]["log_time"] = twin.rows[last]["log_time"] + 1     # frame 0 recorded after a restart
    assert live._fused([(0.0, 0), (0.0, -last)], ranked=2, about="two frames") is not None
    assert live.last["seq"] == 0 and live.last["log_time"] == twin.rows[0]["log_time"]
