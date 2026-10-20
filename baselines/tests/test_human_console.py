"""The operator console flies the clicks it is sent and commits the operator's point.

A click is a fraction of the frame, so the stop it flies must be the ground
point under it through the crop's own corners; a move past the budget must be
refused without spending it; the commit ends the episode at the clicked point.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.human.policy import run_episode, world_to_px
from baselines.seam import BaselineHarness
from benchmarks.citynav.run_benchmark.episode_runner import EpisodeState
from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.observation import Observation


def _crop(half_of):
    def crop(xy, alt):
        x, y = xy
        half = half_of(alt)
        x0, y0, x1, y1 = x - half, y - half, x + half, y + half
        return Observation(pose_xy=xy, altitude_m=alt, world_window=(x0, y0, x1, y1),
                           rgb=np.zeros((101, 101, 3), np.uint8),
                           ortho_view_corners_xy=((x0, y1), (x1, y1), (x1, y0), (x0, y0)))
    return crop


def _harness(tmp_path, max_steps: int = 200) -> BaselineHarness:
    hyd = SimpleNamespace(
        crop=None,
        cfg=SimpleNamespace(max_steps=max_steps, altitude_cap_m=150.0),
        episode=SimpleNamespace(episode_id="unit::1#0", map_name="unit_block_0",
                                description="the car by the depot", start_xy=(0.0, 0.0),
                                start_heading_deg=0.0),
        assets=SimpleNamespace(base_features={"buildings": [
            {"name": "Tram Depot", "world_contour": [[10, 10], [40, 10], [40, 30], [10, 30]]}]}),
        state=EpisodeState(pose_xy=(0.0, 0.0), altitude_m=50.0),
        ground_z_m=0.0,
        sensor=resolve_sensor("nadir_1008_90"),
        perception=None,
    )
    h = BaselineHarness(hyd, "human", tmp_path)
    h._crop = _crop(h.view_half_m)
    return h


def _fly(h, monkeypatch, actions: list[dict]) -> tuple:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("HUMAN_CONSOLE_PORT", str(port))
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(result=run_episode(h)), daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "state", timeout=2)
            break
        except OSError:
            time.sleep(0.05)
    replies = []
    for a in actions:
        req = urllib.request.Request(url, data=json.dumps(a).encode(), method="POST")
        try:
            replies.append(json.load(urllib.request.urlopen(req, timeout=5)))
        except urllib.error.HTTPError as err:
            replies.append(json.load(err))
    t.join(timeout=5)
    return out["result"], replies


def test_world_to_px_inverts_the_crop_corners(tmp_path):
    h = _harness(tmp_path)
    obs = h.observe((5.0, -3.0), 50.0)
    x, y = h.pixel_to_world(obs, 20.0, 70.0)
    assert world_to_px(obs.ortho_view_corners_xy, obs.rgb.shape[:2], (x, y)) == pytest.approx((70.0, 20.0))


def test_the_console_flies_clicks_and_commits_the_clicked_point(tmp_path, monkeypatch):
    h = _harness(tmp_path)
    half = h.view_half_m(50.0)
    result, replies = _fly(h, monkeypatch, [
        {"action": "move", "u": 0.5, "v": 0.0, "alt_m": 50.0},  # the frame's top edge: due north
        {"action": "move", "alt_m": 80.0},  # altitude only
        {"action": "commit", "u": 1.0, "v": 0.5},  # the frame's right edge: due east
    ])
    assert [r["ok"] for r in replies] == [True, True, True]
    (xy0, alt0), (xy1, alt1) = result.stops
    assert xy0 == pytest.approx((0.0, half)) and alt0 == 50.0
    assert xy1 == pytest.approx(xy0) and alt1 == 80.0
    assert result.commit_xy == pytest.approx((h.view_half_m(80.0), half))
    assert result.commit_tier == "human:operator"
    assert (tmp_path / "human" / "stop_1.jpg").exists()
    assert len(json.loads((tmp_path / "human" / "session.json").read_text())["actions"]) == 3


def test_a_searched_landmark_is_flown_over_at_its_centroid(tmp_path, monkeypatch):
    h = _harness(tmp_path)
    (depot,) = h.landmarks()
    result, replies = _fly(h, monkeypatch, [
        {"action": "move", "x": depot.centroid[0], "y": depot.centroid[1], "alt_m": 60.0},
        {"action": "commit"},
    ])
    assert [r["ok"] for r in replies] == [True, True]
    assert result.stops == [(pytest.approx(depot.centroid), 60.0)]
    assert result.commit_xy == pytest.approx(depot.centroid)


def test_a_move_past_the_budget_is_refused_and_spends_nothing(tmp_path, monkeypatch):
    h = _harness(tmp_path, max_steps=1)
    result, replies = _fly(h, monkeypatch, [
        {"action": "move", "u": 0.5, "v": 0.0, "alt_m": 50.0},
        {"action": "commit"},
    ])
    assert replies[0]["ok"] is False and replies[1]["ok"] is True
    assert result.stops == [] and h.budget.used == 0
    assert result.commit_xy == pytest.approx((0.0, 0.0))
    assert result.extra_metrics == {"moves_refused": 1}
