"""The rig's served footprint is the sensor's own at the height the config flies."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.run_record.config import load_config
from maps.perception.geometry.sensor_spec import resolve_sensor, served_description
from maps.perception.ortho.sensor_frame import footprint_m
from maps.mission.served_text import served_system_prompt

_CALCITY = Path(__file__).resolve().parents[3] / "configs" / "calcity.yaml"


def test_the_rigs_footprint_prose_is_footprint_m_at_the_configs_agl():
    cfg = load_config(_CALCITY)
    sensor = resolve_sensor(cfg.sensor)
    text = served_system_prompt(served_description(sensor, agl_m=cfg.start_altitude_agl_m))
    m = re.search(r"at the ([\d,]+) ft AGL this run flies, the footprint is ~(\d+) x (\d+) m", text)
    # VACUITY FLOOR: the rig block states a footprint at all.
    assert m, text
    agl_m = int(m.group(1).replace(",", "")) * 0.3048
    assert abs(agl_m - cfg.start_altitude_agl_m) < 0.3048 / 2
    w, h = footprint_m(sensor, agl_m)
    assert (int(m.group(2)), int(m.group(3))) == (round(w), round(h)), (m.group(0), (w, h))
