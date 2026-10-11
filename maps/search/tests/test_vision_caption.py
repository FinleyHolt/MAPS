"""The delivered-image caption states each view's TRUE orientation."""
from __future__ import annotations

from maps.search.parse_compile.reasoner_loop import _look_vision_text

_LEGEND = [{"digit": 1, "label": "car c1", "role": "center"}]


def test_unrotated_crop_says_north_up_on_its_line_not_the_header():
    t = _look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                            "context_m": 15.0}])
    assert "(15 m window; north-up)" in t
    assert "north-up" not in t.splitlines()[0]


def test_rotated_crop_states_true_up_never_north_up():
    t = _look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                            "context_m": 20.0,
                            "frame_up": [1.0, 0.0],
                            "frame_note": "image-up = east"}])
    assert "up 90 deg -- image-up = east" in t
    assert "north-up" not in t


def test_south_up_bearing_is_180():
    t = _look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                            "context_m": 20.0,
                            "frame_up": [0.0, -1.0]}])
    assert "up 180 deg" in t
