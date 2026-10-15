"""The delivered-image caption states each view's TRUE orientation."""
from __future__ import annotations

from maps.search.parse_compile.views import look_vision_text

_LEGEND = [{"digit": 1, "label": "car c1", "role": "center"}]


def test_unrotated_crop_says_north_up_on_its_line_not_the_header():
    t = look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                            "context_m": 15.0}])
    assert "(15 m window; north-up)" in t
    assert "north-up" not in t.splitlines()[0]


def test_rotated_crop_states_true_up_never_north_up():
    t = look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                            "context_m": 20.0,
                            "frame_up": [1.0, 0.0],
                            "frame_note": "image-up = east"}])
    assert "up 90 deg -- image-up = east" in t
    assert "north-up" not in t


def test_south_up_bearing_is_180():
    t = look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                            "context_m": 20.0,
                            "frame_up": [0.0, -1.0]}])
    assert "up 180 deg" in t


def test_a_point_subject_is_named_by_its_legend_label():
    # A flight view of a point target carries no fid: its legend centre
    # label (the point) names it.
    t = look_vision_text([{"center_fid": "", "legend": [
        {"digit": 1, "label": "[3411, 1468]", "role": "center"}]}])
    assert "in order: [[3411, 1468]]" in t


def test_a_turned_view_says_its_corners_are_outside_the_window():
    turned = look_vision_text([{"center_fid": "c1", "legend": _LEGEND,
                                 "context_m": 20.0, "frame_up": [0.7071, 0.7071]}])
    assert "its corners lie outside the turned window and render dark" in turned
    for square in (None, [1.0, 0.0], [0.0, -1.0]):
        c = {"center_fid": "c1", "legend": _LEGEND, "context_m": 20.0}
        if square:
            c["frame_up"] = square
        assert "corners" not in look_vision_text([c]), square
