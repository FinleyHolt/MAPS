"""A class label names a color only as a whole word."""
from __future__ import annotations

from maps.query.surface import _color_of


def test_a_color_inside_another_word_is_not_the_labels_color():
    assert _color_of("water tank") == ""
    assert _color_of("covered parking") == ""
    assert _color_of("shredded") == ""
    assert _color_of("white car") == "white"
    assert _color_of("Golden dome") == ""
    assert _color_of("gold dome") == "gold"
