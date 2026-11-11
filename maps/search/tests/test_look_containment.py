"""A crop subject is in its window by centroid or by footprint."""

from __future__ import annotations

import pytest

from maps.render.look_render import subject_window_containment

# A recorded crop window and a long strip footprint whose centroid lies west
# of it while its eastern end is in frame.
_W = (400.0, 185.51, 418.0, 203.51)
_STRIP = [(353.38, 209.97), (350.58, 198.71), (424.74, 180.25), (427.54, 191.51)]


@pytest.mark.parametrize("window, xy, boundary, expected", [
    (_W, (389.06, 195.11), _STRIP, (True, False)),
    (_W, (389.06, 195.11), None, (False, False)),
    ((0.0, 0.0, 18.0, 18.0), (9.0, 9.0), None, (True, True)),
    (_W, (435.0, 215.0), [(430.0, 210.0), (440.0, 210.0), (440.0, 220.0), (430.0, 220.0)],
     (False, False)),
    # crosses the window edge with no vertex inside: the clip, not a vertex test
    (_W, (392.0, 195.0), [(395.0, 194.0), (425.0, 194.0), (425.0, 196.0), (395.0, 196.0)],
     (True, False)),
])
def test_containment_is_centroid_or_footprint(window, xy, boundary, expected):
    assert subject_window_containment(window, xy, boundary) == expected
