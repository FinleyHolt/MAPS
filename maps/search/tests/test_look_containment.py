from __future__ import annotations

from maps.search.render.look_render import subject_window_containment

# The 8267 recorded crop geometry, verbatim from the flight/look manifest.
_WINDOW_8267 = (400.0, 185.51, 418.0, 203.51)
_CENTROID_8267 = (389.06, 195.11)
_GT_8267 = (411.0625, 189.640625)

# Corners centroid +- 38.2*u +- 5.8*perp.
_STRIP_8267 = [
    (353.38, 209.97),
    (350.58, 198.71),
    (424.74, 180.25),
    (427.54, 191.51),
]


def test_8267_subject_in_window_true_via_footprint():
    in_win, centroid_in = subject_window_containment(
        _WINDOW_8267, _CENTROID_8267, _STRIP_8267)
    assert in_win is True          # the footprint's eastern end is in frame
    assert centroid_in is False    # the centroid genuinely is not
    wx0, wy0, wx1, wy1 = _WINDOW_8267
    assert wx0 <= _GT_8267[0] <= wx1 and wy0 <= _GT_8267[1] <= wy1


def test_8267_centroid_point_alone_reproduces_the_old_false_negative():
    in_win, centroid_in = subject_window_containment(
        _WINDOW_8267, _CENTROID_8267, None)
    assert in_win is False and centroid_in is False


def test_nadir_centered_subject_never_warns():
    # A fid at the window center (the nadir hover) must always be contained.
    in_win, centroid_in = subject_window_containment(
        (0.0, 0.0, 18.0, 18.0), (9.0, 9.0), None)
    assert in_win is True and centroid_in is True


def test_fully_outside_footprint_still_warns():
    # A footprint that never enters the window keeps the truthful warning.
    poly = [(430.0, 210.0), (440.0, 210.0), (440.0, 220.0), (430.0, 220.0)]
    in_win, centroid_in = subject_window_containment(
        _WINDOW_8267, (435.0, 215.0), poly)
    assert in_win is False and centroid_in is False


def test_footprint_straddling_one_edge_counts_as_in_window():
    # A polygon crossing the window edge with no vertex inside still clips
    # to positive area (the Sutherland-Hodgman path, not a vertex test).
    poly = [(395.0, 194.0), (425.0, 194.0), (425.0, 196.0), (395.0, 196.0)]
    in_win, _ = subject_window_containment(
        _WINDOW_8267, (392.0, 195.0), poly)
    assert in_win is True
