"""The containment note is about each row it names, so it rides an answer
that sorted or cut the compared rows."""
from __future__ import annotations

from maps.search.tests.fakes import drive, run, tool_results


def _sq(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


def test_nearest_a_point_inside_a_footprint_says_the_row_contains_it():
    cands = [
        {"fid": "prior_9", "name": "Castle Ward", "xy": [15.0, 15.0],
         "mask": _sq(-30, -30, 45, 45), "mask_centroid": [7.5, 7.5],
         "class": "building", "search_text": "building"},
        {"fid": "prior_2", "name": "Two", "xy": [60.0, 10.0],
         "mask": _sq(55, 0, 65, 20), "mask_centroid": [60.0, 10.0],
         "class": "building", "search_text": "building"}]
    _r, fr, _f, _d = drive(
        [[run('w = find("building", nearest=[10.0, 10.0])[0]')], "done"],
        cands=cands)
    env = next(iter(tool_results(fr)[0]["result"]["results"].values()))
    assert "row(s) ['prior_9'] read 'distance to 10_10'=0" in env["hint"], env
