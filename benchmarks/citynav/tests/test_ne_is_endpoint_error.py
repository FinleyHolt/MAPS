"""NE measures the endpoint, and every label that prints it says so.

Readers assume "navigation error" means deviation from an optimal path.
It is the straight-line distance from the committed point to the goal,
and ``spl_pos`` is the metric carrying the route term. A run whose
summary says only ``mean_NE: 144.30 m`` invites the wrong reading of
every number under it.

Pinned against ``score_goal_prediction_citynav_native`` -- the scorer the
runner actually calls. The legacy ``score_goal_prediction`` these checks
used to run against had no caller left once the round-era commit-only
resume stopped naming it, so a green test there proved nothing about a
flown row.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from benchmarks.citynav import bench_fullset, eval as cn_eval

_REPO = Path(__file__).resolve().parents[3]


def test_ne_ignores_the_path_taken():
    """Same endpoint, wildly different routes -> identical NE."""
    direct = cn_eval.score_goal_prediction_citynav_native(
        final_pos_xy=(101.0, 100.0),
        pred_xy=(101.0, 100.0),
        target_xy=(100.0, 100.0),
        pos_trajectory=[(0.0, 0.0), (101.0, 100.0)],
    )
    scenic = cn_eval.score_goal_prediction_citynav_native(
        final_pos_xy=(101.0, 100.0),
        pred_xy=(101.0, 100.0),
        target_xy=(100.0, 100.0),
        pos_trajectory=[(0.0, 0.0), (0.0, 3000.0), (3000.0, 3000.0), (101.0, 100.0)],
    )
    assert direct["NE"] == scenic["NE"] == 1.0, "NE is endpoint-only"
    assert direct["SR"] == scenic["SR"] == 1, "both commit inside 20 m"
    # The 3 km detour is invisible to NE and visible to spl_pos. Both halves
    # are asserted: without the second, a build that stopped pricing the
    # route at all would still pass the first.
    assert scenic["spl_pos"] < direct["spl_pos"], (
        "spl_pos is the route term -- the detour must cost something there"
    )


def test_ne_is_distance_to_goal_not_to_start():
    scored = cn_eval.score_goal_prediction_citynav_native(
        final_pos_xy=(0.0, 0.0),
        pred_xy=(0.0, 0.0),
        target_xy=(30.0, 40.0),
    )
    assert scored["NE"] == 50.0, "3-4-5 from the commit to the goal"
    assert scored["SR"] == 0, "50 m is outside the 20 m radius"


def _says_what_ne_measures(text: str) -> bool:
    low = text.lower()
    return "straight" in low and "goal" in low


def test_every_surface_that_prints_ne_says_what_it_measures():
    """Vacuity floor: these three surfaces exist and each is qualified.

    A grep that finds nothing passes, so the count is asserted too. If a
    surface is renamed, fix the name here rather than dropping the check.
    """
    surfaces = {
        "eval.score_goal_prediction_citynav_native docstring": inspect.getdoc(
            cn_eval.score_goal_prediction_citynav_native
        ),
        "eval module docstring": inspect.getdoc(cn_eval),
        "bench_fullset summary line": inspect.getsource(bench_fullset.__dict__["Driver"])
        if "Driver" in bench_fullset.__dict__
        else inspect.getsource(bench_fullset),
    }
    assert len(surfaces) == 3, "the scan must cover every NE surface"
    for name, text in surfaces.items():
        assert text, f"{name} has no text to check"
        assert "NE" in text, f"{name} no longer mentions NE -- retarget this check"
        assert _says_what_ne_measures(text), (
            f"{name} prints NE without saying it is the straight-line "
            f"distance to the goal"
        )
