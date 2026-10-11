"""NE is the straight-line distance from the committed point to the goal; spl_pos carries the route."""

from __future__ import annotations

from benchmarks.citynav import eval as cn_eval


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
