"""One verdict per row, under the bench's own success mask.

``eval.score_arrival`` is the single entry point every writer of a scored
row goes through — the main merge and the zero-flight branch. What it
returns IS the row's metrics, so the two cannot drift onto different
criteria.
"""

from __future__ import annotations

import pytest

from benchmarks.citynav import eval as cn_eval

#: A square target, 20 m on a side, well away from the origin.
SHAPE = [(500.0, 500.0), (520.0, 500.0), (520.0, 520.0), (500.0, 520.0)]
CENTRE = (510.0, 510.0)

#: 18 m from the centroid and 8 m outside the footprint: INSIDE the 20 m
#: ring, OUTSIDE the shape. The two criteria disagree here by construction.
RING_YES_SHAPE_NO = (528.0, 510.0)

#: On the footprint's edge and 10 m from the centroid: both say yes.
BOTH_YES = (520.0, 510.0)


def _score(criterion: str, xy, **kw):
    return cn_eval.score_arrival(
        criterion=criterion,
        final_pos_xy=xy,
        pred_xy=xy,
        target_xy=CENTRE,
        pos_trajectory=[(0.0, 0.0), xy],
        pred_trajectory=[xy],
        success_dist_m=20.0,
        gt_polygon=SHAPE,
        radius_m=20.0,
        **kw,
    )


def test_the_two_criteria_can_disagree():
    """THE VACUITY FLOOR for every test below. If no point separated the
    ring from the shape, "the criterion is honoured" would be unfalsifiable
    -- both branches would return the same verdict on every input.
    """
    assert _score("ring", RING_YES_SHAPE_NO)["SR"] == 1
    assert _score("shape", RING_YES_SHAPE_NO)["SR"] == 0
    # ...and they agree where they should, so the disagreement above is the
    # geometry and not a broken shape branch.
    assert _score("ring", BOTH_YES)["SR"] == 1
    assert _score("shape", BOTH_YES)["SR"] == 1


def test_sr_is_the_verdict_and_the_row_says_under_which_mask():
    for criterion in ("ring", "shape"):
        row = _score(criterion, BOTH_YES)
        assert row["SR"] in (0, 1)
        assert row["success_criterion"] == criterion
        # There is no second success key to quote by mistake.
        assert "SR_precise" not in row


def test_a_shape_row_carries_no_ring_key():
    """Ring keys answer "within R of the target POINT". On a bench scored
    against the footprint that is a different question, and a reader that
    found one there would be reading a number nobody scored on."""
    row = _score("shape", BOTH_YES)
    for key in cn_eval._RING_ONLY:
        assert key not in row, key
    # The criterion-neutral distances stay: they carry no radius.
    for key in ("NE", "ne_pos_to_goal", "ne_pred_to_goal",
                "min_dist_pos_to_target_m", "scoring_basis"):
        assert key in row, key
    # ...as does the mask's own detail.
    assert row["precise_target"] == "shape"
    assert row["precise_distance_m"] is not None


def test_a_ring_row_carries_no_shape_key():
    """The mirror floor: a build that ran score_precise on every bench would
    pass the test above and put a second arrival number on every CityNav
    row."""
    row = _score("ring", BOTH_YES)
    for key in ("precise_target", "precise_distance_m", "precise_reach_m",
                "precise_floor_applied"):
        assert key not in row, key
    assert row["OSR"] in (0, 1)
    assert row["spl_pos"] is not None


def test_an_unknown_criterion_refuses():
    with pytest.raises(ValueError):
        _score("disc", BOTH_YES)


def test_a_never_flown_episode_scores_zero_under_either_mask():
    """The honest zero. ``flew=False`` zeroes EVERY success flag while the
    distances stay true -- and the scorer owns that list, so a caller cannot
    zero four of them and forget the other two.

    Scored at a point that would otherwise WIN under both criteria, which is
    what makes this test able to fail.
    """
    for criterion in ("ring", "shape"):
        won = _score(criterion, BOTH_YES)
        assert won["SR"] == 1, "the fixture must win when it flies"
        row = _score(criterion, BOTH_YES, flew=False)
        for key in cn_eval._SUCCESS_FLAGS:
            if key in row:
                assert row[key] == 0, f"{criterion}: {key}"
        # The honest part: NE is still start->target, not zeroed.
        assert row["NE"] == won["NE"]
