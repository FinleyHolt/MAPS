"""SR: arrival judged against the target's real extent.

Where an episode has an annotated shape, success is landing inside it; where
it has none, it falls back
to a disc of the dataset's radius. Synthetic geometry throughout — the
function is pure, so none of this needs a run.
"""

import pytest

from benchmarks.citynav.eval import score_precise

# A 10 m x 10 m building footprint with its centre at (5, 5).
BUILDING = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_inside_the_shape_is_success():
    out = score_precise(
        final_xy=(5.0, 5.0), target_xy=(5.0, 5.0), gt_polygon=BUILDING, radius_m=15.0
    )
    assert out["SR"] == 1
    assert out["precise_target"] == "shape"


def test_just_outside_the_shape_fails_at_zero_tolerance():
    """Strict containment is the point: 1 m from the wall is not arrival,
    even though it is 6 m from the centre and would pass any disc."""
    out = score_precise(
        final_xy=(11.0, 5.0), target_xy=(5.0, 5.0), gt_polygon=BUILDING, radius_m=15.0
    )
    assert out["SR"] == 0


def test_the_shape_beats_the_radius_when_both_are_available():
    """A generous radius must not rescue a miss on an annotated shape —
    otherwise annotating a target would make the episode EASIER."""
    out = score_precise(
        final_xy=(30.0, 5.0),
        target_xy=(5.0, 5.0),
        gt_polygon=BUILDING,
        radius_m=100.0,
    )
    assert out["SR"] == 0
    assert out["precise_target"] == "shape"


def test_falls_back_to_the_radius_with_no_shape():
    out = score_precise(
        final_xy=(5.0, 19.0), target_xy=(5.0, 5.0), gt_polygon=None, radius_m=15.0
    )
    assert out["SR"] == 1
    assert out["precise_target"] == "radius"


@pytest.mark.parametrize("y, sr", [(20.0, 1), (20.01, 0)])
def test_the_radius_fallback_is_inclusive_like_sr(y, sr):
    """SR is ``NE <= success_radius_m``; the fallback must not disagree with
    the headline metric at the boundary."""
    out = score_precise(final_xy=(5.0, y), target_xy=(5.0, 5.0), gt_polygon=None, radius_m=15.0)
    assert out["SR"] == sr


def test_degenerate_polygon_is_not_treated_as_a_shape():
    """Fewer than 3 points is not a footprint; the episode falls back to its
    radius rather than being scored against nothing."""
    out = score_precise(
        final_xy=(5.0, 10.0),
        target_xy=(5.0, 5.0),
        gt_polygon=[(0.0, 0.0), (1.0, 1.0)],
        radius_m=15.0,
    )
    assert out["precise_target"] == "radius"
    assert out["SR"] == 1


def test_zero_area_polygon_raises_rather_than_scoring_a_miss():
    """Three collinear points have no interior. Silently scoring 0 would
    blame the agent for a bad annotation, so this is loud."""
    with pytest.raises(ValueError):
        score_precise(
            final_xy=(5.0, 5.0),
            target_xy=(5.0, 5.0),
            gt_polygon=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            radius_m=15.0,
        )


def test_a_narrow_shape_is_scored_as_drawn():
    """A 4 m-wide road intersection gets no widening: 1 m past its edge misses."""
    crossing = [(0.0, 0.0), (12.0, 0.0), (12.0, 4.0), (0.0, 4.0)]
    out = score_precise(final_xy=(13.0, 2.0), target_xy=(6.0, 2.0), gt_polygon=crossing, radius_m=15.0)
    assert out["SR"] == 0
