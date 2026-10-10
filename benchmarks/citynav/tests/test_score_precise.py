"""SR: arrival judged against the target's real extent.

Where an episode has an annotated shape, success is landing inside it (or
within a configured tolerance of its edge); where it has none, it falls back
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
    assert out["precise_distance_m"] == 0.0


def test_just_outside_the_shape_fails_at_zero_tolerance():
    """Strict containment is the point: 1 m from the wall is not arrival,
    even though it is 6 m from the centre and would pass any disc."""
    out = score_precise(
        final_xy=(11.0, 5.0), target_xy=(5.0, 5.0), gt_polygon=BUILDING, radius_m=15.0
    )
    assert out["SR"] == 0
    assert out["precise_distance_m"] == 1.0


def test_tolerance_admits_a_margin_around_the_edge():
    out = score_precise(
        final_xy=(11.0, 5.0),
        target_xy=(5.0, 5.0),
        gt_polygon=BUILDING,
        radius_m=15.0,
        tolerance_m=2.0,
    )
    assert out["SR"] == 1


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
    assert out["precise_distance_m"] == 14.0


def test_radius_boundary_is_inclusive_like_sr():
    """SR is ``NE <= success_radius_m``; the fallback must not disagree with
    the headline metric at the boundary."""
    out = score_precise(
        final_xy=(5.0, 20.0), target_xy=(5.0, 5.0), gt_polygon=None, radius_m=15.0
    )
    assert out["SR"] == 1


def test_radius_just_outside_fails():
    out = score_precise(
        final_xy=(5.0, 20.01), target_xy=(5.0, 5.0), gt_polygon=None, radius_m=15.0
    )
    assert out["SR"] == 0


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


def test_floor_widens_a_narrow_shape_instead_of_replacing_it():
    """A 4 m-wide road intersection is tighter than the agent can aim. The
    floor buys it 3 m of reach (5 m floor less its 2 m inradius) without
    moving the target: 3 m past the edge passes, 4 m still misses."""
    crossing = [(0.0, 0.0), (12.0, 0.0), (12.0, 4.0), (0.0, 4.0)]
    near = score_precise(
        final_xy=(15.0, 2.0),
        target_xy=(6.0, 2.0),
        gt_polygon=crossing,
        radius_m=15.0,
        floor_radius_m=5.0,
    )
    assert near["SR"] == 1
    assert near["precise_target"] == "shape"  # still scored on the footprint
    assert near["precise_floor_applied"] is True
    assert near["precise_reach_m"] == pytest.approx(3.0, abs=0.01)

    far = score_precise(
        final_xy=(16.0, 2.0),
        target_xy=(6.0, 2.0),
        gt_polygon=crossing,
        radius_m=15.0,
        floor_radius_m=5.0,
    )
    assert far["SR"] == 0


def test_floor_measures_width_not_area():
    """A long thin sliver has plenty of area and is still unaimable. Judging
    the floor by equivalent radius would let this one through; judging it by
    the largest circle that fits does not."""
    sliver = [(0.0, 0.0), (100.0, 0.0), (100.0, 3.0), (0.0, 3.0)]
    out = score_precise(
        final_xy=(50.0, 1.5),
        target_xy=(50.0, 1.5),
        gt_polygon=sliver,
        radius_m=15.0,
        floor_radius_m=5.0,
    )
    assert out["precise_floor_applied"] is True
    assert out["precise_reach_m"] == pytest.approx(3.5, abs=0.01)


def test_floor_leaves_a_shape_wider_than_it_alone():
    """The floor is a floor, not a buffer: a building that already holds a
    5 m circle scores by strict containment exactly as before."""
    out = score_precise(
        final_xy=(11.0, 5.0),
        target_xy=(5.0, 5.0),
        gt_polygon=BUILDING,
        radius_m=15.0,
        floor_radius_m=5.0,
    )
    assert out["SR"] == 0
    assert out["precise_floor_applied"] is False
    assert out["precise_reach_m"] == 0.0


def test_floor_never_tightens_an_explicit_tolerance():
    """Tolerance and floor are both slack; the wider one wins rather than
    the later one overwriting the earlier."""
    out = score_precise(
        final_xy=(5.0, 5.0),
        target_xy=(5.0, 5.0),
        gt_polygon=BUILDING,
        radius_m=15.0,
        tolerance_m=8.0,
        floor_radius_m=5.0,
    )
    assert out["precise_reach_m"] == 8.0
    assert out["precise_floor_applied"] is False


def test_floor_does_not_touch_the_radius_fallback():
    """An episode with no shape is already scored by a disc; the floor has
    nothing to widen and must not shrink it either."""
    out = score_precise(
        final_xy=(5.0, 17.0),
        target_xy=(5.0, 5.0),
        gt_polygon=None,
        radius_m=15.0,
        floor_radius_m=5.0,
    )
    assert out["SR"] == 1
    assert out["precise_reach_m"] == 15.0
    assert out["precise_floor_applied"] is False


def test_across_the_street_is_a_miss():
    """SR asks only where the agent finished: the correct building
    identified from across the street is still a miss."""
    across_the_street = score_precise(
        final_xy=(-8.0, 5.0), target_xy=(5.0, 5.0), gt_polygon=BUILDING, radius_m=20.0
    )
    assert across_the_street["SR"] == 0
    assert across_the_street["precise_distance_m"] == 8.0
