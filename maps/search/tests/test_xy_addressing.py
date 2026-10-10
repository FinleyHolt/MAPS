"""Movement addressing: an ``[x, y]`` world point is accepted where the
drone FLIES — ``viewpoints.add[].target_fid``, crop centers and marks. The
agent points a stop at a derived crossing/corner no row sits on.

One grammar (``maps.util.xy_token``) feeds every accept site. The SELECT
takes no world point — the commit binds the selected fid's centroid (see
``test_served_spelling_verbatim`` for the accept contract and
``test_commit_agent_pick_owns`` for the walk).
"""
from __future__ import annotations

from maps.util.xy_token import parse_xy_token


# --------------------------------------------------------------------------- #
# the ONE token grammar                                                        #
# --------------------------------------------------------------------------- #


def test_xy_token_grammar():
    assert parse_xy_token("[1530, 1378]") == (1530.0, 1378.0)
    assert parse_xy_token("[1530.5,-378.25]") == (1530.5, -378.25)
    assert parse_xy_token(" 12, 34 ") == (12.0, 34.0)  # bare pair tolerated
    for bad in ("prior_12", "detected_5", "r3", "[1530]", "[a, b]",
                "[1, 2, 3]", "", None, 12):
        assert parse_xy_token(bad) is None, bad


# --------------------------------------------------------------------------- #
# viewpoints: an "[x, y]" target flies a stop AT that coordinate               #
# --------------------------------------------------------------------------- #


class _Graph:
    def __init__(self, nodes):
        self.nodes = dict(nodes)


def test_xy_viewpoint_target_synthesizes_a_stop_at_the_point():
    import numpy as np

    from maps.search.parse_compile.tour_synth import (
        TourSynthesizerConfig,
        synthesize_extension,
    )

    class _TNode:
        def __init__(self, centroid):
            self.centroid = centroid
            self.provenance = "prebaked"

    height_map = np.zeros((10, 10), dtype=float)
    bounds = (0.0, 0.0, 10.0, 10.0)
    graph = _Graph({"car_1": _TNode((2, 2))})
    stops = synthesize_extension(
        existing_tour=[],
        viewpoints_to_add=[{"target_fid": "[6.0, 7.0]"}],
        runtime_graph=graph,
        bounds=bounds,
        cell_size_m=1.0,
        cfg=TourSynthesizerConfig(),
        height_map=height_map,
    )
    # the point target resolved a pose OVER the coordinate — no grounded
    # node exists for it, and it was NOT silently dropped.
    assert len(stops) == 1, stops
    obs = stops[0].observations[0]
    assert obs.target_fid == "[6.0, 7.0]"
    assert obs.target_pose == (6.0, 7.0)
    # the picker may grid-snap; the pose is OVER the point, not elsewhere
    assert abs(stops[0].pose[0] - 6.0) <= 2.0, stops[0].pose
    assert abs(stops[0].pose[1] - 7.0) <= 2.0, stops[0].pose
