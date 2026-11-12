"""A prior with no cells projects to no mask, and the rest of the stop survives."""

from __future__ import annotations

from maps.perception.geometry.geolocate import TileFrame
from maps.perception.ortho.prior_overlay import build_prior_polys_ortho

_IMG = 256
_BOUNDS = TileFrame(x_min=0.0, y_min=0.0, cell_size_m=2.0, grid_shape=(512, 512))


def test_a_prior_with_no_cells_is_skipped_and_the_rest_of_the_stop_survives():
    """An empty footprint projects to no mask, the same as one wholly outside
    the view, so the builder skips that prior and keeps the others."""
    features = {
        "building": [
            {"centroid_row": 62, "centroid_col": 74, "cells": []},
            {
                "centroid_row": 62,
                "centroid_col": 74,
                "cells": [[r, c] for r in range(60, 66) for c in range(70, 78)],
            },
        ]
    }
    corners = [(100.0, 340.0), (420.0, 340.0), (420.0, 100.0), (100.0, 100.0)]
    polys = build_prior_polys_ortho(
        rgb_shape=(_IMG, _IMG),
        features=features,
        bounds=_BOUNDS,
        view_corners_xy=corners,
        fids_to_render=["building_0", "building_1"],
    )
    assert [fid for fid, _ in polys] == ["building_1"]
    assert polys[0][1].crop.any()
