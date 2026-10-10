"""The bake's own 2 m cell mesh: what a ROW is made of.

Every graph node persists ``cells`` — the (row, col) grid cells its footprint
occupies. ``sws.graph.io`` derives the centerline from them on load and
``build_surface`` derives the mask loops from them, so the cells are the
primary record and the reductions are views of it.

This module answers ONE question: which cells does a row occupy? A row is one
OSM way; a named road is the SET of rows carrying its name. A fid addresses
one way, in this mesh and in the runtime graph alike.

SPATIAL MATH DOES NOT LIVE HERE. Distance and intersection run on the EXACT
geometry each node was given (``geom_compute.gap_witness`` and
``polyline_crossings`` over the ``world_contour`` chains), not on the cell
raster. The cells are what a feature COVERS; the chain is what it IS.
"""

from __future__ import annotations

import numpy as np


def cells_of(row) -> np.ndarray:
    """A surface row's or graph node's cells as an ``(n, 2)`` int array.

    Accepts either shape the tree carries: a dict row (``row["cells"]``) or a
    ``MapObject`` (``node.cells``). Empty when the node carries none.
    """
    c = row.get("cells") if isinstance(row, dict) else getattr(row, "cells", None)
    if c is None:
        return np.zeros((0, 2), dtype=int)
    a = np.asarray(c).reshape(-1, 2)
    return a.astype(int, copy=False)


