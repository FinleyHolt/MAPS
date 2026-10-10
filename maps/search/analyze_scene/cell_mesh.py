"""The bake's own 2 m cell mesh: what a ROW is made of.

Every graph node persists ``cells`` — the (row, col) grid cells its footprint
occupies. ``sws.graph.io`` derives the centerline from them on load and
``build_surface`` derives the mask loops from them, so the cells are the
primary record and the reductions are views of it.

This module answers ONE question: which cells does a row occupy? A row is one
OSM way, and a named road is routinely several of them (75 rows carry
"California City Boulevard" on the calcity bake) — the road is the SET those
rows form, which ``find`` names with ``.exact`` and ``explore`` covers in one
pass. Nothing here folds them: a fid addresses one way, in this mesh and in
the runtime graph alike, so ``node_coverage_stops`` walks the same ground the
row describes.

SPATIAL MATH DOES NOT LIVE HERE. Distance and intersection run on the EXACT
geometry each node was given (``geom_compute.gap_witness`` and
``polyline_crossings`` over the ``world_contour`` chains), not on the cell
raster: every node persists its own vertices now — a ring, a way, or a single
point — and measuring them at 2 m throws that away. A cell-set intersection
was measured against the crossings op over all 397 road pairs the model asked
for and agreed on only 85.0% of sentences, missing crossings whose cells fall
a cell short and splitting single crossings whose shared band fragments. The
cells are what a feature COVERS; the chain is what it IS.
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


