"""The cell mesh answers which cells a ROW occupies, and nothing else."""

import numpy as np

from maps.search.analyze_scene import cell_mesh as M


def test_cells_read_off_a_dict_row_and_off_a_node():
    class _Node:
        cells = np.array([[1, 2], [1, 3]], dtype=int)
    assert len(M.cells_of({"cells": [[0, 0], [0, 1]]})) == 2
    assert len(M.cells_of(_Node())) == 2
    assert len(M.cells_of({})) == 0
    assert len(M.cells_of({"cells": None})) == 0


