import numpy as np

from maps.perception.base import DetectedObject
from maps.search.tests.fakes import search_loop
from maps.sws.graph.model import ObjectGraph


def test_the_step_record_carries_no_tile_cells():
    loop = search_loop()
    loop.start(ObjectGraph(nodes={}, cell_size_m=2.0))
    cells = np.stack([np.arange(2000), np.arange(2000)], 1).astype(np.int32)
    det = DetectedObject(world_xy=(3.0, 3.0), label="car", confidence=0.9, source_tier="sam3_1_image",
                         metadata={"tile_cells": cells, "sam3_1_score": 0.9})
    [rec] = loop.update_perception([det], {})
    assert "tile_cells" not in rec["metadata"]
    assert rec["metadata"]["sam3_1_score"] == 0.9
