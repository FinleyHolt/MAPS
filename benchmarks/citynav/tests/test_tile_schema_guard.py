"""The tile-schema guard refuses another bake shape and passes the current one."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from benchmarks.citynav.paths import assert_current_tile_schema

_REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("payload", [
    {"building": [{"cells": [[0, 0]]}], "footpath": [{"cells": [[0, 1]]}]},
    {"prior": [["cells", [0, 0]]]},
])
def test_another_bake_shape_is_refused(payload) -> None:
    with pytest.raises(RuntimeError, match="not the current tile bake"):
        assert_current_tile_schema(payload, "tiles/other")


def test_current_tags_verbatim_shape_passes() -> None:
    assert_current_tile_schema(
        {"prior": [
            {"cells": [[0, 0]], "osm_id": "w1", "highway": "residential"},
            {"cells": [[1, 1]], "name": "Parker's Piece"},
        ]},
        "tiles/citynav_2m_osmv3",
    )


def test_empty_payload_passes() -> None:
    """A block with no priors is a valid (if barren) current bake."""
    assert_current_tile_schema({}, "tiles/citynav_2m_osmv3")


def test_every_block_of_the_real_bake_passes() -> None:
    bake = str(_REPO / "tiles" / "citynav_2m_osmv3")
    blocks = sorted(glob.glob(os.path.join(bake, "*_features.json")))
    if not blocks:
        pytest.skip(f"canonical bake not staged at {bake}")
    # Vacuity floor: a guard that refused everything would still pass the
    # hand-built cases above by raising. It cannot pass this one.
    assert len(blocks) >= 30, f"only {len(blocks)} blocks -- is the bake partial?"
    for path in blocks:
        with open(path) as fh:
            assert_current_tile_schema(json.load(fh), path)
