"""The tile-schema guard refuses every retired bake shape and passes the tags-only one."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from benchmarks.citynav.paths import assert_current_tile_schema

_REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("payload, match", [
    # The typed-class bake: class keys, rows minting prior_role.
    ({"building": [{"cells": [[0, 0]], "source": "osm", "prior_role": "static_map"}],
      "footpath": [{"cells": [[0, 1]], "source": "osm", "prior_role": "static_map"}]},
     "typed class keys"),
    ({"prior": [{"cells": [[0, 0]], "source": "osm", "prior_source": "osm"}]}, "prior_role"),
    # The oldest: one prior bucket, no provenance on rows.
    ({"prior": [{"cells": [[0, 0]], "name": "Common"}]}, "source provenance"),
])
def test_a_retired_bake_shape_is_refused(payload, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        assert_current_tile_schema(payload, "tiles/retired")


def test_current_tags_verbatim_shape_passes() -> None:
    assert_current_tile_schema(
        {"prior": [
            {"cells": [[0, 0]], "source": "osm", "osm_id": "w1",
             "highway": "residential"},
            {"cells": [[1, 1]], "source": "cityrefer_landmark",
             "name": "Parker's Piece"},
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
