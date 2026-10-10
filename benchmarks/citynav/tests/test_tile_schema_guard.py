"""The tile-schema guard refuses every retired bake shape, not just the oldest.

A wrong bake completes the run and only moves SR, so the guard is the sole
thing standing between a stale ``SEARCH_CITYNAV_TILES`` and a number nobody
can compare. The current bake is tags-only (PRIOR_CONTRACT config B): ONE
``prior`` list whose rows are mask + raw tags verbatim + ``source``
provenance. Retired shapes the guard must refuse: the typed-class bake
(``building``/``footpath``/... keys, rows minting ``prior_role``/
``prior_source``) and the older opaque ``prior`` bucket whose rows carry no
``source``. The guard itself once pinned the typed shape as current — it was
written from the on-disk bake while the bake code had already moved to
tags-only, so the one loadable bake was the retired one.

The last test is the vacuity floor. The hand-built payloads would keep
passing if the guard grew a bug that refuses everything; the floor asserts
the real bake on disk still passes, block by block.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)

from benchmarks.citynav.paths import assert_current_tile_schema  # noqa: E402


def test_typed_class_bake_is_refused() -> None:
    """The pre-pivot shape: typed class keys, rows minting prior_role."""
    with pytest.raises(RuntimeError, match="typed class keys"):
        assert_current_tile_schema(
            {
                "building": [{"cells": [[0, 0]], "source": "osm",
                              "prior_role": "static_map"}],
                "footpath": [{"cells": [[0, 1]], "source": "osm",
                              "prior_role": "static_map"}],
            },
            "tiles/citynav_2m_osmv2",
        )


def test_prior_rows_minting_role_are_refused() -> None:
    with pytest.raises(RuntimeError, match="prior_role"):
        assert_current_tile_schema(
            {"prior": [{"cells": [[0, 0]], "source": "osm",
                        "prior_source": "osm"}]},
            "tiles/citynav_2m_osmv2_flat",
        )


def test_opaque_sourceless_prior_bucket_is_refused() -> None:
    """The oldest retired shape: one prior bucket, no provenance on rows."""
    with pytest.raises(RuntimeError, match="source provenance"):
        assert_current_tile_schema(
            {"prior": [{"cells": [[0, 0]], "name": "Common"}]},
            "tiles/citynav_2m",
        )


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
    bake = os.path.join(_REPO, "tiles", "citynav_2m_osmv3")
    blocks = sorted(glob.glob(os.path.join(bake, "*_features.json")))
    if not blocks:
        pytest.skip(f"canonical bake not staged at {bake}")
    # Vacuity floor: a guard that refused everything would still pass the
    # hand-built cases above by raising. It cannot pass this one.
    assert len(blocks) >= 30, f"only {len(blocks)} blocks -- is the bake partial?"
    for path in blocks:
        with open(path) as fh:
            assert_current_tile_schema(json.load(fh), path)
