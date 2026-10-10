"""The bake keeps a prior only where imagery exists, not where the bbox does.

The block bounds rectangle over-admits: the PLY footprint is tighter than the
nominal bounds (a third of a block rectangle can be no-data), and a prior over
no-data becomes a graph node the agent can request a viewpoint on — the stop
renders an all-black crop with zero detections and returns silently. The bake
clips every record's cells to the imagery coverage mask
(``_cell_coverage`` -> ``_features_from_records(coverage=...)``).

The orientation test is the load-bearing one: ortho rows run north-down while
bake cell rows run south-up, so an unflipped reduce passes every shape check
and silently keeps the wrong half of every block. The last test is the
vacuity floor on the real bake: a coverage mask that silently degraded to
all-True would drop nothing anywhere, which the real no-data fractions make
impossible.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)

import benchmarks.citynav.paths as citynav_paths  # noqa: E402
from benchmarks.citynav.bake.city import (  # noqa: E402
    _cell_coverage,
    _features_from_records,
)
from benchmarks.citynav.eval import WorldBounds  # noqa: E402
from maps.search.tests.ortho_fixture import write_rasters  # noqa: E402


def _poly(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


BOUNDS = WorldBounds(x_min=0.0, y_min=0.0, x_max=8.0, y_max=8.0)
GRID = (4, 4)
CELL = 2.0


def test_cell_coverage_orientation(tmp_path, monkeypatch) -> None:
    """North-half imagery covers the HIGH cell rows (south-up grid)."""
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:40, :] = 90  # raster row 0 is y_max: the NORTH half has imagery
    write_rasters(tmp_path, "toy_block_0", rgb, np.zeros((80, 80), dtype=np.float32))
    monkeypatch.setattr(citynav_paths, "CITYNAV_NATIVE_RGBD_DIR", tmp_path)
    meta: dict = {}
    cov = _cell_coverage("toy_block_0", meta, BOUNDS, GRID, CELL)
    assert cov is not None and meta["coverage_source"] == "ortho alpha"
    assert cov[2:, :].all(), "north imagery must cover the high cell rows"
    assert not cov[:2, :].any(), "south no-data must stay uncovered"


def test_cell_coverage_missing_ortho_keeps_rectangle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(citynav_paths, "CITYNAV_NATIVE_RGBD_DIR", tmp_path)
    meta: dict = {}
    assert _cell_coverage("cambridge", meta, BOUNDS, GRID, CELL) is None
    assert meta["coverage_source"] == "bounds rectangle"


def test_records_clip_to_coverage() -> None:
    """Fully-off records drop; straddlers keep only on-imagery cells."""
    coverage = np.zeros(GRID, dtype=bool)
    coverage[:, 0] = True  # only the western cell column has imagery
    records = [
        ({"building": "yes", "osm_id": "w_off"}, _poly(4.1, 0.1, 7.9, 7.9)),
        ({"building": "yes", "osm_id": "w_straddle", "name": "Straddle House"},
         _poly(0.1, 0.1, 7.9, 3.9)),
    ]
    features = _features_from_records(
        records, bounds=BOUNDS, grid_shape=GRID, cell_size_m=CELL,
        line_buffer_m=1.0, coverage=coverage,
    )
    rows = features.get("prior", [])
    assert [r["osm_id"] for r in rows] == ["w_straddle"]
    assert all(coverage[r, c] for r, c in rows[0]["cells"]), rows[0]["cells"]
    rr = int(round(rows[0]["centroid_row"]))
    cc = int(round(rows[0]["centroid_col"]))
    assert coverage[rr, cc], "centroid must sit over imagery"
    # The straddler's true bbox center (4.0, 2.0) is over no-data: flagged.
    assert rows[0].get("off_raster") is True and rows[0]["off_raster_m"] > 0


def test_no_coverage_keeps_every_rasterized_cell() -> None:
    records = [({"building": "yes", "osm_id": "w1"}, _poly(4.1, 0.1, 7.9, 7.9))]
    features = _features_from_records(
        records, bounds=BOUNDS, grid_shape=GRID, cell_size_m=CELL,
        line_buffer_m=1.0, coverage=None,
    )
    assert [r["osm_id"] for r in features["prior"]] == ["w1"]


def test_real_bake_dropped_something() -> None:
    """Vacuity floor: on the real bake, coverage ran and actually bit."""
    bake = os.path.join(_REPO, "tiles", "citynav_2m_osmv3")
    metas = sorted(glob.glob(os.path.join(bake, "*_block_*_meta.json")))
    if not metas:
        pytest.skip(f"canonical bake not staged at {bake}")
    assert len(metas) >= 30, f"only {len(metas)} block metas -- partial bake?"
    dropped = 0
    for path in metas:
        with open(path) as fh:
            meta = json.load(fh)
        assert meta.get("coverage_source") == "ortho alpha", path
        cov = (meta.get("bake_counts") or {}).get("raster_coverage") or {}
        dropped += int(cov.get("off_coverage_dropped") or 0)
    assert dropped > 0, "no block dropped a single off-imagery record"
