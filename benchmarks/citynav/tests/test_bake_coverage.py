"""The bake keeps a prior only where imagery exists, not where the bbox does.

The block bounds rectangle over-admits: the PLY footprint is tighter than the
nominal bounds (a third of a block rectangle can be no-data), and a prior over
no-data becomes a graph node the agent can request a viewpoint on: the stop
renders an all-black crop with zero detections and returns silently. The bake
clips every record's cells to the imagery the block ortho carries.

Orientation is load-bearing: ortho rows run north-down while bake cell rows run
south-up, so an unflipped reduce passes every shape check and silently keeps
the wrong half of every block. The last test is the vacuity floor on the real
bake: a coverage mask that silently degraded to all-True would drop nothing
anywhere, which the real no-data fractions make impossible.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

import benchmarks.citynav.paths as citynav_paths
from benchmarks.citynav.bake.city import bake_block
from maps.search.tests.ortho_fixture import write_rasters

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BLOCK = "toy_block_0"


def _poly(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


RECORDS = [
    ({"landuse": "grass", "osm_id": "w_all"}, _poly(0.1, 0.1, 7.9, 7.9)),
    ({"building": "yes", "osm_id": "w_south"}, _poly(0.1, 0.1, 7.9, 3.9)),
    ({"building": "yes", "osm_id": "w_straddle", "name": "Straddle House"}, _poly(0.1, 0.1, 3.9, 5.9)),
]


def _bake(tmp_path, monkeypatch, *, imagery_north: bool) -> tuple[dict, dict]:
    """An 8 m x 8 m block of 2 m cells; ``imagery_north`` writes a raster whose
    north half alone has imagery, else no raster exists."""
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / f"{BLOCK}_meta.json").write_text(json.dumps({
        "cell_size_m": 2.0, "grid_shape": [4, 4],
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 8.0, "y_max": 8.0},
    }))
    if imagery_north:
        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        rgb[:40, :] = 90  # raster row 0 is y_max: the NORTH half has imagery
        write_rasters(tmp_path / "rasters", BLOCK, rgb, np.zeros((80, 80), dtype=np.float32))
    monkeypatch.setattr(citynav_paths, "CITYNAV_NATIVE_RGBD_DIR", tmp_path / "rasters")
    bake_block(BLOCK, RECORDS, tiles)
    rows = json.loads((tiles / f"{BLOCK}_features.json").read_text())["prior"]
    return {r["osm_id"]: r for r in rows}, json.loads((tiles / f"{BLOCK}_meta.json").read_text())


def test_records_clip_to_the_imagery(tmp_path, monkeypatch) -> None:
    """North imagery covers the HIGH cell rows; wholly-south records drop and
    straddlers keep only their on-imagery cells, centroid included."""
    rows, meta = _bake(tmp_path, monkeypatch, imagery_north=True)
    assert meta["coverage_source"] == "ortho alpha"
    assert sorted(rows) == ["w_all", "w_straddle"]
    assert rows["w_all"]["cells"] == [[r, c] for r in (2, 3) for c in range(4)]
    assert rows["w_straddle"]["cells"] == [[2, 0], [2, 1]]
    assert rows["w_straddle"]["centroid_row"] == 2.0
    # The straddler's true bbox center (2.0, 3.0) is over no-data: flagged.
    assert rows["w_straddle"]["off_raster"] is True and rows["w_straddle"]["off_raster_m"] > 0
    assert meta["bake_counts"]["raster_coverage"]["off_coverage_dropped"] == 1


def test_no_raster_keeps_every_rasterized_cell(tmp_path, monkeypatch) -> None:
    rows, meta = _bake(tmp_path, monkeypatch, imagery_north=False)
    assert meta["coverage_source"] == "bounds rectangle"
    assert sorted(rows) == ["w_all", "w_south", "w_straddle"]
    assert len(rows["w_all"]["cells"]) == 16


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
