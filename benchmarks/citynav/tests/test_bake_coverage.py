"""The bake keeps a prior only where imagery exists, not where the bbox does.

The block bounds rectangle over-admits: the PLY footprint is tighter than the
nominal bounds (a third of a block rectangle can be no-data), and a prior over
no-data becomes a graph node the agent can request a viewpoint on: the stop
renders an all-black crop with zero detections and returns silently. The bake
clips every record's cells to the imagery the block ortho carries.

Orientation is load-bearing: ortho rows run north-down while bake cell rows run
south-up, so an unflipped reduce passes every shape check and silently keeps
the wrong half of every block.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import maps.perception.ortho.camera_model as camera_model
from benchmarks.citynav.bake.city import BakeError, _cell_coverage, bake_block
from benchmarks.citynav.eval import WorldBounds
from maps.mission.tests.ortho_fixture import write_rasters

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


def _bake(tmp_path, monkeypatch, *, imagery: bool) -> dict:
    """An 8 m x 8 m block of 2 m cells. ``imagery``: a raster whose north half
    alone has imagery; else a block with no CityNav imagery baked over its
    bounds (the bounds rectangle is the coverage)."""
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    if imagery:
        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        rgb[:40, :] = 90  # raster row 0 is y_max: the NORTH half has imagery
        write_rasters(tmp_path / "rasters", BLOCK, rgb, np.zeros((80, 80), dtype=np.float32))
    bake_block(BLOCK, RECORDS, tiles, rgbd_dir=tmp_path / "rasters",
               bounds=None if imagery else WorldBounds(x_min=0.0, y_min=0.0, x_max=8.0, y_max=8.0))
    rows = json.loads((tiles / f"{BLOCK}_features.json").read_text())["prior"]
    return {r["osm_id"]: r for r in rows}


def test_records_clip_to_the_imagery(tmp_path, monkeypatch, caplog) -> None:
    """North imagery covers the HIGH cell rows; wholly-south records drop and
    straddlers keep only their on-imagery cells, centroid included."""
    with caplog.at_level(logging.WARNING):
        rows = _bake(tmp_path, monkeypatch, imagery=True)
    assert sorted(rows) == ["w_all", "w_straddle"]
    assert rows["w_all"]["cells"] == [[r, c] for r in (2, 3) for c in range(4)]
    assert rows["w_straddle"]["cells"] == [[2, 0], [2, 1]]
    assert rows["w_straddle"]["centroid_row"] == 2.0
    # The straddler's true bbox center (2.0, 3.0) is over no-data: the bake warns.
    assert "Straddle House" in caplog.text


def test_no_imagery_keeps_every_rasterized_cell(tmp_path, monkeypatch) -> None:
    rows = _bake(tmp_path, monkeypatch, imagery=False)
    assert sorted(rows) == ["w_all", "w_south", "w_straddle"]
    assert len(rows["w_all"]["cells"]) == 16


def test_an_unreadable_ortho_refuses_the_bake(monkeypatch, tmp_path) -> None:
    """A block whose imagery cannot be read has no coverage to clip to."""
    def unreadable(*_a, **_k):
        raise OSError("truncated raster")

    monkeypatch.setattr(camera_model, "load_ortho_or_rasters", unreadable)
    with pytest.raises(BakeError, match="truncated raster"):
        _cell_coverage(BLOCK, WorldBounds(x_min=0.0, y_min=0.0, x_max=8.0, y_max=8.0), (4, 4), 2.0, tmp_path)
