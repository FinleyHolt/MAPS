"""The Cal City export: window placement, georef, exclusions, loader shapes.

Every raster here is a small synthetic UTM GeoTIFF; the real 89705 x 68069
ortho is never opened.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")

from rasterio.transform import from_origin  # noqa: E402

from baselines.released import calcity_export as ce  # noqa: E402

OE, ON = 405000.0, 3880000.0  # UTM origin of the local frame
SRC_RES = 0.15


def _ortho(path, width_m=300.0, height_m=300.0):
    """R encodes local x, G local y (2 units per metre), B is constant 7."""
    w, h = round(width_m / SRC_RES), round(height_m / SRC_RES)
    cols = (np.arange(w) + 0.5) * SRC_RES
    rows = height_m - (np.arange(h) + 0.5) * SRC_RES
    r = np.broadcast_to(np.clip(cols * 0.8, 0, 255), (h, w))
    g = np.broadcast_to(np.clip(rows[:, None] * 0.8, 0, 255), (h, w))
    data = np.stack([r, g, np.full((h, w), 7.0)]).round().astype("uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
        crs="EPSG:32611", transform=from_origin(OE, ON + height_m, SRC_RES, SRC_RES),
    ) as dst:
        dst.write(data)
    return (0.0, 0.0, width_m, height_m)


def test_place_block_holds_both_points_inside_the_imagery():
    imagery = (0.0, 0.0, 1000.0, 1000.0)
    for seed in range(50):
        placed = ce.place_block([(300.0, 300.0), (500.0, 450.0)], imagery, random.Random(seed))
        x0, y0 = placed
        assert x0 <= 300.0 - ce.MARGIN_M and x0 + ce.BLOCK_SIDE_M >= 500.0 + ce.MARGIN_M
        assert y0 <= 300.0 - ce.MARGIN_M and y0 + ce.BLOCK_SIDE_M >= 450.0 + ce.MARGIN_M
        assert 0.0 <= x0 and x0 + ce.BLOCK_SIDE_M <= 1000.0


def test_place_block_does_not_centre_on_the_target():
    imagery = (0.0, 0.0, 5000.0, 5000.0)
    centres = {
        ce.place_block([(1000.0, 1000.0), (1100.0, 1000.0)], imagery, random.Random(s))[0]
        for s in range(20)
    }
    assert len(centres) > 10  # the window moves; its centre carries no target fix


def test_place_block_reasons():
    imagery = (0.0, 0.0, 5000.0, 5000.0)
    rng = random.Random(0)
    assert ce.place_block([(0.0, 0.0), (390.0, 0.0)], imagery, rng) == "start_and_target_exceed_block"
    assert ce.place_block([(5.0, 2000.0), (50.0, 2000.0)], imagery, rng) == "window_leaves_imagery"


def test_block_rasters_round_trip_the_georef(tmp_path):
    _ortho(tmp_path / "ortho.tif")
    window = (40.0, 60.0, 80.0, 100.0)  # a 40 m block
    png, tif = tmp_path / "b.png", tmp_path / "b.tif"
    with rasterio.open(tmp_path / "ortho.tif") as src:
        ce.write_block_rasters(src, (OE, ON), window, png, tif)
    from PIL import Image

    rgb = np.asarray(Image.open(png))
    with rasterio.open(tif) as height:
        assert height.shape == rgb.shape[:2] == (400, 400)
        assert height.transform.a == pytest.approx(ce.RESOLUTION_M)
        assert (height.transform.c, height.transform.f) == pytest.approx((40.0, 100.0))
        assert float(height.read(1).max()) == 0.0  # flat ground at level 0
        for x, y in [(45.0, 65.0), (60.0, 80.0), (78.3, 98.1)]:
            r, c = height.index(x, y)  # what cropclient does for every crop
            assert abs(int(rgb[r, c, 0]) - x * 0.8) <= 1.5
            assert abs(int(rgb[r, c, 1]) - y * 0.8) <= 1.5
            assert rgb[r, c, 2] == 7


def _row(start, target, mask=None):
    return {
        "area": "calcity", "descriptions": ["the red roof beside Main Street"],
        "ann_ids": [0], "object_ids": [], "target_positions": [[*target, 0.0]],
        "marker_positions": [], "trajectory": [[*start, 125.0]],
        "target_mask": mask, "success_radius_m": 20.0,
    }


@pytest.fixture
def exported(tmp_path):
    imagery = _ortho(tmp_path / "ortho.tif", 900.0, 900.0)
    mask = [[440.0, 440.0], [460.0, 440.0], [460.0, 460.0], [440.0, 460.0]]
    rows = [
        _row((300.0, 300.0), (450.0, 450.0), mask),   # fits
        _row((50.0, 450.0), (850.0, 450.0)),          # 800 m apart
        _row((5.0, 450.0), (60.0, 450.0)),            # against the imagery edge
    ]
    shapes = {
        f"calcity::0#{i}": {"gt_polygon": r["target_mask"],
                            "target_xy": tuple(r["target_positions"][0][:2]),
                            "start_xy": tuple(r["trajectory"][0][:2]),
                            "description": r["descriptions"][0]}
        for i, r in enumerate(rows)
    }
    catalog = {
        "0": {"id": 0, "name": "Main Street", "object_type": "TrafficRoad",
              "position": [450.0, 0.0, 0.1], "dimension": [10.0, 5000.0, 0.2],
              "contour": [[445.0, -2000.0], [455.0, -2000.0], [455.0, 3000.0], [445.0, 3000.0]],
              "descriptions": [], "tags": {"highway": "primary"}},
        "1": {"id": 1, "name": "Far Hall", "object_type": "Building",
              "position": [880.0, 880.0, 4.0], "dimension": [5.0, 5.0, 8.0],
              "contour": [[878.0, 878.0], [882.0, 878.0], [882.0, 882.0], [878.0, 882.0]],
              "descriptions": []},
    }
    out = tmp_path / "export"
    manifest = ce.export(shapes, catalog, out, ortho_path=tmp_path / "ortho.tif",
                         utm_origin=(OE, ON), imagery=imagery, side_m=200.0)
    return out, manifest


def test_export_excludes_with_reasons(exported):
    _, manifest = exported
    assert list(manifest["blocks"]) == ["calcity_block_0"]
    assert manifest["excluded"] == {
        "calcity::0#1": "start_and_target_exceed_block",
        "calcity::0#2": "window_leaves_imagery",
    }
    assert manifest["n_rows"] == 3


def test_export_writes_loader_shapes(exported):
    out, manifest = exported
    data = out / "shards" / "shard_000" / "data"
    episodes = json.loads((data / "citynav" / "citynav_test_unseen.json").read_text())
    assert len(episodes) == 1
    ep = episodes[0]
    # Exactly the upstream MTurkTrajectory fields (no optional extras).
    assert set(ep) == {"area", "block", "object_ids", "ann_ids", "descriptions",
                       "trajectory", "marker_positions", "target_positions",
                       "total_score", "dist_marker_to_target", "split"}
    assert f"{ep['area']}_block_{ep['block']}" == "calcity_block_0"
    assert len(ep["trajectory"][0]) == 5 and ep["trajectory"][0][:2] == [300.0, 300.0]
    assert ep["marker_positions"] == ep["target_positions"] == [[450.0, 450.0, 0.0]]

    objects = json.loads((data / "cityrefer" / "objects.json").read_text())["calcity_block_0"]
    target = objects[str(ce.TARGET_OBJECT_ID)]
    assert target["name"] == "" and target["contour"][0] == [440.0, 440.0]
    road = objects["0"]
    assert set(road) == set(ce.CITYREFER_FIELDS) and road["map_name"] == "calcity_block_0"
    x0, y0, x1, y1 = manifest["blocks"]["calcity_block_0"]["bounds"]
    assert all(y0 - 1e-6 <= y <= y1 + 1e-6 for _, y in road["contour"])  # clipped
    assert "1" not in objects  # outside the window
    assert manifest["blocks"]["calcity_block_0"]["landmarks"] == ["Main Street"]

    blocks = json.loads((out / "blocks.json").read_text())
    assert blocks["GROUND_LEVEL"] == {"calcity_block_0": 0.0}
    assert blocks["MAP_BOUNDS"]["calcity_block_0"] == [x0, y0, x1, y1]
    assert (data / "rgbd" / "calcity_block_0.png").exists()
    with rasterio.open(data / "rgbd" / "calcity_block_0.tif") as tif:
        assert tif.bounds.left == pytest.approx(x0) and tif.bounds.top == pytest.approx(y1)
    link = data / "cityrefer" / "processed_descriptions.json"
    assert link.is_symlink() and link.resolve() == (out / "processed_descriptions.json").resolve()
