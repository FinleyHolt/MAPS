"""Under a live station a look crops what the run captured, never the bake."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.search.render import look_render


class _Imagery:
    """A run that imaged the western ``frac`` of every window it is asked for."""

    gsd_m = 0.25

    def __init__(self, frac=0.6):
        self.frac = frac
        self.asked: list[tuple] = []

    def ortho(self, window):
        self.asked.append(tuple(round(v, 2) for v in window))
        x0, y0, x1, y1 = window
        w = int(round((x1 - x0) / self.gsd_m)); h = int(round((y1 - y0) / self.gsd_m))
        rgb = np.zeros((h, w, 3), np.uint8); alpha = np.zeros((h, w), np.uint8)
        seen = int(w * self.frac)
        rgb[:, :seen] = (90, 160, 60); alpha[:, :seen] = 255
        return {"rgb": rgb, "alpha": alpha, "height": np.zeros((h, w), np.float32),
                "x_min": x0, "y_min": y0, "px_per_m": 1.0 / self.gsd_m}


def test_a_look_under_a_live_station_is_cut_from_the_runs_imagery_and_says_what_is_unobserved(monkeypatch):
    pytest.importorskip("cv2")
    from maps.perception.ortho import camera_model

    def never(*a, **k):
        raise AssertionError("the bake's raster was read for a live look")

    monkeypatch.setattr(camera_model, "load_ortho_or_rasters", never)
    map_name = next(iter(GROUND_LEVEL))
    run = _Imagery()
    # The subject sits in the imaged half; a mark sits in the unimaged half.
    crops = look_render.render_candidate_crops(
        [{"center_fid": "b1", "center_xy": [1000.0, 2000.0], "center_label": "b1", "annotate": True, "marks": []}],
        map_name=map_name, rgbd_dir="/nowhere", imagery=run)
    assert len(crops) == 1 and run.asked, "no crop, or the run's imagery was never asked"
    rec = crops[0]
    assert rec["source"] == "run" and rec["source_gsd_m"] == 0.25
    assert rec["subject_on_data"] is True
    assert 0.5 < rec["window_valid_frac"] < 0.7, rec["window_valid_frac"]
    x0, y0, x1, y1 = rec["world_window"]
    assert x0 < 1000.0 < x1 and y0 < 2000.0 < y1
    # The unimaged half is black, the imaged half is the run's colour.
    import io

    from PIL import Image

    img = np.asarray(Image.open(io.BytesIO(rec["jpeg"])).convert("RGB"))
    h, w = img.shape[:2]
    left, right = img[h // 2, w // 8], img[h // 2, w - w // 8]
    assert int(left[1]) > 120 and int(right.max()) < 40, (left, right)
    # The partly imaged window says how much is unobserved, by that word.
    from maps.search.parse_compile.reasoner_loop import _look_vision_text

    assert "not imaged" in _look_vision_text(crops) and "raster" not in _look_vision_text(crops)
    # And a subject on ground the run never flew reads unobserved, by name.
    far = look_render.render_candidate_crops(
        [{"center_fid": "b2", "center_xy": [1000.0, 2000.0], "center_label": "b2", "annotate": True, "marks": []}],
        map_name=map_name, rgbd_dir="/nowhere", imagery=_Imagery(0.0))
    words = _look_vision_text(far)
    assert far[0]["source"] == "run" and far[0]["subject_on_data"] is False and "unobserved" in words, words


def test_the_bench_still_crops_the_bake(monkeypatch):
    """Without a run's imagery the renderer reads the raster it always did:
    the simulated camera is a crop of it."""
    from maps.perception.ortho import camera_model

    seen: list = []

    def loader(map_name, *a, **k):
        seen.append(map_name)
        w = h = 400
        return {"rgb": np.full((h, w, 3), 120, np.uint8), "alpha": np.full((h, w), 255, np.uint8),
                "height": np.zeros((h, w), np.float32), "x_min": 900.0, "y_min": 1900.0, "px_per_m": 4.0,
                "m_per_px": 0.25}

    monkeypatch.setattr(camera_model, "load_ortho_or_rasters", loader)
    monkeypatch.setattr(look_render, "ortho_world_bounds", lambda *a, **k: (900.0, 1900.0, 1000.0, 2000.0), raising=False)
    map_name = next(iter(GROUND_LEVEL))
    crops = look_render.render_candidate_crops(
        [{"center_fid": "b1", "center_xy": [950.0, 1950.0], "center_label": "b1", "annotate": True, "marks": []}],
        map_name=map_name, rgbd_dir="/somewhere")
    assert seen and crops and crops[0]["source"] == "bake"


def test_a_bake_crops_source_detail_is_the_configured_rasters_own_cell(monkeypatch, tmp_path):
    """The look caption's "detail limited by" figure is a fact of the raster
    the bench is configured with: a block's own map (``<map>_rgb.tif``) is a
    capture laid at its native cell, so its cell is the detail limit. A
    raster that states no capture cell (the PNG rasters) reports nothing,
    whatever the map is called."""
    import rasterio
    from rasterio.transform import from_origin

    from maps.perception.ortho import camera_model

    # Named for the bench whose raster this used to be looked up by.
    map_name = "calcity"
    monkeypatch.setitem(GROUND_LEVEL, map_name, 0.0)

    def loader(map_name, *a, **k):
        w = h = 400
        return {"rgb": np.full((h, w, 3), 120, np.uint8), "alpha": np.full((h, w), 255, np.uint8),
                "height": np.zeros((h, w), np.float32), "x_min": 900.0, "y_min": 1900.0, "px_per_m": 4.0,
                "m_per_px": 0.25}

    monkeypatch.setattr(camera_model, "load_ortho_or_rasters", loader)
    monkeypatch.setattr(look_render, "ortho_world_bounds", lambda *a, **k: (900.0, 1900.0, 1000.0, 2000.0), raising=False)
    req = [{"center_fid": "b1", "center_xy": [950.0, 1950.0], "center_label": "b1", "annotate": True, "marks": []}]

    png_dir = tmp_path / "png"
    png_dir.mkdir()
    crops = look_render.render_candidate_crops(req, map_name=map_name, rgbd_dir=str(png_dir))
    assert crops[0]["source"] == "bake" and crops[0]["source_gsd_m"] is None

    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    with rasterio.open(pair_dir / f"{map_name}_rgb.tif", "w", driver="GTiff", width=8, height=6, count=3,
                       dtype="uint8", transform=from_origin(0.0, 0.9, 0.15, 0.15)) as dst:
        dst.write(np.zeros((3, 6, 8), np.uint8))
    crops = look_render.render_candidate_crops(req, map_name=map_name, rgbd_dir=str(pair_dir))
    assert crops[0]["source_gsd_m"] == pytest.approx(0.15)
