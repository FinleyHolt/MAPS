"""The mapped ortho is the one its rasters build, and one process builds it.

``load_ortho_or_rasters`` publishes the ortho as a memory-mapped cache beside
a block's rasters. A raster rewritten in place makes that cache stale: the
next load rebuilds it rather than serving the old pixels. Workers that start
together on a cold cache build it once: the first builds and publishes, the
others wait for it and map it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from maps.perception.ortho import camera_model

H, W = 40, 50


def _write_rasters(rgbd: Path, block: str, colour: tuple[int, int, int], kind: str = "png") -> Path:
    """The block's raster pair; returns the RGB raster's path."""
    import rasterio
    from PIL import Image
    from rasterio.transform import Affine

    tr = Affine(0.1, 0.0, 0.0, 0.0, -0.1, H * 0.1)
    with rasterio.open(rgbd / f"{block}.tif", "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", transform=tr) as dst:
        dst.write(np.full((1, H, W), 3.0, np.float32))
    rgb = np.full((H, W, 3), colour, np.uint8)
    if kind == "geotiff":
        path = rgbd / f"{block}_rgb.tif"
        with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=3, dtype="uint8",
                           transform=tr) as dst:
            dst.write(rgb.transpose(2, 0, 1))
    else:
        path = rgbd / f"{block}.png"
        Image.fromarray(rgb).save(path)
    return path


@pytest.mark.parametrize("kind", ["png", "geotiff"])
def test_a_raster_rewritten_in_place_rebuilds_the_mapped_ortho(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(camera_model, "_ORTHO_RASTER_CACHE", {})
    rgb_path = _write_rasters(tmp_path, "blk", (200, 30, 30), kind)
    first = camera_model.load_ortho_or_rasters("blk", rgbd_dir=tmp_path)
    assert tuple(first["rgb"][0, 0]) == (200, 30, 30)
    assert (tmp_path / "ortho" / "blk_m0.1.mmap" / "meta.json").is_file()

    size, mtime_ns = rgb_path.stat().st_size, rgb_path.stat().st_mtime_ns
    _write_rasters(tmp_path, "blk", (30, 30, 200), kind)
    assert rgb_path.stat().st_size == size  # the byte size alone cannot tell them apart
    os.utime(rgb_path, ns=(mtime_ns + 10**9, mtime_ns + 10**9))
    camera_model._ORTHO_RASTER_CACHE.clear()
    second = camera_model.load_ortho_or_rasters("blk", rgbd_dir=tmp_path)
    assert tuple(second["rgb"][0, 0]) == (30, 30, 200)


_WORKER = textwrap.dedent(
    """
    import sys, time
    from maps.perception.ortho import camera_model

    real = camera_model.bake_ortho_from_rasters

    def slow_build(*args, **kwargs):
        with open(sys.argv[2], "a") as log:
            log.write("built\\n")
        time.sleep(2.0)
        return real(*args, **kwargs)

    camera_model.bake_ortho_from_rasters = slow_build
    bake = camera_model.load_ortho_or_rasters("blk", rgbd_dir=sys.argv[1])
    assert tuple(bake["rgb"][0, 0]) == (200, 30, 30)
    """
)


def test_workers_starting_on_a_cold_cache_build_the_ortho_once(tmp_path):
    rgbd = tmp_path / "rgbd"
    rgbd.mkdir()
    _write_rasters(rgbd, "blk", (200, 30, 30))
    log = tmp_path / "builds.log"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    workers = [
        subprocess.Popen([sys.executable, "-c", _WORKER, str(rgbd), str(log)], env=env)
        for _ in range(4)
    ]
    assert [w.wait(timeout=120) for w in workers] == [0, 0, 0, 0]
    assert log.read_text().count("built") == 1
