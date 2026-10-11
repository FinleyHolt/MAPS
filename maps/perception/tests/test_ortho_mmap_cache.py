"""The ortho is built once and mapped after, and what it maps is what it built.

Building the unified Cal City ortho from its rasters is ~20 s and every worker
process paid it: on a 48-worker arm, 47 cold builds cost 936 s -- more than the
435 warm stops in the same window put together. The result is a pure function
of the rasters, so the first process to build it publishes it and the rest map
it.

Two properties, and neither is about speed:

  * what comes back from the cache is what went in, EXACTLY. This is the hole-
    filled nadir ortho every crop is taken over, so a byte that changes here is
    a pixel the segmenter sees differently.
  * a reader never sees a HALF-WRITTEN cache. The readers are other workers
    starting at the same moment, and a torn ``rgb.npy`` at its final path is an
    ortho all of them would map and trust.
"""

from __future__ import annotations

import json

import numpy as np

from maps.perception.ortho import camera_model as cm

# What a cache under test was built from (the rasters it names need not exist).
_STAMP = {"blk.png": [1, 2], "blk.tif": [3, 4]}


def _bake(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "rgb": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
        "height": rng.random((48, 64)).astype(np.float32) * 30.0,
        "alpha": (rng.random((48, 64)) > 0.2).astype(np.uint8) * 255,
        "x_min": 1234.5,
        "y_min": -678.25,
        "px_per_m": 10.0,
    }


def test_the_round_trip_is_exact(tmp_path):
    src = _bake()
    d = tmp_path / "calcity_m0.1.mmap"
    cm._write_ortho_mmap(d, src, _STAMP)
    got = cm._read_ortho_mmap(d, _STAMP)

    assert got is not None
    for k in ("rgb", "height", "alpha"):
        assert np.array_equal(got[k], src[k]), k
        assert got[k].dtype == src[k].dtype, k
    for k in ("x_min", "y_min", "px_per_m"):
        assert got[k] == src[k], k
    # vacuity floor: an all-zero fixture would satisfy array_equal for free
    assert src["rgb"].any() and src["height"].any()


def test_it_is_mapped_not_loaded(tmp_path):
    """The memory claim, not just the value claim: N workers holding one
    ~3.6 GB map between them is the reason this exists."""
    d = tmp_path / "m.mmap"
    cm._write_ortho_mmap(d, _bake(), _STAMP)
    got = cm._read_ortho_mmap(d, _STAMP)
    assert isinstance(got["rgb"], np.memmap), type(got["rgb"])
    assert not got["rgb"].flags.writeable


def test_a_missing_or_partial_cache_reads_as_absent(tmp_path):
    """Absent is a build. PARTIAL must also be a build -- reading half a cache
    is the failure the atomic rename exists to prevent, and if one ever does
    appear it has to look like nothing rather than like an ortho."""
    assert cm._read_ortho_mmap(tmp_path / "nope.mmap", _STAMP) is None

    d = tmp_path / "partial.mmap"
    cm._write_ortho_mmap(d, _bake(), _STAMP)
    (d / "alpha.npy").unlink()
    assert cm._read_ortho_mmap(d, _STAMP) is None

    d2 = tmp_path / "nometa.mmap"
    cm._write_ortho_mmap(d2, _bake(), _STAMP)
    (d2 / "meta.json").unlink()
    assert cm._read_ortho_mmap(d2, _STAMP) is None


def test_a_write_leaves_no_temp_behind_and_publishes_atomically(tmp_path):
    d = tmp_path / "atomic.mmap"
    cm._write_ortho_mmap(d, _bake(), _STAMP)
    assert d.is_dir()
    assert not list(tmp_path.glob("*.tmp*")), list(tmp_path.glob("*.tmp*"))
    assert json.loads((d / "meta.json").read_text())["px_per_m"] == 10.0


def test_a_second_write_replaces_the_first(tmp_path):
    """A rebuilt ortho has to win, or a stale cache outlives its rasters."""
    d = tmp_path / "replace.mmap"
    cm._write_ortho_mmap(d, _bake(seed=1), _STAMP)
    second = _bake(seed=2)
    cm._write_ortho_mmap(d, second, _STAMP)
    got = cm._read_ortho_mmap(d, _STAMP)
    assert np.array_equal(got["rgb"], second["rgb"])


def test_an_unwritable_destination_costs_the_build_not_the_run(tmp_path):
    """Best-effort by design: a read-only filesystem must not raise into a
    flight."""
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        cm._write_ortho_mmap(ro / "x.mmap", _bake(), _STAMP)  # must not raise
        assert cm._read_ortho_mmap(ro / "x.mmap", _STAMP) is None
    finally:
        ro.chmod(0o700)


def test_concurrent_cold_cache_builds_once_under_the_lock(tmp_path, monkeypatch):
    """Two threads reaching a cold cache at once build ONE ortho, not two.

    The camera pane and the look/observation path both call this in one
    process; unlocked they both built and both wrote the same PID-keyed tmp
    dir, so one thread's ``os.replace(tmp, d)`` pulled the dir out from under
    the other's ``np.save`` and a reader's mmap outlived a writer's ``rmtree``
    -- the SIGSEGV that killed the station mid-flight. The build lock plus a
    re-check inside it must serialize them to a single build.
    """
    import threading
    import time

    cm._ORTHO_RASTER_CACHE.clear()
    builds: list[str] = []

    def fake_bake(rgbd_dir, map_name):
        builds.append(map_name)
        time.sleep(0.05)  # widen the window an unlocked race would exploit
        return {
            "rgb": np.zeros((2, 2, 3), np.uint8),
            "height": np.zeros((2, 2), np.float32),
            "alpha": np.zeros((2, 2), np.uint8),
            "x_min": 0.0,
            "y_min": 0.0,
            "px_per_m": 10.0,
        }

    monkeypatch.setattr(cm, "_read_ortho_mmap", lambda d, stamp: None)
    monkeypatch.setattr(cm, "_source_stamp", lambda rgbd_dir, map_name: _STAMP)
    monkeypatch.setattr(cm, "bake_ortho_from_rasters", fake_bake)
    monkeypatch.setattr(cm, "_write_ortho_mmap", lambda d, bake, stamp: None)

    barrier = threading.Barrier(2)
    results: list = []

    def worker():
        barrier.wait()
        results.append(
            cm.load_ortho_or_rasters("calcity", rgbd_dir=str(tmp_path))
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1, f"expected ONE build under the lock, got {len(builds)}"
    assert results[0] is results[1], "both threads must get the one published bake"
    cm._ORTHO_RASTER_CACHE.clear()


def _geotiff_pair(d, seed=3, H=37, W=53, px=0.15, ground=610.0):
    """A block's rasters as the release ships them: ``<map>_rgb.tif`` + the flat height plane."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    rgb = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    rgb[5:9, 10:20] = 0  # ground the aircraft did not image
    rgb[0, :] = 0
    tr = from_origin(1200.0, 3400.0 + H * px, px, px)
    with rasterio.open(d / "blk_rgb.tif", "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
                       transform=tr, compress="deflate") as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)
    with rasterio.open(d / "blk.tif", "w", driver="GTiff", width=W, height=H, count=1, dtype="float32",
                       transform=tr) as dst:
        dst.write(np.full((H, W), ground, np.float32), 1)
    return rgb


def test_a_geotiff_pair_is_streamed_into_the_cache_and_never_filled(tmp_path):
    """The block's own map is gigapixels, so it goes raster -> mmap a band of
    rows at a time, and ground the aircraft did not image stays black."""
    rgb = _geotiff_pair(tmp_path)
    cm._ORTHO_RASTER_CACHE.clear()
    got = cm.load_ortho_or_rasters("blk", rgbd_dir=str(tmp_path))
    cm._ORTHO_RASTER_CACHE.clear()

    assert isinstance(got["rgb"], np.memmap) and not got["rgb"].flags.writeable
    assert np.array_equal(got["rgb"], rgb)
    covered = rgb.max(axis=2) > 0
    assert np.array_equal(got["alpha"] > 0, covered)
    assert np.all(got["height"][covered] == 610.0) and np.all(got["height"][~covered] == 0.0)
    assert (got["x_min"], got["y_min"]) == (1200.0, 3400.0)
    assert got["px_per_m"] == 1.0 / 0.15
    # vacuity floor: the hole is a real interior hole, so a fill WOULD have changed it
    assert (~covered)[5:9, 10:20].all() and covered[4, 10] and covered[9, 10]
    assert not list(tmp_path.glob("ortho/*.tmp*"))


def test_the_streamed_bake_matches_the_in_heap_bake_band_for_band(tmp_path):
    """Same contract as the PNG path where the map has pixels (the PNG path
    also fills enclosed holes): one band boundary must not shift a row."""
    from PIL import Image

    rgb = _geotiff_pair(tmp_path, H=41)
    Image.fromarray(rgb).save(tmp_path / "blk.png")
    heap = cm.bake_ortho_from_rasters(tmp_path, "blk")
    cm._bake_geotiff_mmap(tmp_path, "blk", tmp_path / "s.mmap", _STAMP, band_rows=7)
    got = cm._read_ortho_mmap(tmp_path / "s.mmap", _STAMP)
    covered = rgb.max(axis=2) > 0
    for k in ("rgb", "alpha", "height"):
        assert np.array_equal(got[k][covered], heap[k][covered]), k
    for k in ("x_min", "y_min", "px_per_m"):
        assert got[k] == heap[k], k
    # The streamed bake sentinel-clears height under no-data and never fills.
    assert np.all(got["height"][~covered] == 0.0) and (~covered).any()
    assert np.all(got["alpha"][~covered] == 0)


def test_an_unwritable_cache_raises_for_a_geotiff_pair(tmp_path):
    """No in-heap fallback exists for a raster this size, so the failure is loud."""
    import pytest

    _geotiff_pair(tmp_path)
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        with pytest.raises(OSError):
            cm._bake_geotiff_mmap(tmp_path, "blk", ro / "x.mmap", _STAMP)
    finally:
        ro.chmod(0o700)
