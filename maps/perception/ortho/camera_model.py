"""Top-down ortho loading: the 2D RGB+height ortho of a block from its
rasters (``load_ortho_or_rasters``), the one source every observation crops.

A PNG-sourced bake has its interior holes filled
(``ortho_fill.fill_ortho_holes``); a georeferenced ``<map>_rgb.tif`` is
streamed into the memory-mapped cache unfilled, so ground it did not image
stays black with ``alpha`` 0. The cache records the rasters it was built from
and is rebuilt when they change; concurrent workers build it once, under a
file lock. World frame: x = east, y = north, z = up (meters).
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import logging
import os
import shutil
import threading
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from maps.perception.ortho.ortho_fill import fill_ortho_holes

log = logging.getLogger(__name__)

# The rasters opened here are our own baked orthos, not untrusted uploads;
# a whole-AOI map (~460 MP for calcity at 0.3 m/px) exceeds PIL's
# decompression-bomb ceiling (~178 MP), so lift it deliberately.
try:
    from PIL import Image as _pil_image

    _pil_image.MAX_IMAGE_PIXELS = None
except ImportError:  # pragma: no cover - PIL is a hard dep in practice
    pass


def _mmap_dir(rgbd_dir, map_name: str) -> Path:
    """Where the memory-mapped copy of a raster bake lives: beside the rasters
    it was built from."""
    return Path(rgbd_dir) / "ortho" / f"{map_name}.mmap"


# The three arrays that carry the pixels. Everything else in the bake dict is
# a scalar and rides in ``meta.json`` beside them.
_MMAP_ARRAYS = ("rgb", "height", "alpha")


def _source_stamp(rgbd_dir, map_name: str) -> dict:
    """What the published cache must have been built from: the RGB raster
    (``<map>_rgb.tif`` when present, else ``<map>.png``) and the height plane
    ``<map>.tif``, each as ``[byte size, mtime_ns]``. A raster rewritten in
    place changes its mtime, so its stale cache is rebuilt."""
    rgb = _geotiff_pair(rgbd_dir, map_name) or Path(rgbd_dir) / f"{map_name}.png"
    out = {}
    for path in (rgb, Path(rgbd_dir) / f"{map_name}.tif"):
        st = path.stat()
        out[path.name] = [int(st.st_size), int(st.st_mtime_ns)]
    return out


def _read_ortho_mmap(d: Path, stamp: dict) -> dict | None:
    """The bake, memory-mapped (workers on a node share one page-cache copy).
    ``None`` when the cache is absent, partial, or built from other rasters
    than ``stamp`` names."""
    meta_path = d / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("source") != stamp:
            return None
        bake: dict = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in _MMAP_ARRAYS}
    except (OSError, ValueError, KeyError):
        return None
    bake.update(
        x_min=float(meta["x_min"]),
        y_min=float(meta["y_min"]),
        px_per_m=float(meta["px_per_m"]),
    )
    return bake


def _publish_mmap(tmp: Path, d: Path) -> None:
    """Rename a fully written cache dir into place, so a reader sees the whole
    cache or none. An existing (stale) cache is moved aside first, since
    ``os.replace`` refuses a non-empty directory and the rebuild has to win.
    """
    old = d.with_name(f"{d.name}.old{os.getpid()}")
    try:
        if d.exists():
            os.replace(d, old)
        os.replace(tmp, d)
    finally:
        shutil.rmtree(old, ignore_errors=True)


def _write_ortho_mmap(d: Path, bake: dict, stamp: dict) -> None:
    """Publish an in-heap ``bake``, built from what ``stamp`` names, for the
    next process to map. Best-effort: a failed write costs a rebuild, never
    the run."""
    tmp = d.with_name(f"{d.name}.tmp{os.getpid()}")
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        for k in _MMAP_ARRAYS:
            np.save(tmp / f"{k}.npy", np.ascontiguousarray(bake[k]))
        meta = {k: float(bake[k]) for k in ("x_min", "y_min", "px_per_m")}
        (tmp / "meta.json").write_text(json.dumps({**meta, "source": stamp}))
        _publish_mmap(tmp, d)
    except OSError as exc:
        log.warning("ortho mmap cache write failed at %s: %s", d, exc)
        shutil.rmtree(tmp, ignore_errors=True)


def _geotiff_pair(rgbd_dir, map_name: str) -> Path | None:
    """The block's georeferenced RGB raster, when the rasters are that kind."""
    path = Path(rgbd_dir) / f"{map_name}_rgb.tif"
    return path if path.is_file() else None


@functools.lru_cache(maxsize=8)
def raster_source_gsd_m(rgbd_dir, map_name: str) -> float | None:
    """The capture cell of a georeferenced ``<map>_rgb.tif``, in meters;
    ``None`` for a PNG-sourced raster, which states none."""
    path = _geotiff_pair(rgbd_dir, map_name)
    if path is None:
        return None
    import rasterio

    with rasterio.open(path) as src:
        return float(abs(src.transform.a))


#: Rows the streamed bake reads per band.
_BAND_ROWS = 1024


def _bake_geotiff_mmap(rgbd_dir, map_name: str, d: Path, stamp: dict) -> None:
    """Stream ``<map>_rgb.tif`` + ``<map>.tif`` into the cache at ``d`` one
    band of rows at a time (the raster is gigapixels), recording ``stamp`` as
    its source. The height plane's affine gives the extent and cell; height
    under no-data is 0. No hole fill. A destination that cannot be written
    raises: there is no in-heap alternative at this size.
    """
    import rasterio
    from rasterio.windows import Window

    rgbd_dir = Path(rgbd_dir)
    tmp = d.with_name(f"{d.name}.tmp{os.getpid()}")
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        with (
            rasterio.open(rgbd_dir / f"{map_name}_rgb.tif") as rgb_src,
            rasterio.open(rgbd_dir / f"{map_name}.tif") as h_src,
        ):
            H, W = rgb_src.height, rgb_src.width
            if (h_src.height, h_src.width) != (H, W):
                raise ValueError(
                    f"{map_name}: the RGB raster is {W}x{H} and the height plane "
                    f"{h_src.width}x{h_src.height}; the pair shares one grid"
                )
            if rgb_src.count < 3:
                raise ValueError(
                    f"{map_name}_rgb.tif carries {rgb_src.count} band(s); the RGB raster carries three"
                )
            b = h_src.bounds
            meta = {
                "x_min": float(b.left),
                "y_min": float(b.bottom),
                "px_per_m": 1.0 / abs(float(h_src.transform.a)),
            }
            rgb = open_memmap(tmp / "rgb.npy", mode="w+", dtype=np.uint8, shape=(H, W, 3))
            height = open_memmap(tmp / "height.npy", mode="w+", dtype=np.float32, shape=(H, W))
            alpha = open_memmap(tmp / "alpha.npy", mode="w+", dtype=np.uint8, shape=(H, W))
            for r0 in range(0, H, _BAND_ROWS):
                n = min(_BAND_ROWS, H - r0)
                win = Window(0, r0, W, n)
                band = np.ascontiguousarray(rgb_src.read([1, 2, 3], window=win).transpose(1, 2, 0))
                covered = band.max(axis=2) > 0
                h = h_src.read(1, window=win).astype(np.float32)
                h[~covered] = 0.0
                rgb[r0 : r0 + n] = band
                height[r0 : r0 + n] = h
                alpha[r0 : r0 + n] = covered.astype(np.uint8) * 255
            for m in (rgb, height, alpha):
                m.flush()
            del rgb, height, alpha
        (tmp / "meta.json").write_text(json.dumps({**meta, "source": stamp}))
        _publish_mmap(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def bake_ortho_from_rasters(rgbd_dir, map_name: str) -> dict:
    """The filled ortho ``{rgb, height, alpha, x_min, y_min, px_per_m}``
    (row 0 = y_max) from a co-registered 0.1 m/px ``<map>.png`` + height
    ``<map>.tif`` under ``rgbd_dir``; the GeoTIFF affine supplies the extent.
    Interior holes are filled (:func:`ortho_fill.fill_ortho_holes`); boundary
    no-data keeps ``alpha`` 0 and height 0.
    """
    import rasterio
    from PIL import Image

    rgbd_dir = Path(rgbd_dir)
    rgb = np.asarray(Image.open(rgbd_dir / f"{map_name}.png").convert("RGB"), dtype=np.uint8)
    with rasterio.open(rgbd_dir / f"{map_name}.tif") as src:
        height = np.asarray(src.read(1), dtype=np.float32)
        b = src.bounds
        px_per_m = 1.0 / abs(float(src.transform.a))
        x_min, y_min = float(b.left), float(b.bottom)
    if rgb.shape[:2] != height.shape:
        height = np.asarray(
            Image.fromarray(height).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST),
            dtype=np.float32,
        )
    # Upstream nodata is literal 0 RGB; coverage = any nonzero channel.
    alpha = (rgb.max(axis=2) > 0).astype(np.uint8) * 255

    rgb, height, alpha = fill_ortho_holes(rgb, height, alpha)
    height = height.copy()
    height[alpha == 0] = 0.0  # boundary no-data: sentinel-clear height
    return {
        "rgb": rgb,
        "height": height,
        "alpha": alpha,
        "x_min": x_min,
        "y_min": y_min,
        "px_per_m": float(px_per_m),
    }


@contextlib.contextmanager
def _build_lock(d: Path):
    """An exclusive cross-process lock on ``<d>.lock``: the first worker
    builds the cache, the rest wait and map it. Where the lock cannot be taken
    each process builds unlocked; the atomic publish keeps readers whole.
    """
    fd = None
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(d.with_name(f"{d.name}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        log.warning("ortho build lock unavailable beside %s: %s", d, exc)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)


_ORTHO_RASTER_CACHE: dict[tuple, dict] = {}
_ORTHO_RASTER_BUILD_LOCK = threading.Lock()

# A flight crops one map, so two entries cover it.
_ORTHO_CACHE_MAX_ENTRIES = 2


def load_ortho_or_rasters(map_name: str, *, rgbd_dir=None) -> dict | None:
    """The ortho dict for ``map_name`` from the rasters under ``rgbd_dir``
    (``None`` without one): a georeferenced ``<map>_rgb.tif`` as captured,
    else the PNG + GeoTIFF built and filled. Cached in-process and on disk.
    """
    if rgbd_dir is None:
        return None
    key = (str(map_name), str(rgbd_dir))
    cached = _ORTHO_RASTER_CACHE.get(key)
    if cached is not None:
        return cached
    # Serialize the miss path per process: two threads building at once would
    # share _write_ortho_mmap's PID-keyed tmp dir and tear each other's write.
    with _ORTHO_RASTER_BUILD_LOCK:
        cached = _ORTHO_RASTER_CACHE.get(key)
        if cached is not None:
            return cached
        # Map it if another process already built it from these rasters: the
        # first builds and publishes under the file lock, the rest map it.
        mm = _mmap_dir(rgbd_dir, map_name)
        stamp = _source_stamp(rgbd_dir, map_name)
        bake = _read_ortho_mmap(mm, stamp)
        if bake is None:
            with _build_lock(mm):
                bake = _read_ortho_mmap(mm, stamp)
                if bake is None and _geotiff_pair(rgbd_dir, map_name) is not None:
                    # Gigapixels: streamed into the cache, never into the heap.
                    _bake_geotiff_mmap(rgbd_dir, map_name, mm, stamp)
                    bake = _read_ortho_mmap(mm, stamp)
                    if bake is None:
                        raise RuntimeError(
                            f"{map_name}: the ortho cache at {mm} was written but does not read back"
                        )
                elif bake is None:
                    bake = bake_ortho_from_rasters(rgbd_dir, map_name)
                    _write_ortho_mmap(mm, bake, stamp)
        if len(_ORTHO_RASTER_CACHE) >= _ORTHO_CACHE_MAX_ENTRIES:
            _ORTHO_RASTER_CACHE.pop(next(iter(_ORTHO_RASTER_CACHE)))
        _ORTHO_RASTER_CACHE[key] = bake
        return bake


def ortho_world_bounds(map_name: str, *, rgbd_dir=None) -> tuple[float, float, float, float]:
    """World extent ``(x_min, y_min, x_max, y_max)`` of the ortho."""
    ortho = load_ortho_or_rasters(map_name, rgbd_dir=rgbd_dir)
    if ortho is None:
        raise FileNotFoundError(f"No ortho for {map_name!r} (pass rgbd_dir).")
    x_min, y_min, ppm = ortho["x_min"], ortho["y_min"], ortho["px_per_m"]
    h, w = ortho["rgb"].shape[:2]
    return (float(x_min), float(y_min), float(x_min + w / ppm), float(y_min + h / ppm))


def ortho_gsd_m(map_name: str, *, rgbd_dir=None) -> float:
    """Metres per pixel of the ortho grid: what a span costs in pixels, not
    how much detail it holds (that is :func:`raster_source_gsd_m`)."""
    ortho = load_ortho_or_rasters(map_name, rgbd_dir=rgbd_dir)
    if ortho is None:
        raise FileNotFoundError(f"No ortho for {map_name!r} (pass rgbd_dir).")
    return 1.0 / float(ortho["px_per_m"])
