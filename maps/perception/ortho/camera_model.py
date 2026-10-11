"""Top-down ortho loading: the 2D RGB+height ortho of a block from its
rasters (``load_ortho_or_rasters``).

The ortho is the single 2D source of truth. A PNG-sourced bake has
its interior holes (cells enclosed by coverage — photogrammetry gaps under
canopy) filled by ``ortho_fill.fill_ortho_holes`` at build time, while
boundary no-data (the block's exterior frame, water, scan edges) keeps
``alpha=0`` so no fill projects to a frame edge. A block whose rasters are a
georeferenced ``<map>_rgb.tif`` (the program's own map of the block, at its
native cell) is streamed straight into the memory-mapped cache
(``_bake_geotiff_mmap``) with no fill: it is a capture, and ground it did not
image stays black with ``alpha`` 0. The cache records the size and mtime of
the rasters it was built from and is rebuilt when they change; workers that
find no cache at once build it once, under a file lock beside it. Runtime
nadir perception crops directly
over this ortho (``observation.crop_ortho_observation``). This is an
ortho-only build: the runtime observation is always the nadir ortho crop.

World frame convention (matches ``perception/observation.py``):
    x = east, y = north, z = up (meters).
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
    """Where the MAPPABLE copy of a raster bake lives.

    Beside the rasters it was built from, so whatever process needs it finds
    it with no second path to configure, and pointing at different rasters
    invalidates it by construction. The ``_m0.1`` suffix names the 0.1 m/px
    grid of the PNG-sourced bake; caches already on disk carry it.
    """
    return Path(rgbd_dir) / "ortho" / f"{map_name}_m0.1.mmap"


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
    """The bake, memory-mapped. ``None`` when the cache is absent, partial,
    or built from other rasters than ``stamp`` names.

    ``mmap_mode="r"`` is the point twice over. The arrays never enter this
    process's heap, so N bench workers on one node share ONE page-cache copy
    of a ~3.6 GB map instead of holding N of them; and the load is a few
    ``mmap`` calls instead of a decode.
    """
    meta_path = d / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("source") != stamp:
            return None
        bake: dict = {
            k: np.load(d / f"{k}.npy", mmap_mode="r") for k in _MMAP_ARRAYS
        }
    except (OSError, ValueError, KeyError):
        return None
    bake.update(
        x_min=float(meta["x_min"]),
        y_min=float(meta["y_min"]),
        px_per_m=float(meta["px_per_m"]),
    )
    return bake


def _publish_mmap(tmp: Path, d: Path) -> None:
    """RENAME a fully written cache dir into place.

    The readers are other bench workers starting at the same moment, and a
    half-written ``rgb.npy`` sitting at its final path is a torn ortho every
    one of them would map and trust. Rename is atomic: a reader sees the whole
    cache or no cache.

    A REBUILD HAS TO WIN. ``os.replace`` refuses a non-empty directory, so
    renaming onto a cache that already exists fails and leaves the stale one
    serving -- an ortho outliving the rasters it was built from, which is the
    one way this cache could be wrong rather than merely slow. The old cache is
    moved aside first. A reader arriving in the gap between the two renames
    sees no cache and builds, which is correct and only costs it the build.
    """
    old = d.with_name(f"{d.name}.old{os.getpid()}")
    try:
        if d.exists():
            os.replace(d, old)
        os.replace(tmp, d)
    finally:
        shutil.rmtree(old, ignore_errors=True)


def _write_ortho_mmap(d: Path, bake: dict, stamp: dict) -> None:
    """Publish an in-heap ``bake``, built from what ``stamp`` names, so the
    next process maps it instead of building it.

    Best-effort. A read-only or full filesystem costs the build again, never
    the run.
    """
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
    """The detail limit of a block's raster, when the raster states one.

    A georeferenced ``<map>_rgb.tif`` is the block's own map laid at the
    collection's native cell, so that cell is what the camera resolved. A
    PNG-sourced raster states no capture cell (it may be a resample of a
    coarser source), so it reports nothing rather than a guess.
    """
    path = _geotiff_pair(rgbd_dir, map_name)
    if path is None:
        return None
    import rasterio  # noqa: PLC0415

    with rasterio.open(path) as src:
        return float(abs(src.transform.a))


def _bake_geotiff_mmap(rgbd_dir, map_name: str, d: Path, stamp: dict, band_rows: int = 1024) -> None:
    """Stream ``<map>_rgb.tif`` + ``<map>.tif`` into the cache at ``d``,
    recording ``stamp`` as its source.

    The block's own map at its native cell is gigapixels (Cal City at 0.15 m is
    six), so nothing here holds more than one band of rows: rasterio reads a
    window and the three arrays are ``open_memmap`` files written band by band.
    The height plane's affine supplies the world extent and the cell, as it
    does for the PNG-sourced bake, and height under no-data is sentinel-cleared
    to 0 the same way.

    No hole fill. The raster is the program's own capture of the block, and
    ground the aircraft did not image stays black with ``alpha`` 0 -- the rule
    a live run's imagery follows -- rather than being inpainted from its
    neighbours across a gap between passes.

    Not best-effort: there is no in-heap alternative for a raster this size,
    so a destination that cannot be written raises instead of costing the
    run an OOM.
    """
    import rasterio  # noqa: PLC0415
    from numpy.lib.format import open_memmap  # noqa: PLC0415
    from rasterio.windows import Window  # noqa: PLC0415

    rgbd_dir = Path(rgbd_dir)
    tmp = d.with_name(f"{d.name}.tmp{os.getpid()}")
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        with rasterio.open(rgbd_dir / f"{map_name}_rgb.tif") as rgb_src, \
                rasterio.open(rgbd_dir / f"{map_name}.tif") as h_src:
            H, W = rgb_src.height, rgb_src.width
            if (h_src.height, h_src.width) != (H, W):
                raise ValueError(
                    f"{map_name}: the RGB raster is {W}x{H} and the height plane "
                    f"{h_src.width}x{h_src.height}; the pair shares one grid"
                )
            if rgb_src.count < 3:
                raise ValueError(f"{map_name}_rgb.tif carries {rgb_src.count} band(s); the RGB raster carries three")
            b = h_src.bounds
            meta = {
                "x_min": float(b.left),
                "y_min": float(b.bottom),
                "px_per_m": 1.0 / abs(float(h_src.transform.a)),
            }
            rgb = open_memmap(tmp / "rgb.npy", mode="w+", dtype=np.uint8, shape=(H, W, 3))
            height = open_memmap(tmp / "height.npy", mode="w+", dtype=np.float32, shape=(H, W))
            alpha = open_memmap(tmp / "alpha.npy", mode="w+", dtype=np.uint8, shape=(H, W))
            for r0 in range(0, H, band_rows):
                n = min(band_rows, H - r0)
                win = Window(0, r0, W, n)
                band = np.ascontiguousarray(rgb_src.read([1, 2, 3], window=win).transpose(1, 2, 0))
                covered = band.max(axis=2) > 0
                h = h_src.read(1, window=win).astype(np.float32)
                h[~covered] = 0.0
                rgb[r0:r0 + n] = band
                height[r0:r0 + n] = h
                alpha[r0:r0 + n] = covered.astype(np.uint8) * 255
            for m in (rgb, height, alpha):
                m.flush()
            del rgb, height, alpha
        (tmp / "meta.json").write_text(json.dumps({**meta, "source": stamp}))
        _publish_mmap(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def bake_ortho_from_rasters(rgbd_dir, map_name: str) -> dict:
    """Build the canonical filled ortho dict from a co-registered 0.1 m/px RGB
    PNG + height GeoTIFF (``<map>.png`` + ``<map>.tif`` under ``rgbd_dir``,
    the CityNav ``rgbd_upstream`` bake).

    Output contract ``{rgb, height, alpha, x_min, y_min, px_per_m}`` in the
    world frame (x=east, y=north; row 0 = y_max). The GeoTIFF affine supplies
    the world extent, so it co-registers with ``GROUND_LEVEL``/view-square
    geometry exactly as the vendored cropclient did. Interior holes are filled
    (:func:`ortho_fill.fill_ortho_holes`); boundary no-data keeps ``alpha`` 0
    and height 0.
    """
    import rasterio  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    rgbd_dir = Path(rgbd_dir)
    rgb = np.asarray(Image.open(rgbd_dir / f"{map_name}.png").convert("RGB"), dtype=np.uint8)
    with rasterio.open(rgbd_dir / f"{map_name}.tif") as src:
        height = np.asarray(src.read(1), dtype=np.float32)
        b = src.bounds
        px_per_m = 1.0 / abs(float(src.transform.a))
        x_min, y_min = float(b.left), float(b.bottom)
    if rgb.shape[:2] != height.shape:
        height = np.asarray(
            Image.fromarray(height).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.float32,
        )
    # Upstream nodata is literal 0 RGB; coverage = any nonzero channel.
    alpha = (rgb.max(axis=2) > 0).astype(np.uint8) * 255
    from maps.perception.ortho.ortho_fill import fill_ortho_holes  # noqa: PLC0415

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
    """Hold an exclusive lock on ``<d>.lock`` across processes, so of the
    workers that find no cache at once the first builds and publishes it and
    the rest wait, then map it. A directory that cannot hold the lock file
    (read-only, or a filesystem without ``flock``) builds unlocked: each
    process pays the build, and the atomic publish keeps every reader whole.
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

# A flight crops one map, so two entries cover it. Entry size scales with
# the map: rgb u8 + height f32 + alpha u8 runs ~130 MB for a SensatUrban
# block and ~3.6 GB for the ~460 MP unified calcity map (which only ever
# occupies one slot).
_ORTHO_CACHE_MAX_ENTRIES = 2


def load_ortho_or_rasters(map_name: str, *, rgbd_dir=None) -> dict | None:
    """Return the canonical ortho dict for ``map_name`` from the rasters under
    ``rgbd_dir``: a georeferenced ``<map>_rgb.tif`` streamed as captured, else
    the 0.1 m/px PNG + GeoTIFF built and filled. This is the ONE 2D
    source-of-truth loader shared by the observation and the ``look`` tool.
    Cached (LRU) so repeated crops of a block are cheap.
    """
    if rgbd_dir is None:
        return None
    key = (str(map_name), str(rgbd_dir))
    cached = _ORTHO_RASTER_CACHE.get(key)
    if cached is not None:
        return cached
    # Serialize the miss path per process. Two threads reach here on a cold
    # cache -- the camera pane (station/review_routes) and the look/observation
    # path -- and unlocked both build and both write the SAME PID-keyed tmp dir
    # in _write_ortho_mmap: one thread's os.replace(tmp, d) pulls the dir out
    # from under the other's np.save, and a reader's mmap outlives a writer's
    # rmtree -- a SIGSEGV that killed the station mid-flight. Under the lock the
    # first builds and publishes; the waiter re-reads the cache and maps it.
    with _ORTHO_RASTER_BUILD_LOCK:
        cached = _ORTHO_RASTER_CACHE.get(key)
        if cached is not None:
            return cached
        # MAP IT IF SOMEONE ALREADY BUILT IT FROM THESE RASTERS. Building this
        # from rasters is ~20 s for the unified Cal City map and every process
        # pays it: measured on a 48-worker arm, 47 cold builds cost 936 s, more
        # than the 435 warm stops in the same window put together. The result
        # is a pure function of the rasters, so the first process to build it
        # publishes it under the file lock and the rest map it -- which also
        # collapses N heap copies of a ~3.6 GB map into one page-cache copy the
        # whole node shares.
        mm = _mmap_dir(rgbd_dir, map_name)
        stamp = _source_stamp(rgbd_dir, map_name)
        bake = _read_ortho_mmap(mm, stamp)
        if bake is None:
            with _build_lock(mm):
                bake = _read_ortho_mmap(mm, stamp)
                if bake is None and _geotiff_pair(rgbd_dir, map_name) is not None:
                    # Gigapixels: never in the heap. Streamed into the cache and
                    # mapped back like any other process would map it.
                    _bake_geotiff_mmap(rgbd_dir, map_name, mm, stamp)
                    bake = _read_ortho_mmap(mm, stamp)
                    if bake is None:
                        raise RuntimeError(f"{map_name}: the ortho cache at {mm} was written but does not read back")
                elif bake is None:
                    bake = bake_ortho_from_rasters(rgbd_dir, map_name)
                    _write_ortho_mmap(mm, bake, stamp)
        if len(_ORTHO_RASTER_CACHE) >= _ORTHO_CACHE_MAX_ENTRIES:
            _ORTHO_RASTER_CACHE.pop(next(iter(_ORTHO_RASTER_CACHE)))
        _ORTHO_RASTER_CACHE[key] = bake
        return bake


def ortho_world_bounds(map_name: str, *, rgbd_dir=None) -> tuple[float, float, float, float]:
    """World extent ``(x_min, y_min, x_max, y_max)`` of the canonical ortho.

    Used by the ``look`` tool to shift a near-edge crop window in-bounds. Reads
    the same source as :func:`load_ortho_or_rasters`.
    """
    ortho = load_ortho_or_rasters(map_name, rgbd_dir=rgbd_dir)
    if ortho is None:
        raise FileNotFoundError(f"No ortho for {map_name!r} (pass rgbd_dir).")
    x_min, y_min, ppm = ortho["x_min"], ortho["y_min"], ortho["px_per_m"]
    h, w = ortho["rgb"].shape[:2]
    return (float(x_min), float(y_min), float(x_min + w / ppm), float(y_min + h / ppm))


def ortho_gsd_m(map_name: str, *, rgbd_dir=None) -> float:
    """Metres per pixel of the canonical ortho GRID for ``map_name``.

    The divisor that turns a window in metres into delivered pixels, so a
    caller sizing a crop against a pixel budget reads the map's real grid
    instead of assuming one. Not the camera's resolution: CityNav's grid is
    0.1 m/px and Cal City's is the block raster's own cell, so this says how
    many PIXELS a span costs, never how much DETAIL it holds.
    """
    ortho = load_ortho_or_rasters(map_name, rgbd_dir=rgbd_dir)
    if ortho is None:
        raise FileNotFoundError(f"No ortho for {map_name!r} (pass rgbd_dir).")
    return 1.0 / float(ortho["px_per_m"])
