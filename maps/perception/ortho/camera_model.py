"""Top-down ortho mosaic baking + sensor-spec types.

Bakes a 0.10 m/px RGB+height orthomosaic from a SensatUrban PLY
(``bake_ortho_from_ply``), persists it next to the PLY (``save_ortho``
/ ``load_ortho``), and exposes the pinhole intrinsics dataclass
(``CameraIntrinsics``) the rest of the perception package consumes.

The bake is the single 2D source of truth. A PLY or PNG-sourced bake has
its interior holes (cells enclosed by coverage — photogrammetry gaps under
canopy) filled by ``ortho_fill.fill_ortho_holes`` at bake time, while
boundary no-data (the block's exterior frame, water, scan edges) keeps
``alpha=0`` so no fill projects to a frame edge. A block whose rasters are a
georeferenced ``<map>_rgb.tif`` (the program's own map of the block, at its
native cell) is streamed straight into the memory-mapped cache
(``_bake_geotiff_mmap``) with no fill: it is a capture, and ground it did not
image stays black with ``alpha`` 0. Runtime nadir perception crops directly
over this ortho (``observation.crop_ortho_observation``). This is an
ortho-only build: the runtime observation is always the nadir ortho crop.

World frame convention (matches ``perception/observation.py``):
    x = east, y = north, z = up (meters).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CameraIntrinsics:
    hfov_deg: float
    vfov_deg: float
    image_width: int
    image_height: int

    @property
    def fx(self) -> float:
        return self.image_width / (2.0 * np.tan(np.deg2rad(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.image_height / (2.0 * np.tan(np.deg2rad(self.vfov_deg) / 2.0))

    @property
    def cx(self) -> float:
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        return self.image_height / 2.0






# Module-level cache for decoded PLY point clouds keyed by absolute path.
# Only used by the offline ``bake_ortho_from_ply`` path; runtime never
# loads a PLY.
_PLY_CACHE: dict[str, object] = {}
_PLY_CACHE_MAX_ENTRIES = 2


def _load_ply_cloud(ply_path):
    """Return (and memoise) the Open3D PointCloud for ``ply_path``."""
    import open3d as o3d  # type: ignore[import-not-found]

    key = str(ply_path)
    cached = _PLY_CACHE.get(key)
    if cached is not None:
        return cached
    pc = o3d.io.read_point_cloud(key)
    if not pc.has_points():
        raise RuntimeError(f"PLY loaded empty: {ply_path}")
    if len(_PLY_CACHE) >= _PLY_CACHE_MAX_ENTRIES:
        _PLY_CACHE.pop(next(iter(_PLY_CACHE)))
    _PLY_CACHE[key] = pc
    return pc


def _res_tag(m_per_px: float) -> str:
    return f"{m_per_px:.3f}".rstrip("0").rstrip(".")


def _ortho_path(ply_path, m_per_px: float) -> Path:
    """Path of the orthomosaic bake for ``ply_path`` at ``m_per_px`` resolution."""
    src = Path(ply_path)
    return src.parent / "ortho" / f"{src.stem}_ortho_m{_res_tag(m_per_px)}.npz"


def _rgbd_ortho_path(rgbd_dir, map_name: str, m_per_px: float) -> Path:
    """Path of a prerendered ortho npz next to the rasters (raster-baked blocks)."""
    return Path(rgbd_dir) / "ortho" / f"{map_name}_ortho_m{_res_tag(m_per_px)}.npz"


def _mmap_dir(rgbd_dir, map_name: str, m_per_px: float) -> Path:
    """Where the MAPPABLE copy of a raster bake lives.

    Beside the rasters it was built from, so whatever process needs it finds
    it with no second path to configure, and pointing at different rasters
    invalidates it by construction.
    """
    return Path(rgbd_dir) / "ortho" / f"{map_name}_m{_res_tag(m_per_px)}.mmap"


# The three arrays that carry the pixels. Everything else in the bake dict is
# a scalar and rides in ``meta.json`` beside them.
_MMAP_ARRAYS = ("rgb", "height", "alpha")


def _read_ortho_mmap(d: Path, m_per_px: float) -> dict | None:
    """The bake, memory-mapped. ``None`` when the cache is absent or partial.

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
        bake: dict = {
            k: np.load(d / f"{k}.npy", mmap_mode="r") for k in _MMAP_ARRAYS
        }
    except (OSError, ValueError, KeyError):
        return None
    bake.update(
        x_min=float(meta["x_min"]),
        y_min=float(meta["y_min"]),
        px_per_m=float(meta["px_per_m"]),
        m_per_px=float(m_per_px),
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


def _write_ortho_mmap(d: Path, bake: dict) -> None:
    """Publish an in-heap ``bake`` so the next process maps it instead of building it.

    Best-effort. A read-only or full filesystem costs the build again, never
    the run.
    """
    tmp = d.with_name(f"{d.name}.tmp{os.getpid()}")
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        for k in _MMAP_ARRAYS:
            np.save(tmp / f"{k}.npy", np.ascontiguousarray(bake[k]))
        (tmp / "meta.json").write_text(
            json.dumps({k: float(bake[k])
                        for k in ("x_min", "y_min", "px_per_m")})
        )
        _publish_mmap(tmp, d)
    except OSError as exc:
        log.warning("ortho mmap cache write failed at %s: %s", d, exc)
        shutil.rmtree(tmp, ignore_errors=True)


def _geotiff_pair(rgbd_dir, map_name: str) -> Path | None:
    """The block's georeferenced RGB raster, when the rasters are that kind."""
    path = Path(rgbd_dir) / f"{map_name}_rgb.tif"
    return path if path.is_file() else None


def _bake_geotiff_mmap(rgbd_dir, map_name: str, d: Path, band_rows: int = 1024) -> None:
    """Stream ``<map>_rgb.tif`` + ``<map>.tif`` into the cache at ``d``.

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
        (tmp / "meta.json").write_text(json.dumps(meta))
        _publish_mmap(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _read_ortho_npz(path, m_per_px: float) -> dict:
    """Decode an ortho npz into the canonical bake dict."""
    with np.load(path) as data:
        return {
            "rgb": data["rgb"].astype(np.uint8),
            "height": data["height"].astype(np.float32),
            "alpha": data["alpha"].astype(np.uint8),
            "x_min": float(data["x_min"]),
            "y_min": float(data["y_min"]),
            "px_per_m": float(data["px_per_m"]),
            "m_per_px": float(m_per_px),
        }


def bake_ortho_from_ply(
    ply_path,
    *,
    m_per_px: float = 0.10,
    chunk: int = 5_000_000,
    fill: bool = True,
) -> dict:
    """Top-down orthographic mosaic of the PLY at ``m_per_px`` GSD.

    For each output cell keep the highest-Z PLY point and record its RGB
    + height. Empty cells (water, glass, parking lots, scan boundaries) get
    ``alpha=0``. Measured empty rates at 0.10 m/px: cambridge_block_21 5.35%,
    birmingham_block_1 25.56% — but most of that is the block's exterior frame
    plus large true no-data; the *interior* speckle (photogrammetry gaps under
    canopy) is the part that hurts perception.

    With ``fill=True`` (the default, and the single hole-free 2D source of
    truth), :func:`ortho_fill.fill_ortho_holes` fills the interior holes
    (enclosed by coverage) in ``rgb``/``height`` and marks them ``alpha=255``;
    boundary no-data (frame/water/scan edge) stays black so no fill projects to
    a frame edge. Pass ``fill=False`` for the raw single-cell bake (debugging).

    Streams in chunks so peak memory stays bounded at ~``chunk * 12`` bytes.
    Returns a dict with the cached arrays plus the world bounds.
    """
    pc = _load_ply_cloud(str(ply_path))
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = np.clip(np.asarray(pc.colors, dtype=np.float32) * 255.0, 0, 255).astype(
        np.uint8
    )

    x_min = float(xyz[:, 0].min())
    y_min = float(xyz[:, 1].min())
    x_max = float(xyz[:, 0].max())
    y_max = float(xyz[:, 1].max())
    px_per_m = 1.0 / float(m_per_px)
    W = int(np.ceil((x_max - x_min) * px_per_m)) + 1
    H = int(np.ceil((y_max - y_min) * px_per_m)) + 1

    out_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    out_h = np.full((H, W), -np.inf, dtype=np.float32)
    out_a = np.zeros((H, W), dtype=np.uint8)

    n = xyz.shape[0]
    # Streaming top-Z-wins fill. ``np.maximum.at`` handles duplicate cells
    # correctly within a chunk; the color pass writes whichever point had
    # the now-current max Z. With chunked iteration this stays correct
    # because we stage h_pass first, then commit color for points whose Z
    # equals the current cell maximum.
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cols = np.clip(
            ((xyz[start:end, 0] - x_min) * px_per_m).astype(np.int32), 0, W - 1
        )
        rows = np.clip(
            (H - 1 - (xyz[start:end, 1] - y_min) * px_per_m).astype(np.int32),
            0,
            H - 1,
        )
        zs = xyz[start:end, 2]
        np.maximum.at(out_h, (rows, cols), zs)
        # Color pass: a point wins at its cell iff its Z equals the cell's
        # current max. Late-arriving ties just overwrite; chunk order is
        # arbitrary and the mosaic is photometrically smooth anyway.
        wins = zs == out_h[rows, cols]
        if wins.any():
            out_rgb[rows[wins], cols[wins]] = rgb[start:end][wins]
            out_a[rows[wins], cols[wins]] = 255

    # Interior holes (enclosed by coverage) are filled so this ortho is the
    # single hole-free 2D source; boundary no-data (frame/water/scan edge) stays
    # black. Fill BEFORE the height sentinel-clear so filled interior cells take
    # a real nearest height, not the 0.0 no-data sentinel.
    if fill:
        from maps.perception.ortho.ortho_fill import fill_ortho_holes

        out_rgb, out_h, out_a = fill_ortho_holes(out_rgb, out_h, out_a)
    # Remaining no-data (boundary) keeps alpha=0; sentinel-clear its height.
    out_h[out_a == 0] = 0.0

    return {
        "rgb": out_rgb,
        "height": out_h,
        "alpha": out_a,
        "x_min": float(x_min),
        "y_min": float(y_min),
        "px_per_m": float(px_per_m),
        "m_per_px": float(m_per_px),
    }


def save_ortho(
    ply_path,
    bake: dict,
    *,
    m_per_px: float | None = None,
) -> Path:
    """Persist a ``bake_ortho_from_ply`` result as a compressed npz."""
    res = float(m_per_px if m_per_px is not None else bake["m_per_px"])
    out = _ortho_path(ply_path, res)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rgb=bake["rgb"],
        height=bake["height"],
        alpha=bake["alpha"],
        x_min=np.float64(bake["x_min"]),
        y_min=np.float64(bake["y_min"]),
        px_per_m=np.float64(bake["px_per_m"]),
    )
    return out


_ORTHO_CACHE: dict[tuple[str, float], dict] = {}

# A flight crops one map, so two entries cover it. An audit browses across
# maps, and a miss costs a full rebake — 4.3 s measured on the 0.1 m/px
# rasters — so the render daemon raises this via ``WORLDS_ORTHO_CACHE_BLOCKS``.
# Entry size scales with the map: rgb u8 + height f32 + alpha u8 runs
# ~130 MB for a SensatUrban block and ~3.6 GB for the ~460 MP unified
# calcity map (which only ever occupies one slot).
_ORTHO_CACHE_MAX_ENTRIES = max(
    1, int(os.environ.get("WORLDS_ORTHO_CACHE_BLOCKS", "2"))
)


def load_ortho(ply_path, m_per_px: float = 0.10) -> dict | None:
    """Return the cached orthomosaic dict for ``ply_path`` or ``None`` if absent.

    Memoises decoded npzs at module scope so subsequent crops reuse the
    arrays without re-reading from disk. The first call costs the npz
    decompress (~1-2 s for a 400×400 m block at 0.10 m/px).
    """
    key = (str(ply_path), float(m_per_px))
    cached = _ORTHO_CACHE.get(key)
    if cached is not None:
        return cached

    path = _ortho_path(ply_path, m_per_px)
    if not path.exists():
        return None
    with np.load(path) as data:
        bake = {
            "rgb": data["rgb"].astype(np.uint8),
            "height": data["height"].astype(np.float32),
            "alpha": data["alpha"].astype(np.uint8),
            "x_min": float(data["x_min"]),
            "y_min": float(data["y_min"]),
            "px_per_m": float(data["px_per_m"]),
            "m_per_px": float(m_per_px),
        }

    if len(_ORTHO_CACHE) >= _ORTHO_CACHE_MAX_ENTRIES:
        _ORTHO_CACHE.pop(next(iter(_ORTHO_CACHE)))
    _ORTHO_CACHE[key] = bake
    return bake


def bake_ortho_from_rasters(
    rgbd_dir,
    map_name: str,
    *,
    m_per_px: float = 0.10,
    fill: bool = True,
) -> dict:
    """Build the canonical ortho dict from a co-registered 0.1 m/px RGB PNG +
    height GeoTIFF (``<map>.png`` + ``<map>.tif`` under ``rgbd_dir``).

    The alternative source to :func:`bake_ortho_from_ply` for blocks/sites that
    ship rasters (the CityNav ``rgbd_upstream`` bake, the fieldsite DSM ortho)
    rather than a PLY — same output contract ``{rgb, height, alpha, x_min,
    y_min, px_per_m}`` in the world frame (x=east, y=north; row 0 = y_max). The
    GeoTIFF affine supplies the world extent, so it co-registers with
    ``GROUND_LEVEL``/view-square geometry exactly as the vendored cropclient did.
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
    if fill:
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
        "m_per_px": float(m_per_px),
    }


_ORTHO_RASTER_CACHE: dict[tuple, dict] = {}
_ORTHO_RASTER_BUILD_LOCK = threading.Lock()


def load_ortho_or_rasters(
    map_name: str,
    *,
    ply_path=None,
    rgbd_dir=None,
    m_per_px: float = 0.10,
) -> dict | None:
    """Return the canonical filled ortho dict for ``map_name`` from whichever
    source is on disk — the baked npz if present, else built + filled from the
    0.1 m/px rasters. This is the ONE 2D source-of-truth loader shared by the
    observation, the ``look`` tool, and the fieldsite paths (PLY-baked npz vs
    GeoTIFF rasters). Cached (LRU) so repeated crops of a block are cheap.
    """
    if ply_path is not None:
        bake = load_ortho(ply_path, m_per_px)
        if bake is not None:
            return bake
    if rgbd_dir is None:
        return None
    key = (str(map_name), str(rgbd_dir), float(m_per_px))
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
        # MAP IT IF SOMEONE ALREADY BUILT IT. Building this from rasters is ~20 s
        # for the unified Cal City map and every process pays it: measured on a
        # 48-worker arm, 47 cold builds cost 936 s, more than the 435 warm stops
        # in the same window put together. The result is a pure function of the
        # rasters, so the first process to build it publishes it and the rest map
        # it -- which also collapses N heap copies of a ~3.6 GB map into one
        # page-cache copy the whole node shares.
        mm = _mmap_dir(rgbd_dir, map_name, m_per_px)
        bake = _read_ortho_mmap(mm, m_per_px)
        if bake is None:
            # Prefer a prerendered npz next to the rasters
            # (benchmarks.citynav.bake.ortho writes one per block); fall back to
            # building + filling on the fly.
            npz = _rgbd_ortho_path(rgbd_dir, map_name, m_per_px)
            if npz.exists():
                bake = _read_ortho_npz(npz, m_per_px)
                _write_ortho_mmap(mm, bake)
            elif _geotiff_pair(rgbd_dir, map_name) is not None:
                # Gigapixels: never in the heap. Streamed into the cache and
                # mapped back like any other process would map it.
                _bake_geotiff_mmap(rgbd_dir, map_name, mm)
                bake = _read_ortho_mmap(mm, m_per_px)
                if bake is None:
                    raise RuntimeError(f"{map_name}: the ortho cache at {mm} was written but does not read back")
            else:
                bake = bake_ortho_from_rasters(
                    rgbd_dir, map_name, m_per_px=m_per_px, fill=True
                )
                _write_ortho_mmap(mm, bake)
        if len(_ORTHO_RASTER_CACHE) >= _ORTHO_CACHE_MAX_ENTRIES:
            _ORTHO_RASTER_CACHE.pop(next(iter(_ORTHO_RASTER_CACHE)))
        _ORTHO_RASTER_CACHE[key] = bake
        return bake


def ortho_world_bounds(
    map_name: str,
    *,
    rgbd_dir=None,
    ply_path=None,
    m_per_px: float = 0.10,
) -> tuple[float, float, float, float]:
    """World extent ``(x_min, y_min, x_max, y_max)`` of the canonical ortho.

    Used by the ``look`` tool to shift a near-edge crop window in-bounds. Reads
    the same source as :func:`load_ortho_or_rasters` (baked npz or rasters).
    """
    ortho = load_ortho_or_rasters(
        map_name, ply_path=ply_path, rgbd_dir=rgbd_dir, m_per_px=m_per_px
    )
    if ortho is None:
        raise FileNotFoundError(
            f"No ortho for {map_name!r} (pass rgbd_dir or ply_path)."
        )
    x_min, y_min, ppm = ortho["x_min"], ortho["y_min"], ortho["px_per_m"]
    h, w = ortho["rgb"].shape[:2]
    return (float(x_min), float(y_min), float(x_min + w / ppm), float(y_min + h / ppm))


def ortho_gsd_m(
    map_name: str,
    *,
    rgbd_dir=None,
    ply_path=None,
    m_per_px: float = 0.10,
) -> float:
    """Metres per pixel of the canonical ortho GRID for ``map_name``.

    The divisor that turns a window in metres into delivered pixels, so a
    caller sizing a crop against a pixel budget reads the map's real grid
    instead of assuming one. Not the camera's resolution: CityNav's grid is
    0.1 m/px and Cal City's is the block raster's own cell, so this says how
    many PIXELS a span costs, never how much DETAIL it holds.
    """
    ortho = load_ortho_or_rasters(
        map_name, ply_path=ply_path, rgbd_dir=rgbd_dir, m_per_px=m_per_px
    )
    if ortho is None:
        raise FileNotFoundError(
            f"No ortho for {map_name!r} (pass rgbd_dir or ply_path)."
        )
    return 1.0 / float(ortho["px_per_m"])
