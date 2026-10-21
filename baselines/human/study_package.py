"""The capacity study's download, built once from the bench's own files (``STUDY.md``).

A participant's machine needs no reasoner, no segmenter and no 46 GB ortho: only the
pictures the console shows and the records MAPS's results are read from. This writes:

  ``calcity_ortho.mbtiles``  the bench's ortho (``<map>_rgb.tif``) as web-mercator WebP
                             tiles, zoom 12 to 20, no-data transparent
  ``maps_root/``             the MAPS root the plan replays: its ledger and timing, and per
                             attempt the records a verify screen reads (the turns, the
                             detections, the stops, the frames)
  ``tiles/``                 the bench's OSM prior bake (``<map>_meta.json``, ``<map>_features.json``)
  ``data/cityrefer/objects.json``  the labelled objects
  ``plan/``                  the participants' lists, dealt by ``capacity plan``, and the
                             practice reports

    python -m baselines.human.study_package tiles --raster <map>_rgb.tif --out <dir>/calcity_ortho.mbtiles
    python -m baselines.human.study_package records --maps-root <root> --study <plan dir> --out <dir>
"""

from __future__ import annotations

import argparse
import io
import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

#: Web mercator's half-width, metres.
MERC_O = 20037508.342789244
#: Zoom 12 frames the whole block; 20 is 0.12 m a pixel at Cal City, finer than the
#: bench's 0.15 m, so the closest zoom shows every pixel the bench has.
ZOOMS = range(12, 21)
TILE_PX = 256
#: The attempt files a verify screen reads (``audit_console.blind_review``), and its frames.
ATTEMPT_FILES = ("conversation.jsonl", "detections.jsonl", "flight/tour_cursor.json")

#: The open raster per overview level, one set per worker process.
_SRC: dict = {}


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    s = 2 * MERC_O / 2**z
    return -MERC_O + x * s, MERC_O - (y + 1) * s, -MERC_O + (x + 1) * s, MERC_O - y * s


def _render(job: tuple[str, int, int, int]) -> tuple[int, int, int, bytes | None]:
    """One tile, warped from the raster's coarsest overview still as fine as the tile (a
    warp off the full raster reads all of it for a zoomed-out tile). Black is no data in
    the bench's raster, so it is cut clear."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    raster, z, x, y = job
    if not _SRC:
        _SRC[None] = rasterio.open(raster)
    base = _SRC[None]
    # A tile's ground metres a pixel: mercator's, shrunk by the cosine of its latitude.
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / 2**z)))
    tile_m = 2 * MERC_O / 2**z / TILE_PX * np.cos(lat)
    level = None
    for i, f in enumerate(base.overviews(1)):
        if base.res[0] * f <= tile_m:
            level = i
    if level not in _SRC:
        _SRC[level] = rasterio.open(raster, overview_level=level)
    a = np.zeros((3, TILE_PX, TILE_PX), np.uint8)
    reproject(rasterio.band(_SRC[level], (1, 2, 3)), a, dst_transform=from_bounds(*_tile_bounds(z, x, y), TILE_PX, TILE_PX),
              dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    rgb = a.transpose(1, 2, 0)
    alpha = np.where(rgb.max(axis=2) == 0, 0, 255).astype(np.uint8)
    if not alpha.any():
        return z, x, y, None
    buf = io.BytesIO()
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(buf, format="WEBP", quality=80)
    return z, x, y, buf.getvalue()


def build_tiles(raster: Path, out: Path, workers: int = 8) -> int:
    """Every tile over the raster at every zoom in ``ZOOMS``, into one MBTiles file."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster) as src:
        l, b, r, t = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    jobs = []
    for z in ZOOMS:
        step = 2 * MERC_O / 2**z
        for x in range(int((l + MERC_O) // step), int((r + MERC_O) // step) + 1):
            for y in range(int((MERC_O - t) // step), int((MERC_O - b) // step) + 1):
                jobs.append((str(raster), z, x, y))
    tmp = out.with_suffix(".partial")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("CREATE TABLE metadata (name TEXT, value TEXT);"
                     "CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB);")
    meta = {"name": raster.stem, "format": "webp", "type": "baselayer", "minzoom": min(ZOOMS), "maxzoom": max(ZOOMS),
            "bounds": f"{w},{s},{e},{n}", "center": f"{(w + e) / 2},{(s + n) / 2},{min(ZOOMS) + 1}"}
    db.executemany("INSERT INTO metadata VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    n_tiles = 0
    with ProcessPoolExecutor(workers) as pool:
        for i, (z, x, y, data) in enumerate(pool.map(_render, jobs, chunksize=64)):
            if data is not None:
                # MBTiles rows count from the south (TMS).
                db.execute("INSERT INTO tiles VALUES (?, ?, ?, ?)", (z, x, 2**z - 1 - y, data))
                n_tiles += 1
            if i % 5000 == 0:
                db.commit()
                print(f"tiles: {i}/{len(jobs)} ({n_tiles} kept)", flush=True)
    db.execute("CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row)")
    db.commit()
    db.close()
    tmp.replace(out)
    return n_tiles


def copy_records(maps_root: Path, out: Path) -> int:
    """The MAPS root's ledger, timing and manifest, and per attempt what a verify screen reads."""
    from baselines.human.session_plan import scored_rows

    out.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "batch_timing.json", "run_manifest.json"):
        if (maps_root / name).is_file():
            shutil.copy2(maps_root / name, out / name)
    rows = scored_rows(maps_root)
    for row in rows.values():
        src, dst = maps_root / row["workdir"], out / row["workdir"]
        for rel in ATTEMPT_FILES:
            if (src / rel).is_file():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, dst / rel)
        if (src / "frames").is_dir():
            shutil.copytree(src / "frames", dst / "frames", dirs_exist_ok=True)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiles", help="the ortho as one MBTiles file")
    t.add_argument("--raster", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("records", help="the MAPS root, the prior, the labels and the plan, as the download lays them out")
    r.add_argument("--maps-root", type=Path, required=True)
    r.add_argument("--study", type=Path, required=True, help="the plan's folder (capacity plan and practice wrote it)")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "tiles":
        print(f"{build_tiles(args.raster, args.out, args.workers)} tiles -> {args.out}")
        return 0
    from benchmarks.citynav.paths import OBJECTS_JSON, TILES_DIR

    print(f"{copy_records(args.maps_root, args.out / 'maps_root')} attempts -> {args.out / 'maps_root'}")
    (args.out / "tiles").mkdir(parents=True, exist_ok=True)
    for f in Path(TILES_DIR).glob("*_meta.json"):
        shutil.copy2(f, args.out / "tiles" / f.name)
        feats = f.with_name(f.name.replace("_meta.json", "_features.json"))
        if feats.is_file():
            shutil.copy2(feats, args.out / "tiles" / feats.name)
    (args.out / "data/cityrefer").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OBJECTS_JSON, args.out / "data/cityrefer/objects.json")
    # The lists and the practice root, never a session a pilot already wrote.
    shutil.copytree(args.study, args.out / "plan", dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names if n.startswith(".")
                                             or (Path(d).parent == args.study and Path(d).name != "practice"
                                                 and n in ("solo", "team", "practice"))])
    print(f"prior, labels and plan -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
