"""Cal City in the layout a released CityNav checkpoint reads.

Every released checkpoint was trained on SensatUrban blocks of at most about
400 m a side, rasterized at 0.1 m/px, and addresses its world through
``gsamllavanav``'s ``MAP_BOUNDS`` / ``GROUND_LEVEL`` dicts keyed by block
name. Cal City is one 13.4 x 10.2 km map at 0.15 m/px, so each episode gets
its own synthetic block: a square window that holds both the start and the
target, cut from the flown ortho and resampled to 0.1 m/px.

Per episode, under ``<out>/shards/shard_<k>/``:

  ``data/rgbd/<block>.png``   RGB, 0.1 m/px, north-up
  ``data/rgbd/<block>.tif``   flat height plane (ground level 0) on the same
                              grid; its affine is the block's local-frame
                              origin, the georef every crop reads
  ``data/citynav/citynav_test_unseen.json``  the shard's episodes, CityNav
                              MTurk rows, the dataset start as trajectory[0]
  ``data/cityrefer/objects.json``  the shard's blocks: landmarks clipped to
                              each window plus one unnamed target object

and at ``<out>``: ``blocks.json`` (the mapdata entries the launcher injects),
``manifest.json`` (block -> episode, every exclusion with its reason, the
export parameters). ``processed_descriptions.json`` is written there by
:mod:`describe_instructions`; each shard links to it.

The window is placed uniformly at random (seeded per episode) among the
positions that contain both points with a margin, so neither the window's
centre nor its edges point at the target. Coordinates stay in Cal City's
local metric frame (x east, y north), so an outcome scores against the
corpus with no transform.

Usage::

    python -m baselines.released.calcity_export --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

#: Window side. FlightGPT reads the whole block as one image under a
#: max_pixels of 16,032,016 (4004 px square at 0.1 m/px); HETT and MGP
#: normalize goals over a 410 m canvas. 400 m fits all three.
BLOCK_SIDE_M = 400.0
#: Clearance between the window edge and the start or the target shape, so
#: neither sits on the edge of the imagery a checkpoint sees.
MARGIN_M = 20.0
RESOLUTION_M = 0.1
#: The released checkpoints all fly 50 m above ground; ``fix_altitude``
#: rewrites z to ground + 50 on load, so this is only what the file says.
ALTITUDE_M = 50.0
#: Cal City rows carry no heading. North, the direction every frame the
#: harness reads is turned to.
START_YAW_RAD = math.pi / 2
START_PITCH_RAD = -math.pi / 2
#: The synthesized target object's id inside its block. Outside the Cal City
#: catalog's id range so it can never shadow a landmark.
TARGET_OBJECT_ID = 9_999_999
SHARD_SIZE = 50
SPLIT_NAME = "test_unseen"

#: The fields ``gsamllavanav.cityreferobject.CityReferObject`` accepts. The
#: Cal City catalog carries a ``tags`` dict and no ``map_name``; passing
#: either through would fail the dataclass constructor.
CITYREFER_FIELDS = (
    "map_name", "id", "name", "object_type", "position", "dimension",
    "descriptions", "contour",
)


def block_name(row_index: int | str) -> str:
    """``calcity_block_<i>``: what ``MTurkTrajectory.map_name`` builds from
    ``area="calcity"``, ``block=<i>``."""
    return f"calcity_block_{row_index}"


def place_block(
    points: list[tuple[float, float]],
    imagery: tuple[float, float, float, float],
    rng: random.Random,
    side_m: float = BLOCK_SIDE_M,
) -> tuple[float, float] | str:
    """The window's ``(x_min, y_min)``, or the reason no window fits.

    ``points`` are the start and every vertex of the target shape. The
    window holds them all ``MARGIN_M`` from its edge and lies inside the
    imagery ``(x_min, y_min, x_max, y_max)``.
    """
    origin = []
    for axis in (0, 1):
        lo_pt = min(p[axis] for p in points)
        hi_pt = max(p[axis] for p in points)
        if hi_pt - lo_pt + 2 * MARGIN_M > side_m:
            return "start_and_target_exceed_block"
        lo = max(hi_pt + MARGIN_M - side_m, imagery[axis])
        hi = min(lo_pt - MARGIN_M, imagery[axis + 2] - side_m)
        if hi < lo:
            return "window_leaves_imagery"
        origin.append(rng.uniform(lo, hi))
    return origin[0], origin[1]


def mturk_row(
    row_index: int,
    description: str,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> dict:
    """One CityNav MTurk row a released loader accepts.

    The marker is the target itself (no human placed one), so the loader's
    marker-to-target filter keeps the row and its straight-line teacher path
    ends on the target. The teacher path feeds HETT's SPL denominator and
    nothing any checkpoint reads at decision time.
    """
    sx, sy = start_xy
    tx, ty = target_xy
    return {
        "area": "calcity",
        "block": str(row_index),
        "object_ids": [TARGET_OBJECT_ID],
        "ann_ids": [0],
        "descriptions": [description],
        "trajectory": [[sx, sy, ALTITUDE_M, START_YAW_RAD, START_PITCH_RAD]],
        "marker_positions": [[tx, ty, 0.0]],
        "target_positions": [[tx, ty, 0.0]],
        "total_score": 0.0,
        "dist_marker_to_target": 0.0,
        "split": SPLIT_NAME,
    }


def target_object(
    block: str,
    description: str,
    target_xy: tuple[float, float],
    polygon: list | None,
) -> dict:
    """The episode's target as a CityRefer object.

    Unnamed, so ``filter_landmarks`` never lists it as a landmark: a
    checkpoint learns of it only through the description.
    """
    tx, ty = target_xy
    contour = [[float(x), float(y)] for x, y in polygon] if polygon else [
        [tx - 1, ty - 1], [tx + 1, ty - 1], [tx + 1, ty + 1], [tx - 1, ty + 1],
    ]
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    return {
        "map_name": block,
        "id": TARGET_OBJECT_ID,
        "name": "",
        "object_type": "Unknown",
        "position": [tx, ty, 0.0],
        "dimension": [max(xs) - min(xs), max(ys) - min(ys), 0.0],
        "descriptions": [description],
        "contour": contour,
    }


def landmark_index(catalog: dict):
    """``(objects, polygons, tree)``: the catalog's footprints under an STR tree,
    built once per export rather than intersected against every window."""
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    objs = [o for o in catalog.values() if len(o.get("contour") or []) >= 3]
    polys = [Polygon(o["contour"]).buffer(0) for o in objs]
    return objs, polys, STRtree(polys)


def clip_landmarks(index, block: str, window: tuple[float, float, float, float]) -> list[dict]:
    """Catalog objects clipped to the window, as CityRefer objects of ``block``.

    A road runs kilometres past any window, and its centroid can sit outside
    the block while the road crosses it, so selection is by intersection
    rather than by centre. ``index`` is :func:`landmark_index`'s.
    """
    from shapely.geometry import box

    objs, polys, tree = index
    frame = box(*window)
    out = []
    for i in sorted(tree.query(frame)):
        obj = objs[i]
        piece = polys[i].intersection(frame)
        if piece.is_empty:
            continue
        if piece.geom_type != "Polygon":
            # ponytail: the largest part only, a multi-part clip of one
            # object becomes one landmark; split parts if a block needs them.
            piece = max(
                (g for g in getattr(piece, "geoms", []) if g.geom_type == "Polygon"),
                key=lambda g: g.area,
                default=None,
            )
            if piece is None:
                continue
        x0, y0, x1, y1 = piece.bounds
        z = float((obj.get("position") or [0, 0, 0])[2])
        dz = float((obj.get("dimension") or [0, 0, 0])[2])
        clipped = {k: obj.get(k) for k in CITYREFER_FIELDS}
        clipped.update(
            map_name=block,
            position=[piece.centroid.x, piece.centroid.y, z],
            dimension=[x1 - x0, y1 - y0, dz],
            contour=[[x, y] for x, y in list(piece.exterior.coords)[:-1]],
            descriptions=list(obj.get("descriptions") or []),
        )
        out.append(clipped)
    return out


def write_block_rasters(
    src,
    utm_origin: tuple[float, float],
    window: tuple[float, float, float, float],
    png_path: Path,
    tif_path: Path,
) -> None:
    """Cut ``window`` (local frame) out of the UTM ortho ``src`` at
    ``RESOLUTION_M``; write the PNG and the flat height GeoTIFF on one grid.

    A windowed, resampled read: the full ortho never enters memory.
    Imagery the window runs past reads as black (the export only places
    windows inside the imagery, so that is a boundary pixel at most).
    """
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.windows import from_bounds

    x_min, y_min, x_max, y_max = window
    oe, on = utm_origin
    size = round((x_max - x_min) / RESOLUTION_M)
    rgb = src.read(
        [1, 2, 3],
        window=from_bounds(x_min + oe, y_min + on, x_max + oe, y_max + on,
                           transform=src.transform),
        out_shape=(3, size, size),
        resampling=Resampling.bilinear,
        boundless=True,
        fill_value=0,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 2, 0))).save(png_path)
    with rasterio.open(
        tif_path, "w", driver="GTiff", width=size, height=size, count=1,
        dtype="float32", transform=from_origin(x_min, y_max, RESOLUTION_M, RESOLUTION_M),
        compress="deflate",
    ) as dst:
        dst.write(np.zeros((1, size, size), dtype="float32"))


def export(
    shapes: dict[str, dict],
    catalog: dict,
    out: Path,
    *,
    ortho_path: Path | None,
    utm_origin: tuple[float, float],
    imagery: tuple[float, float, float, float],
    seed: int = 0,
    side_m: float = BLOCK_SIDE_M,
    shard_size: int = SHARD_SIZE,
) -> dict:
    """Write the export under ``out``; return the manifest.

    ``shapes`` is :func:`calcity_shapes`'s
    ``{episode_id: {gt_polygon, target_xy, start_xy, description}}`` in split
    order, so the window holds the same shape the outcome is later scored
    against. ``ortho_path=None`` writes everything but the rasters (a dry run).
    """
    blocks: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for index, (episode_id, shape) in enumerate(shapes.items()):
        polygon = shape.get("gt_polygon")
        target_xy = tuple(shape["target_xy"])
        points = [shape["start_xy"], target_xy] + [tuple(p) for p in polygon or []]
        rng = random.Random(f"{seed}:{episode_id}")
        placed = place_block(points, imagery, rng, side_m)
        if isinstance(placed, str):
            excluded[episode_id] = placed
            continue
        x0, y0 = placed
        blocks[block_name(index)] = {
            "episode_id": episode_id,
            "row_index": index,
            "bounds": [x0, y0, x0 + side_m, y0 + side_m],
            "description": shape["description"],
            "start_xy": list(shape["start_xy"]),
            "target_xy": list(target_xy),
            "polygon": polygon,
        }

    names = list(blocks)
    index = landmark_index(catalog)
    for k in range(0, len(names), shard_size):
        shard = f"shard_{k // shard_size:03d}"
        root = out / "shards" / shard / "data"
        episodes, objects = [], {}
        for name in names[k:k + shard_size]:
            b = blocks[name]
            b["shard"] = shard
            window = tuple(b["bounds"])
            landmarks = clip_landmarks(index, name, window)
            b["landmarks"] = sorted({o["name"] for o in landmarks if o.get("name")})
            target = target_object(name, b["description"], tuple(b["target_xy"]), b["polygon"])
            objects[name] = {str(o["id"]): o for o in landmarks + [target]}
            episodes.append(mturk_row(b["row_index"], b["description"],
                                      tuple(b["start_xy"]), tuple(b["target_xy"])))
        write_json(root / "citynav" / f"citynav_{SPLIT_NAME}.json", episodes)
        write_json(root / "cityrefer" / "objects.json", objects)
        link = root / "cityrefer" / "processed_descriptions.json"
        if not link.is_symlink():
            link.symlink_to("../../../../processed_descriptions.json")

    if ortho_path is not None:
        import rasterio

        with rasterio.open(ortho_path) as src:
            for name, b in blocks.items():
                rgbd = out / "shards" / b["shard"] / "data" / "rgbd"
                png, tif = rgbd / f"{name}.png", rgbd / f"{name}.tif"
                if png.exists() and tif.exists():
                    continue
                write_block_rasters(src, utm_origin, tuple(b["bounds"]), png, tif)

    write_json(out / "blocks.json", {
        "GROUND_LEVEL": {name: 0.0 for name in blocks},
        "MAP_BOUNDS": {name: b["bounds"] for name, b in blocks.items()},
    })
    manifest = {
        "params": {
            "block_side_m": side_m, "margin_m": MARGIN_M,
            "resolution_m": RESOLUTION_M, "seed": seed,
            "shard_size": shard_size, "target_object_id": TARGET_OBJECT_ID,
            "ortho": str(ortho_path) if ortho_path else None,
            "utm_origin": list(utm_origin), "imagery": list(imagery),
        },
        "n_rows": len(shapes),
        "blocks": {
            name: {k: b[k] for k in ("episode_id", "row_index", "bounds", "shard", "landmarks")}
            for name, b in blocks.items()
        },
        "excluded": excluded,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def calcity_shapes() -> dict[str, dict]:
    """``{episode_id: {gt_polygon, target_xy, start_xy, description}}`` over
    the Cal City split, in split order: the shape the bench scores each row
    against (``episode_runner.success_shape``), its target point, its start
    and the episode's instruction. Selects Cal City for this process first."""
    import benchmarks.calcity

    benchmarks.calcity.use()
    from benchmarks.calcity import paths
    from benchmarks.citynav.adapter import load_objects, load_split, parse_episode
    from benchmarks.citynav.run_benchmark.episode_runner import success_shape

    objects = load_objects(paths.OBJECTS_JSON)
    out: dict[str, dict] = {}
    for index, row in enumerate(load_split(paths.split_path("calcity"))):
        episode = parse_episode(row, index)
        shape = success_shape(episode, objects)
        out[episode.episode_id] = {
            "gt_polygon": [list(map(float, p)) for p in shape] if shape is not None else None,
            "target_xy": episode.target_xy,
            "start_xy": episode.start_xy,
            "description": episode.description,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    shapes = calcity_shapes()
    from benchmarks.calcity import paths

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ortho", type=Path,
                    default=paths.CALCITY_NATIVE_RGBD_DIR / "calcity_rgb.tif",
                    help="the UTM ortho the blocks are cut from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--dry-run", action="store_true",
                    help="write episodes, objects and manifest, no rasters")
    args = ap.parse_args(argv)

    georef = paths.bake_meta()["georef"]
    utm_origin = (float(georef["origin_e"]), float(georef["origin_n"]))
    import rasterio

    with rasterio.open(args.ortho) as src:
        b = src.bounds
        imagery = (b.left - utm_origin[0], b.bottom - utm_origin[1],
                   b.right - utm_origin[0], b.top - utm_origin[1])
    catalog = json.loads(paths.OBJECTS_JSON.read_text())["calcity"]
    manifest = export(
        shapes, catalog, args.out,
        ortho_path=None if args.dry_run else args.ortho,
        utm_origin=utm_origin, imagery=imagery, seed=args.seed,
        shard_size=args.shard_size,
    )
    print(f"{len(manifest['blocks'])} blocks, {len(manifest['excluded'])} excluded "
          f"of {manifest['n_rows']} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
