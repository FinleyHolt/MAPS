#!/usr/bin/env python3
"""Build a CityRefer-schema ``objects.json`` from the world bundle's OSM.

Each OSM footprint becomes one CityRefer object: ``id``, ``position`` (the
footprint centroid at ``aoi.GROUND_LEVEL_M``, the world being flat),
``dimension`` (the footprint's x and y extent) and ``contour``. Geometry is
reprojected from the bundle's WGS84 GeoJSON into the map's local metric frame
and overlapping footprints are deduped; the nested ``{map_name: {id: {...}}}``
dict that ``benchmarks.citynav.adapter.load_objects`` reads is written for the
one map, ``calcity``.

Usage::

    python -m benchmarks.calcity.data_prep.build_objects_json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.calcity.paths import calcity_bench
from benchmarks.calcity.data_prep.aoi import (
    AOI_BBOX,
    GROUND_LEVEL_M,
    build_local_frame,
    bundle_root,
    classify_object_type,
    tag_value,
)
from benchmarks.calcity.data_prep.bundle import load_features_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_objects_json")

# Structures win when the same footprint is tagged multiple ways.
_TYPE_RANK = {"Building": 4, "Water": 3, "Ground": 2, "Parking": 1}
_CONTOUR_SIMPLIFY_M = 1.0
_CONTOUR_MAX_PTS = 300


def _largest_polygon(geom):
    """Return the largest single Polygon from any geometry (or None).

    LineStrings (roads, waterways) are buffered by a nominal half-width so
    they rasterize and target like every other footprint; Points (POI nodes)
    get a small disc.
    """
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type in ("LineString", "MultiLineString"):
        buffered = geom.buffer(4.0, cap_style="flat")
        return _largest_polygon(buffered) if buffered.geom_type != "Polygon" else buffered
    if geom.geom_type == "Point":
        return geom.buffer(6.0)
    return None


def _contour_xy(poly) -> list[list[float]]:
    """Exterior ring as ``[[x, y], ...]`` local meters, simplified + capped."""
    simplified = poly.simplify(_CONTOUR_SIMPLIFY_M, preserve_topology=True)
    ring = simplified.exterior if simplified.geom_type == "Polygon" else poly.exterior
    coords = list(ring.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) > _CONTOUR_MAX_PTS:
        step = int(np.ceil(len(coords) / _CONTOUR_MAX_PTS))
        coords = coords[::step]
    return [[float(x), float(y)] for x, y in coords]


def build_block(bundle: Path, map_name: str, bbox) -> dict:
    """Build one block's ``{id: object}`` dict."""
    lf = build_local_frame(bbox)
    records = load_features_local(bundle, lf)
    log.info("%s: %d bundle features in window", map_name, len(records))

    # --- classify + dedup overlapping footprints ---
    best: dict[tuple, dict] = {}
    for rec in records:
        otype = classify_object_type(rec["tags"])
        if otype is None:
            continue
        poly = _largest_polygon(rec["geom"])
        if poly is None or poly.is_empty:
            continue
        name = tag_value(rec["tags"], "name")
        cx, cy = float(poly.centroid.x), float(poly.centroid.y)
        # Both keys carry the centroid bucket, so dedup only ever collapses
        # things in the same PLACE. A bare ("named", name) key would compete
        # every way of a street for one slot: OSM splits a road at each tag
        # change, so California City Boulevard would bake as one 908 m
        # fragment. One row per way is what CityNav bakes and what
        # the reasoner surface is written against -- "one road is routinely
        # several ways, pass them all".
        # ponytail: a 4 m centroid bucket, not a geometric-overlap test. It
        # catches the co-located duplicate (one shell tagged in both the
        # buildings and the pois layer) and nothing else; go to overlap-based
        # dedup only if duplicates come back.
        bucket = (round(cx / 4.0), round(cy / 4.0))
        key = (("named", name) if name else ("anon", otype)) + bucket
        cand = {
            "poly": poly,
            "name": name,
            "object_type": otype,
            "area": float(poly.area),
        }
        prev = best.get(key)
        if prev is None:
            best[key] = cand
            continue
        if (_TYPE_RANK.get(otype, 0), cand["area"]) > (
            _TYPE_RANK.get(prev["object_type"], 0),
            prev["area"],
        ):
            best[key] = cand

    feats = list(best.values())
    feats.sort(
        key=lambda f: (
            f["name"] or "~",
            round(f["poly"].centroid.x, 2),
            round(f["poly"].centroid.y, 2),
        )
    )

    objects: dict[str, dict] = {}
    for oid, f in enumerate(feats):
        poly = f["poly"]
        minx, miny, maxx, maxy = poly.bounds
        objects[str(oid)] = {
            "id": oid,
            "position": [round(float(poly.centroid.x), 3), round(float(poly.centroid.y), 3),
                         round(GROUND_LEVEL_M, 3)],
            "dimension": [round(float(maxx - minx), 3), round(float(maxy - miny), 3)],
            "contour": _contour_xy(poly),
        }

    log.info("%s: %d objects, type counts %s", map_name, len(objects),
             dict(Counter(f["object_type"] for f in feats)))
    return objects


def build(root: Path, bundle: Path) -> Path:
    all_objects = {"calcity": build_block(bundle, "calcity", AOI_BBOX)}

    obj_path = root / "data" / "cityrefer" / "objects.json"
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    # Write exactly what this build produced: no merge into an existing file.
    obj_path.write_text(json.dumps(all_objects, separators=(",", ":")))
    log.info(
        "wrote %s (%d maps, %d objects)",
        obj_path,
        len(all_objects),
        sum(len(v) for v in all_objects.values()),
    )

    return obj_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bench = calcity_bench(os.environ)
    parser.add_argument("--root", type=Path, default=bench.root)
    parser.add_argument("--bundle", type=Path, default=None)
    args = parser.parse_args(argv)
    build(args.root, args.bundle or bundle_root(bench.root, os.environ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
