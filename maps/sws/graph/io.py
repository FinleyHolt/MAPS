"""Feature graph serialization.

``{stem}_graph.npz`` + ``{stem}_nodes.json``: the npz carries the fid index
+ graph scalars; the per-node JSON carries the ``MapObject`` metadata, one
key per field in field order. Graphs are nodes-only; packed edge columns an
npz may carry are never read.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from maps.sws.graph.model import MapObject, ObjectGraph


# Bumped only when a persisted field that is read changes format; the loader
# refuses any other version. v3: ``world_contour`` states a ring's closure by
# repeating its first vertex (``geom_compute.is_closed_chain``), so a ring
# without it would read as an open chain.
GRAPH_SCHEMA_VERSION = 3

_FIELDS = dataclasses.fields(MapObject)
# ``class_name`` through ``centroid`` are written on every node; every later
# field only when it differs from its default.
_ALWAYS = 4
# The JSON key where it is not the field name.
_KEY = {"class_name": "category"}


def _pairs(cast):
    return lambda v: [[cast(a), cast(b)] for a, b in v]


def _floats(v):
    return [float(x) for x in v]


_TO_JSON = {
    "centroid": list, "best_view_px": int,
    "semantic_attributes": lambda v: {k: dict(c) for k, c in v.items()},
    "cells": _pairs(int), "world_contour": _pairs(float), "centerline": _pairs(float),
    "world_xy": _floats, "axis_length_m": float,
}
_FROM_JSON = {
    "centroid": tuple,
    "semantic_attributes": lambda v: {k: {str(a): int(n) for a, n in c.items()} for k, c in v.items()},
    "cells": lambda v: np.asarray(v, dtype=np.int32).reshape(-1, 2),
    "world_contour": lambda v: _pairs(float)(v) or None,
    "world_xy": lambda v: (float(v[0]), float(v[1])) if v else None,
    "centerline": _pairs(float), "axis_length_m": float,
}


def _is_default(f: dataclasses.Field, v) -> bool:
    if f.default is None:
        return v is None
    return v == (f.default_factory() if f.default is dataclasses.MISSING else f.default)


def _node_to_dict(node: MapObject) -> dict:
    nd: dict = {}
    for i, f in enumerate(_FIELDS):
        v = getattr(node, f.name)
        if i < _ALWAYS or not _is_default(f, v):
            nd[_KEY.get(f.name, f.name)] = _TO_JSON.get(f.name, lambda x: x)(v)
    return nd


def _dict_to_node(nd: dict) -> MapObject:
    return MapObject(**{
        f.name: _FROM_JSON.get(f.name, lambda x: x)(nd[key])
        for f in _FIELDS if (key := _KEY.get(f.name, f.name)) in nd
    })


def save_graph(graph: ObjectGraph, path_stem: str) -> None:
    """Write ``{stem}_graph.npz`` + ``{stem}_nodes.json`` (nodes only)."""
    np.savez(
        f"{path_stem}_graph.npz",
        fids=np.array(list(graph.nodes), dtype=object),
        cell_size_m=np.float32(graph.cell_size_m),
        schema_version=np.int32(GRAPH_SCHEMA_VERSION),
    )
    with open(f"{path_stem}_nodes.json", "w") as f:
        json.dump({fid: _node_to_dict(n) for fid, n in graph.nodes.items()}, f, separators=(",", ":"))


def load_graph(path_stem: str) -> ObjectGraph:
    """Read ``{stem}_graph.npz`` + ``{stem}_nodes.json`` into an ObjectGraph.

    Nodes only. Packed edge columns (``src``/``dst``/``relation``/…) an npz
    may carry are never read, and an npz loads an array only when it is
    accessed, so they never reach RAM.
    """
    npz_path = f"{path_stem}_graph.npz"
    nodes_path = f"{path_stem}_nodes.json"
    if not (os.path.exists(npz_path) and os.path.exists(nodes_path)):
        raise FileNotFoundError(f"No graph sidecar at {npz_path} and {nodes_path}")
    data = np.load(npz_path, allow_pickle=True)
    file_schema = int(data["schema_version"]) if "schema_version" in data.files else 0
    if file_schema != GRAPH_SCHEMA_VERSION:
        raise ValueError(
            f"{npz_path} is schema v{file_schema}, expected v{GRAPH_SCHEMA_VERSION}. "
            "Re-bake the tile (benchmarks/citynav/data_prep/bake_tiles.sh) and re-pull."
        )
    if "cell_size_m" not in data.files:
        raise ValueError(
            f"{npz_path} has no cell_size_m. Re-bake the tile via "
            "benchmarks/citynav/data_prep/bake_tiles.sh."
        )
    with open(nodes_path) as f:
        nodes_doc = json.load(f)
    nodes = {fid: _dict_to_node(nodes_doc[fid]) for fid in (str(x) for x in data["fids"])}
    return ObjectGraph(nodes=nodes, cell_size_m=float(np.asarray(data["cell_size_m"]).item()))
