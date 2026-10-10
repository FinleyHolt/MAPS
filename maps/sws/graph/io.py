"""Feature graph serialization.

``{stem}_graph.npz`` + ``{stem}_nodes.json``: the npz carries the fid index
+ graph scalars; the per-node JSON carries the ``MapObject`` metadata. Graphs
are nodes-only; packed edge columns an npz may carry are never read.
"""

from __future__ import annotations

import json as _json
import os

import numpy as np

from maps.sws.graph.model import ObjectGraph, MapObject


# The centerline family (centerline / centerline_arc_m / axis_length_m /
# axis_residual_m / centerline_method) is a DETERMINISTIC reduction of the
# persisted ``cells``, so it is DERIVED (``derive_centerlines`` below, called by
# the loader AND by the runtime upsert) rather than separately persisted. A
# field re-derived from its source can never drift from it, and a version bump
# for it would hard-refuse every existing tile for nothing. RULE: bump this
# version ONLY for a genuinely persisted-and-read atom, NEVER for a derived
# geometry. The schema_version field + the load-refusal gate are the machinery
# for that case.
#
# v3 is that case. ``world_contour`` holds a feature's EXACT vertex chain for
# every node — a ring, a way or a single point — and states its own closure by
# returning to its first vertex (``geom_compute.is_closed_chain``). A v2 ring
# carries no closing vertex, so read under v3 it becomes an OPEN chain: no
# enclosed area, no containment, and a missing closing edge, all silently.
# That is precisely the "persisted-and-read atom changed format" case, so the
# loader refuses a v2 tile instead of reading it wrong.
GRAPH_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Node <-> dict (shared by JSON + binary formats)
# ---------------------------------------------------------------------------


def _node_to_dict(node: MapObject) -> dict:
    nd: dict = {
        "id": node.id,
        "category": node.class_name,
        "name": node.name,
        "tags": node.tags,
        "centroid": list(node.centroid),
        "bbox": list(node.bbox),
        "cell_count": node.cell_count,
        "orientation": node.orientation,
        "orientation_confidence": node.orientation_confidence,
    }
    if node.provenance != "prebaked":
        nd["provenance"] = node.provenance
    if node.first_seen_t is not None:
        nd["first_seen_t"] = node.first_seen_t
    if node.last_seen_t is not None:
        nd["last_seen_t"] = node.last_seen_t
    if node.observation_count:
        nd["observation_count"] = node.observation_count
    if node.best_view_px:
        nd["best_view_px"] = int(node.best_view_px)
    if node.confidence != 1.0:
        nd["confidence"] = node.confidence
    if node.embedding is not None:
        nd["embedding"] = list(node.embedding)
    if node.semantic_attributes:
        nd["semantic_attributes"] = {
            k: dict(v) for k, v in node.semantic_attributes.items()
        }
    if node.track_id is not None:
        nd["track_id"] = node.track_id
    if node.cells is not None:
        nd["cells"] = [[int(r), int(c)] for r, c in node.cells]
    if node.world_contour is not None:
        nd["world_contour"] = [[float(x), float(y)] for x, y in node.world_contour]
    if node.world_xy is not None:
        nd["world_xy"] = [float(node.world_xy[0]), float(node.world_xy[1])]
    # Raw per-node spatial geometry (grid frame; see MapObject docstring).
    if node.centerline is not None:
        nd["centerline"] = [[float(r), float(c)] for r, c in node.centerline]
    if node.centerline_arc_m is not None:
        nd["centerline_arc_m"] = [float(v) for v in node.centerline_arc_m]
    if node.axis_length_m:
        nd["axis_length_m"] = float(node.axis_length_m)
    if node.axis_residual_m:
        nd["axis_residual_m"] = float(node.axis_residual_m)
    if node.centerline_method:
        nd["centerline_method"] = node.centerline_method
    return nd


def _dict_to_node(nd: dict) -> MapObject:
    embedding_raw = nd.get("embedding")
    return MapObject(
        id=nd["id"],
        class_name=nd["category"],
        name=nd["name"],
        tags=nd["tags"],
        centroid=tuple(nd["centroid"]),
        bbox=tuple(nd.get("bbox", [0, 0, 0, 0])),
        cell_count=nd["cell_count"],
        orientation=nd.get("orientation"),
        orientation_confidence=nd.get("orientation_confidence", 0.0),
        provenance=nd.get("provenance", "prebaked"),
        first_seen_t=nd.get("first_seen_t"),
        last_seen_t=nd.get("last_seen_t"),
        observation_count=nd.get("observation_count", 0),
        best_view_px=nd.get("best_view_px", 0),
        confidence=nd.get("confidence", 1.0),
        embedding=list(embedding_raw) if embedding_raw else None,
        semantic_attributes={
            k: {str(val): int(cnt) for val, cnt in counts.items()}
            for k, counts in nd.get("semantic_attributes", {}).items()
        },
        track_id=nd.get("track_id"),
        cells=(
            np.asarray(nd["cells"], dtype=np.int32).reshape(-1, 2)
            if "cells" in nd
            else None
        ),
        world_contour=(
            [[float(x), float(y)] for x, y in nd["world_contour"]]
            if nd.get("world_contour")
            else None
        ),
        world_xy=(
            (float(nd["world_xy"][0]), float(nd["world_xy"][1]))
            if nd.get("world_xy")
            else None
        ),
        centerline=(
            [[float(r), float(c)] for r, c in nd["centerline"]]
            if nd.get("centerline") is not None
            else None
        ),
        centerline_arc_m=(
            [float(v) for v in nd["centerline_arc_m"]]
            if nd.get("centerline_arc_m") is not None
            else None
        ),
        axis_length_m=float(nd.get("axis_length_m", 0.0)),
        axis_residual_m=float(nd.get("axis_residual_m", 0.0)),
        centerline_method=nd.get("centerline_method", ""),
    )


# ---------------------------------------------------------------------------
# Binary format: npz (fid index + scalars) + nodes.json
# ---------------------------------------------------------------------------


def save_graph_binary(graph: ObjectGraph, path_stem: str) -> None:
    """Write ``{stem}_graph.npz`` + ``{stem}_nodes.json`` (nodes only)."""
    fids = list(graph.nodes.keys())

    np.savez(
        f"{path_stem}_graph.npz",
        fids=np.array(fids, dtype=object),
        cell_size_m=np.float32(graph.cell_size_m),
        step_index=np.int64(graph.step_index),
        tau=np.float64(graph.tau if graph.tau is not None else np.nan),
        schema_version=np.int32(GRAPH_SCHEMA_VERSION),
    )

    nodes_doc = {fid: _node_to_dict(node) for fid, node in graph.nodes.items()}
    with open(f"{path_stem}_nodes.json", "w") as f:
        _json.dump(nodes_doc, f, separators=(",", ":"))


def derive_centerlines(nodes: dict, cell_size_m: float) -> None:
    """Derive the per-node centerline family from each node's ``cells``.

    The centerline (``centerline`` / ``centerline_arc_m`` / ``axis_length_m`` /
    ``axis_residual_m`` / ``centerline_method``) is a parameter-free PCA reduction
    of the mask cells, NOT an independent atom -- so it is derived via the SAME
    ``builder._mask_centerline`` the bake uses, so a loaded graph and a baked
    one carry the same centerline by construction.

    ``cell_size_m`` is the graph's cell size (read from the npz by the load
    caller -- it is required there). Only nodes that carry ``cells`` but no
    ``centerline`` are touched (a tile baked before the field existed); a node
    that already has one is kept as-is.

    TWO callers, and they must stay ONE function. ``load_graph_binary`` runs it
    before the ObjectGraph is constructed, so the mutated nodes are the ones the
    graph serves. ``runtime`` runs it on a detected node as it stamps that
    node's ``cells``, because a graph that is never SAVED is never LOADED: the
    world-state ingest server hands its graph back through ``save_graph`` ->
    ``load_graph`` and its detected nodes arrive carrying a centerline, while
    the in-process upsert produced the same nodes without one. Those are the two
    branches of ``search_loop._upsert_detections``, and the geom ops read
    ``node.centerline`` to decide whether a fid resolves as a LINE reference or
    a point (``analyze_scene/geom_program.py`` ``_cl`` / ``_axis_q``), so the
    difference reached answers. A second copy of this reduction in runtime.py is
    how the two paths would silently drift apart again.

    ponytail: derive-if-absent, so a detected node's centerline stays its FIRST
    frame's footprint while its ``cells`` are re-stamped every frame. That
    staleness is the incumbent behaviour on the server path and is matched here
    deliberately -- correcting it changes the geom surface for detected fids and
    needs a paired SR read, not a plumbing commit. Re-derive on every re-stamp
    when someone pays for that read.
    """
    from maps.sws.graph.builder import _mask_centerline  # local: avoid import cycle

    for node in nodes.values():
        if getattr(node, "centerline", None) is not None:
            continue
        cells = getattr(node, "cells", None)
        if cells is None or len(cells) < 2:
            continue
        geom = _mask_centerline(np.asarray(cells), cell_size_m=cell_size_m)
        if geom is None:
            continue
        (node.centerline, node.centerline_arc_m, node.axis_length_m,
         node.axis_residual_m, node.centerline_method) = geom


def load_graph_binary(path_stem: str) -> ObjectGraph:
    """Read ``{stem}_graph.npz`` + ``{stem}_nodes.json`` into an ObjectGraph.

    Nodes only. Packed edge columns (``src``/``dst``/``relation``/…) an
    older npz carries are never read, and an npz loads an array only when it
    is accessed, so they never reach RAM.
    """
    npz_path = f"{path_stem}_graph.npz"
    nodes_path = f"{path_stem}_nodes.json"
    data = np.load(npz_path, allow_pickle=True)
    with open(nodes_path) as f:
        nodes_doc = _json.load(f)

    fids = [str(x) for x in data["fids"]]
    nodes: dict[str, MapObject] = {fid: _dict_to_node(nodes_doc[fid]) for fid in fids}

    # Schema version is required and must match the current build. Every
    # tile is rebaked when the schema bumps, and the loader refuses a
    # mismatched tile rather than guess a compatibility read.
    # ``./benchmarks/citynav/data_prep/bake_tiles.sh`` is the recipe.
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
    cell_size_m_val = float(np.asarray(data["cell_size_m"]).item())
    # Centerlines are a deterministic reduction of `cells`, derived on load (not
    # persisted) so the served surface can never drift from the trained one. Must
    # precede the ObjectGraph build so the derived nodes are the ones it serves.
    derive_centerlines(nodes, cell_size_m_val)
    graph = ObjectGraph(nodes=nodes, cell_size_m=cell_size_m_val)
    step_index = int(data["step_index"]) if "step_index" in data.files else 0
    tau_arr = data["tau"] if "tau" in data.files else None
    graph.step_index = step_index
    if tau_arr is not None:
        tau_val = float(tau_arr)
        graph.tau = None if np.isnan(tau_val) else tau_val
    for fid, node in nodes.items():
        graph.objects_by_class.setdefault(node.class_name, {})[fid] = node
    return graph


# ---------------------------------------------------------------------------
# Public I/O entry points
# ---------------------------------------------------------------------------


def save_graph(graph: ObjectGraph, path_stem: str) -> None:
    """Write the graph to disk in the binary format."""
    save_graph_binary(graph, path_stem)


def load_graph(path_stem: str) -> ObjectGraph:
    """Load the graph from its binary format."""
    npz_path = f"{path_stem}_graph.npz"
    nodes_path = f"{path_stem}_nodes.json"

    if os.path.exists(npz_path) and os.path.exists(nodes_path):
        return load_graph_binary(path_stem)

    # An npz without nodes.json carries arrays only: signal absence.
    raise FileNotFoundError(
        f"No graph sidecar at {npz_path} and {nodes_path}"
    )
