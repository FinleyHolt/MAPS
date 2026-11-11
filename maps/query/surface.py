"""The ``(cands, anchors)`` surface the analyze executor runs over: one row
per graph node, detections as ``cands`` and priors as ``anchors``."""
from __future__ import annotations

import re

import numpy as np

from maps.query import color_words
from maps.query import spatial_math as G
from maps.query.row_geometry import cells_of, mask_center


def _r2(p) -> list:
    """An ``[x, y]`` at 2 dp."""
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def _modal_attr(n, key):
    """The modal value of ``key`` in the node's ``semantic_attributes``, or None."""
    cc = (n.semantic_attributes or {}).get(key)
    return max(cc.items(), key=lambda kv: kv[1])[0] if cc else None


# Bake bookkeeping some bakes stamp on every row: never served, and never
# searched (as stopwords on every row, ``osm_id`` would donate ``way``).
_UNSERVED_TAGS = frozenset({
    "source", "osm_id", "worldkit_layer", "worldkit_height_m", "source_tag"})


def _row_tags(n) -> dict:
    """A prior row's tags: the node's string-valued OSM tags but _UNSERVED_TAGS."""
    tags = n.tags or {}
    if not isinstance(tags, dict):
        return {}
    return {str(k): v for k, v in tags.items()
            if isinstance(v, str) and str(k).lower() not in _UNSERVED_TAGS}


def _anchor_search_text(n) -> str:
    """A prior row's search text: its name and its tag keys and values."""
    parts = [str(n.name or "")]
    for k, v in _row_tags(n).items():
        parts.append(str(k).replace("_", " "))
        parts.append(str(v).replace("_", " "))
    return " ".join(p for p in parts if p).lower()


def _color_of(label: str) -> str:
    """The first palette word the label carries as a whole word, else ""."""
    words = set(re.findall(r"[a-z]+", (label or "").lower()))
    return next((w for w in color_words.COLOR_FAMILY if w in words), "")


def _color_fields(n) -> dict:
    """``color`` (perceived, else a color word in the class label) plus
    ``color_conf`` / ``color_shade`` / ``color_flag`` where perception set
    them."""
    color = _modal_attr(n, "color")
    fields = {"color": color if color is not None
              else _color_of(str(n.class_name or ""))}
    conf = _modal_attr(n, "color_conf")
    if conf is not None:
        try:
            fields["color_conf"] = round(float(conf), 3)
        except ValueError:
            pass
    shade = _modal_attr(n, "color_shade")
    if shade is not None:
        fields["color_shade"] = shade
    flag = _modal_attr(n, "color_flag")
    if flag:
        fields["color_flag"] = flag
    return fields


def _polyline_arc_midpoint(verts) -> list:
    """The arc-length midpoint of a world-meter polyline."""
    arcs = G.arc_lengths(verts)
    half = arcs[-1] / 2.0
    for i in range(len(verts) - 1):
        if arcs[i + 1] >= half:
            seg = arcs[i + 1] - arcs[i]
            t = (half - arcs[i]) / seg if seg > 0 else 0.0
            return [float(verts[i][0]) + t * (float(verts[i + 1][0]) - float(verts[i][0])),
                    float(verts[i][1]) + t * (float(verts[i + 1][1]) - float(verts[i][1]))]
    return [float(verts[-1][0]), float(verts[-1][1])]


def representative_point(chains) -> list:
    """A point ON the geometry, the row's flight target, at 2 dp: a closed
    chain's area centroid when it lies inside the mask, else the arc
    midpoint of the longest chain (a mean over a bent way lies off it)."""
    geom = G.mask_geom(chains)
    if geom["closed"]:
        c = G.loops_area_centroid([ch for ch in chains
                                   if G.is_closed_chain(ch)])
        if c is not None and bool(G.mask_contains(geom, [c])[0]):
            return _r2(c)
    return _r2(_polyline_arc_midpoint(max(chains, key=len)))


# The row builders read the tile frame ``(tx, ty, cs)``: its origin and cell
# size in metres.

def _cell_xy(rc, frame) -> list:
    tx, ty, cs = frame
    return _r2(G.cell_centre_xy(rc[0], rc[1], (tx, ty), cs))


def _row_xy(n, mask, frame):
    """The row's served point, the ``go_to``/``explore`` flight target: the
    tracker's ``world_xy``, else the mask's :func:`representative_point`,
    else the centroid cell's centre."""
    w = n.world_xy
    if w is not None and len(w) >= 2:
        return _r2(w)
    if mask:
        return representative_point(mask)
    c = n.centroid
    return _cell_xy(c, frame) if c is not None and len(c) >= 2 else None


def _cells(n) -> list:
    """The node's 2 m mesh as a plain (row, col) int list, so a surface
    round-trips through JSON."""
    c = cells_of(n)
    return c.tolist() if len(c) else []


def _exact_chains(n, frame) -> list:
    """The node's ``world_contour`` (one ring, way or point) clipped to the
    box of its own cells plus one cell, since the bake clips cells to the
    tile while a contour runs on. A node with no cells is not clipped."""
    tx, ty, cs = frame
    wc = n.world_contour
    chains = [[[float(x), float(y)] for x, y in wc]] if wc else []
    cl = cells_of(n)
    if not len(cl) or not chains:
        return chains
    r0, c0 = cl[:, 0].min(), cl[:, 1].min()
    r1, c1 = cl[:, 0].max(), cl[:, 1].max()
    return G.clip_chains_to_box(chains, (
        tx + max(int(c0) - 1, 0) * cs, ty + max(int(r0) - 1, 0) * cs,
        tx + (int(c1) + 2) * cs, ty + (int(r1) + 2) * cs))


def _masked(m) -> dict:
    c = mask_center(m)
    return {"mask": m, **({"mask_centroid": _r2(c)} if c is not None else {})}


def _mask_fields(n, frame) -> dict:
    """``mask`` (the contour, else the cell boundary loops), its centroid and
    the shape scalars; none for a node without either."""
    tx, ty, cs = frame
    cells = n.cells
    chains = _exact_chains(n, frame)
    # A one-vertex chain (a POI) draws nothing: the cell boundary does.
    if any(len(ch) >= 2 for ch in chains):
        out = _masked(chains)
        shape = G.contour_shape_features(chains)
        if shape:
            out.update(shape)
        return out
    if cells is None or len(cells) == 0:
        return {}
    loops = G.cells_boundary_loops(cells, (tx, ty), cs)
    out = _masked(loops) if loops else {}
    # Below 6 cells the 2 m grid cannot express shape.
    if len(cells) >= 6:
        out.update(G.mask_shape_features(cells, cs))
    return out


def _cand_row(f, n, frame):
    mf = _mask_fields(n, frame)
    xy = _row_xy(n, mf.get("mask"), frame)
    if not xy:
        return None
    return {"fid": str(f), "xy": xy, "centerline": None,
            "cells": _cells(n),
            "class": str(n.class_name or ""),
            **_color_fields(n),
            **mf,
            "search_text": str(n.class_name or "").lower()}


def _anchor_row(f, n, frame):
    """A prior row is one way and carries no class: its name and tags say
    what it is. A centerline row carries its termini."""
    mf = _mask_fields(n, frame)
    xy = _row_xy(n, mf.get("mask"), frame)
    if not xy:
        return None
    cl = [_cell_xy(p, frame) for p in n.centerline] if n.centerline else None
    axis = n.axis_length_m
    row = {"fid": str(f), "name": n.name,
           "xy": xy,
           "centerline": cl,
           "cells": _cells(n),
           # Only a node with a centerline has an axis length.
           "axis_length_m": round(float(axis), 2) if cl and axis is not None else None,
           "tags": _row_tags(n),
           **_color_fields(n),
           **mf,
           "search_text": _anchor_search_text(n)}
    if cl and len(cl) >= 2:
        row["end0_xy"], row["end1_xy"] = _r2(cl[0]), _r2(cl[-1])
    return row


# A surface row is a pure function of its node's content and the tile
# geometry, so rows are cached by content (_content_key over the node's whole
# __dict__, so a new MapObject field cannot be invisible to the key): ingest
# returns fresh node objects. The cache is dropped whole at its cap.
_ROW_CACHE: "dict[tuple, dict | None]" = {}
_ROW_CACHE_MAX = 50_000


def _content_key(v):
    """A hashable summary of ``v`` that changes whenever its content does.
    Arrays and numeric sequences hash their bytes (``cells`` is the bulk of a
    node)."""
    if isinstance(v, (list, tuple)):
        try:
            a = np.asarray(v)
        except ValueError:  # ragged: walk it
            a = None
        if a is None or a.dtype.kind not in "ifb":
            return ("seq", tuple(_content_key(x) for x in v))
        v = a
    if isinstance(v, np.ndarray):
        return ("nd", v.shape, v.dtype.str, hash(v.tobytes()))
    if isinstance(v, dict):
        return ("map", tuple(sorted((str(k), _content_key(x)) for k, x in v.items())))
    return v


def _rows_by_content(kind, nodes, build_row, frame):
    """One row per ``(fid, node)``, reusing unchanged ones. Each is a
    shallow copy, since the caller stamps fields on it; the heavy values
    are shared and read-only."""
    out = []
    for f, n in nodes:
        key = (kind, str(f), frame, tuple(sorted((k, _content_key(v)) for k, v in vars(n).items())))
        if key in _ROW_CACHE:
            row = _ROW_CACHE[key]
        else:
            row = build_row(f, n, frame)
            if len(_ROW_CACHE) >= _ROW_CACHE_MAX:
                _ROW_CACHE.clear()
            _ROW_CACHE[key] = row
        if row is not None:
            out.append(dict(row))
    return out


def _fid_key(f):
    """detected_2 before detected_100; a fid without a numeric tail after.
    Numeric fid order is mint order."""
    s = str(f)
    head, _, tail = s.rpartition("_")
    return ((head, 0, int(tail), "") if tail.isdigit()
            else (s, 1, 0, s))


def build_surface_from_graph(nodes_iter, origin_xy, cell_m):
    """The ``(cands, anchors)`` surface from a graph's ``(fid, node)`` pairs,
    split by ``node.provenance`` and each sorted by fid: one row per node,
    with its mask chains and shape scalars when masked and a prior's world
    centerline."""
    det, prior = [], []
    for fid, node in nodes_iter:
        (det if node.provenance == "detected" else prior).append((fid, node))
    frame = (float(origin_xy[0]), float(origin_xy[1]), float(cell_m))
    cands = _rows_by_content("c", det, _cand_row, frame)
    anchors = _rows_by_content("a", prior, _anchor_row, frame)
    # Duplicate detections (same rounded xy and class) fold to the one with
    # a color, then the higher color_conf, then the lower fid.
    dups: dict = {}
    for c in cands:
        dups.setdefault((round(float(c["xy"][0]), 1), round(float(c["xy"][1]), 1),
                         c.get("class") or ""), []).append(c)
    cands = sorted((min(g, key=lambda r: (not r.get("color"), -float(r.get("color_conf") or 0.0),
                                          _fid_key(r["fid"])))
                    for g in dups.values()), key=lambda c: _fid_key(c["fid"]))
    anchors.sort(key=lambda a: _fid_key(a["fid"]))
    # A mask containing every row position is 0 m from every row:
    # ``covers_window`` lets compare state that once instead.
    rows = cands + anchors
    if len(rows) >= 3:
        pts = [r["xy"] for r in rows]
        wx0, wx1 = min(p[0] for p in pts), max(p[0] for p in pts)
        wy0, wy1 = min(p[1] for p in pts), max(p[1] for p in pts)
        for r in rows:
            m = r.get("mask")
            if not m:
                continue
            mx = [p[0] for lp in m for p in lp]
            my = [p[1] for lp in m for p in lp]
            if (not mx or min(mx) > wx0 or max(mx) < wx1
                    or min(my) > wy0 or max(my) < wy1):
                continue
            if bool(G.mask_contains(G.mask_geom(m), pts).all()):
                r["covers_window"] = True
    return cands, anchors
