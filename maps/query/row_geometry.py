"""A row's geometry: its cells, its mask chains, and the distance between
served rows, mask to mask (:func:`spatial_math.gap_witness`, the quantity a
``distance to <ref>`` column carries). Geometry is read by fid off the live
surface, since op output rows carry no mask; a set of places measures as its
nearest member (:func:`spatial_math.set_gap`)."""

from __future__ import annotations

import numpy as np

from maps.query import spatial_math as G
from maps.xy_token import is_xy


def cells_of(row) -> np.ndarray:
    """A surface row's or graph node's ``cells`` (the bake's 2 m ``(row,
    col)`` grid) as an ``(n, 2)`` int array, empty when it carries none."""
    c = row.get("cells") if isinstance(row, dict) else getattr(row, "cells", None)
    return (
        np.zeros((0, 2), dtype=int)
        if c is None
        else np.asarray(c).reshape(-1, 2).astype(int, copy=False)
    )


def chains_of(row) -> list:
    """A surface row's mask chains, else its point as one chain."""
    if not isinstance(row, dict):
        return []
    mask = row.get("mask")
    if mask:
        return list(mask)
    xy = row.get("xy")
    if is_xy(xy):
        return [[[float(xy[0]), float(xy[1])]]]
    return []


def mask_center(mask):
    """The mask's area centroid when it is one connected piece, else None:
    several disjoint places have no centre to measure from."""
    return G.loops_area_centroid(mask) if mask and G.one_place(mask) else None


# A mask list's (G.mask_geom, centre), keyed on the list's identity; the
# entry pins the list so the id cannot be reused. Dropped whole at its cap.
_MASK_GEOM: dict = {}
_MASK_GEOM_MAX = 50_000


def geom_of(row) -> tuple:
    """``(geom, centre)`` of a surface row: its mask, built once per mask
    list, else its point."""
    m = row.get("mask")
    if not m:
        xy = row.get("xy") if is_xy(row.get("xy")) else None
        return G.mask_geom([[xy]] if xy else []), xy
    hit = _MASK_GEOM.get(id(m))
    if hit is None or hit[0] is not m:
        if len(_MASK_GEOM) >= _MASK_GEOM_MAX:
            _MASK_GEOM.clear()
        hit = _MASK_GEOM[id(m)] = (m, G.mask_geom(m), mask_center(m))
    return hit[1], row.get("mask_centroid") or hit[2]


def place_geom(row, fallback) -> tuple:
    """:func:`geom_of` a row read off the surface, else of ``fallback`` when
    that row carries no chains (absent, or neither mask nor point)."""
    return geom_of(row if chains_of(row) else fallback)


class SurfaceGeometry:
    """Mask geometry by fid off the LIVE surface.

    ``row_of_fid`` reads one row of the surface as it stands NOW, so a
    detection a flight just folded in measures like any prior. A place is
    its row's :func:`geom_of`, else its own point; a set of places is its
    members, cached until a library call ends and reused down every
    candidate.
    """

    def __init__(self, row_of_fid):
        self.row_of_fid = row_of_fid
        self._sets: dict = {}

    def clear(self) -> None:
        self._sets.clear()

    def _geom(self, place):
        fid, xy = place
        return place_geom(self.row_of_fid(fid) if fid else None, {"xy": xy})[0]

    def geom(self, places: tuple):
        """``(member geoms, their stacked boxes)`` of a tuple of places."""
        got = self._sets.get(places)
        if got is None:
            geoms = [self._geom(p) for p in places]
            got = self._sets[places] = (geoms, G.stack_bboxes(geoms))
        return got

    def distance(self, a: tuple, b: tuple) -> float:
        """Metres between the nearest members of two sets of places. 0.0 on
        touch or containment."""
        (ga, bba), (gb, _) = self.geom(a), self.geom(b)
        if len(gb) > len(ga):
            (ga, bba), gb = self.geom(b), ga
        return min(G.set_gap(ga, bba, g)[0] for g in gb)
