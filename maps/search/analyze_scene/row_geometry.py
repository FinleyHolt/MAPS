"""Footprint geometry between served rows: the measurements ``xy`` cannot make.

A served row carries ONE point, and every distance computed off it is wrong
for an extended feature. Distance on this surface is FOOTPRINT GAP, mask to
mask, over the geometry the live surface holds by fid
(:func:`geom_compute.gap_witness`, the quantity the executor's ``distance to
<ref>`` column carries).

A SET IS ONE PLACE AND MEANS ITS NEAREST MEMBER: the union's chains are one
geometry, so the gap from a house to a set of pools is the gap to the
closest pool.

Geometry is resolved BY FID off the live surface, because op OUTPUT rows
strip the mask fields (``geom_program._MODEL_HIDDEN_ROW_FIELDS``) and a row
holding only its point would degrade every gap to a centroid distance.
"""

from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G
from maps.util.xy_token import is_xy

# A place, resolved: ``(fid | None, (x, y) | None)``. A fid reaches a mask;
# the point is all a computed row (a crossing, a terminus) has.
Place = tuple


def chains_of(row) -> list:
    """One surface row's footprint as mask chains.

    Its mask when the surface holds one, else its own point.
    """
    if not isinstance(row, dict):
        return []
    mask = row.get("mask")
    if mask:
        return list(mask)
    xy = row.get("xy")
    if is_xy(xy):
        return [[[float(xy[0]), float(xy[1])]]]
    return []


class SurfaceGeometry:
    """Mask geometry by fid off the LIVE surface, cached per block.

    ``row_of_fid`` reads one row of the surface as it stands NOW, so a
    detection a flight just folded in measures like any prior. The cache is
    keyed by the whole place tuple: a union of many ways is ONE
    :func:`geom_compute.mask_geom`, built once and reused down every
    candidate.
    """

    def __init__(self, row_of_fid):
        self.row_of_fid = row_of_fid
        self._geoms: dict = {}

    def clear(self) -> None:
        self._geoms.clear()

    def geom(self, places: tuple):
        """The union geometry of a tuple of places, built once."""
        got = self._geoms.get(places)
        if got is None:
            chains: list = []
            for fid, xy in places:
                row = self.row_of_fid(fid) if fid else None
                part = chains_of(row) if isinstance(row, dict) else []
                if not part and is_xy(xy):
                    part = [[[float(xy[0]), float(xy[1])]]]
                chains.extend(part)
            got = self._geoms[places] = G.mask_geom(chains)
        return got

    def distance(self, a: tuple, b: tuple) -> float:
        """Metres between two footprints. 0.0 on touch or containment."""
        return G.gap_witness(self.geom(a), self.geom(b))[0]
