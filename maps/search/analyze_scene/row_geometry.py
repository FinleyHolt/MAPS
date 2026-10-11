"""Footprint geometry between served rows: the measurements ``xy`` cannot make.

A served row carries ONE point, and every distance computed off it is wrong
for an extended feature. Distance on this surface is FOOTPRINT GAP, mask to
mask, over the geometry the live surface holds by fid
(:func:`geom_compute.gap_witness`, the quantity the executor's ``distance to
<ref>`` column carries).

A SET IS ONE PLACE AND MEANS ITS NEAREST MEMBER: the gap from a house to a
set of pools is the gap to the closest pool (:func:`geom_compute.set_gap`).

Geometry is resolved BY FID off the live surface, because op OUTPUT rows
strip the mask fields (``geom_program._MODEL_HIDDEN_ROW_FIELDS``) and a row
holding only its point would degrade every gap to a centroid distance.
"""

from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G
from maps.util.xy_token import is_xy

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
    detection a flight just folded in measures like any prior. Each place is
    one :func:`geom_compute.mask_geom`, and a set of places is its members,
    built once and reused down every candidate.
    """

    def __init__(self, row_of_fid):
        self.row_of_fid = row_of_fid
        self._geoms: dict = {}
        self._sets: dict = {}

    def clear(self) -> None:
        self._geoms.clear()
        self._sets.clear()

    def _geom(self, place):
        got = self._geoms.get(place)
        if got is None:
            fid, xy = place
            row = self.row_of_fid(fid) if fid else None
            chains = chains_of(row) if isinstance(row, dict) else []
            if not chains and is_xy(xy):
                chains = [[[float(xy[0]), float(xy[1])]]]
            got = self._geoms[place] = G.mask_geom(chains)
        return got

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
        if not ga or not gb:
            return float("inf")
        if len(gb) > len(ga):
            (ga, bba), gb = self.geom(b), ga
        return min(G.set_gap(ga, bba, g)[0] for g in gb)
