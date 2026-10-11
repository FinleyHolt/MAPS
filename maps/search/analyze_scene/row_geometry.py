"""Distance between served rows, mask to mask (:func:`geom_compute.gap_witness`,
the quantity a ``distance to <ref>`` column carries). Geometry is read by fid
off the live surface, since op output rows carry no mask; a set of places
measures as its nearest member (:func:`geom_compute.set_gap`)."""

from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G
from maps.util.xy_token import is_xy


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
