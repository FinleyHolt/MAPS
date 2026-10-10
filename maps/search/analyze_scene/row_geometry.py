"""Footprint geometry between served rows: the measurements ``xy`` cannot make.

A served row carries ONE point. Every distance computed off that point is
wrong for an extended feature, and the bake is mostly extended features:
Poppy Boulevard way 5 is 1,198 m long, and the gap from it to the car parked
against it is 7.0 m where the centroid-to-centroid distance is 408.4 m. So
distance on this surface is FOOTPRINT GAP, mask to mask, measured here over
the geometry the live surface holds by fid --
:func:`geom_compute.gap_witness`, the exact quantity the executor's
``distance to <ref>`` column has always carried.

A SET IS ONE PLACE AND MEANS ITS NEAREST MEMBER. The union's chains are one
geometry, so the gap from a house to a set of pools is the gap to whichever
pool is closest. That rule was true inside the executor and served nowhere,
and it is the single measured cause of the calcity::200#198 miss.

The cardinal relation is NOT served here. Which side of an extended feature a
point lies on is an ANCHORED question and it is answered by the ``<dir>_of=``
cut and the signed ``<dir> of <ref>`` column that rides with it
(:mod:`cardinal_side` through the compare op), not by a free measuring call:
ordering is global and cuts are anchored, so a free ``side(a, b, dir)`` was
the wrong arity for either job. It also could not do the one thing that would
have justified it -- an abstention returns ``None``, so using it as a sort key
raised inside the sort.

Geometry is resolved BY FID off the live surface, because op OUTPUT rows
strip the mask fields (``geom_program._MODEL_HIDDEN_ROW_FIELDS``): a carried
row that kept only its own point would silently degrade every gap to a
centroid distance, which is the defect this module exists to make
impossible.
"""

from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G

# A place, resolved: ``(fid | None, (x, y) | None)``. Both halves ride
# because a fid is the only way to reach a mask and the point is the only
# thing a computed row (a crossing, a terminus) has.
Place = tuple


def chains_of(row) -> list:
    """One surface row's footprint as mask chains.

    Its mask when the surface holds one, else its own point -- a point's
    position IS its whole footprint. ``geom_program._row_geom_center``'s law,
    reached from the surface side.
    """
    if not isinstance(row, dict):
        return []
    mask = row.get("mask")
    if mask:
        return list(mask)
    xy = row.get("xy")
    if _is_xy(xy):
        return [[[float(xy[0]), float(xy[1])]]]
    return []


def _is_xy(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                    for c in v))


class SurfaceGeometry:
    """Mask geometry by fid off the LIVE surface, cached per block.

    ``row_of_fid`` reads one row of the surface as it stands NOW, so a
    detection a flight just folded in measures like any prior. The cache is
    keyed by the whole place tuple: a union of 64 ways is ONE
    :func:`geom_compute.mask_geom`, built once and reused down every
    candidate, which is what keeps ``[gap(h, road) for h in houses]``
    linear in the houses rather than quadratic in the ways.
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
                if not part and _is_xy(xy):
                    part = [[[float(xy[0]), float(xy[1])]]]
                chains.extend(part)
            got = self._geoms[places] = G.mask_geom(chains)
        return got

    def distance(self, a: tuple, b: tuple) -> float:
        """Metres between two footprints. 0.0 on touch or containment."""
        return G.gap_witness(self.geom(a), self.geom(b))[0]

def demo() -> None:
    """The gap a centroid cannot measure, and the union that is one place."""
    road = {"fid": "road", "mask": [[[0.0, 0.0], [0.0, 1000.0]]],
            "xy": [0.0, 500.0]}
    car = {"fid": "car", "xy": [7.0, 0.0]}
    far = {"fid": "far", "xy": [400.0, 0.0]}
    sg = SurfaceGeometry({"road": road, "car": car, "far": far}.get)
    p_road, p_car = (("road", None),), (("car", None),)
    assert abs(sg.distance(p_car, p_road) - 7.0) < 1e-6, sg.distance(p_car, p_road)
    # the centroid distance the model would have computed off xy alone
    assert abs(((7.0 - 0.0) ** 2 + (0.0 - 500.0) ** 2) ** 0.5 - 500.05) < 0.1
    # a SET is its NEAREST member
    both = (("car", None), ("far", None))
    assert abs(sg.distance((("far", None),), p_road) - 400.0) < 1e-6
    assert abs(sg.distance(both, p_road) - 7.0) < 1e-6
    print("row_geometry ok: gap 7.0 m where the centroids are 500 m apart")


if __name__ == "__main__":
    demo()
