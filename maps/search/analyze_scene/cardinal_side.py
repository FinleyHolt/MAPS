"""Which side of an extended feature a point is on, in map-fixed cardinals.

"East of Neuralia Road" has to be decidable when the reference is a merged
feature of many ways spanning kilometres. A nearest-point bearing inverts on
a self-approaching shape (a road that loops back past you), a baked
centerline describes one way of many, and a principal axis collapses a bent
road to one direction.

THE RULE: a cardinal is an ABSOLUTE axis, so the only question is how to
aggregate over the reference's extent. Restrict the reference to the cells
lying within a band of the query point on the axis ORTHOGONAL to the
cardinal, then compare along the cardinal: "east of the road WHERE THE ROAD
IS, at my latitude".

It ABSTAINS rather than guessing, in the two cases where there is no side:

* the reference straddles the point inside the band (a loop closing around
  it, a cul-de-sac bulb, a road crossing the band twice), and
* the band holds no reference cells at all (asking "east of" a road that
  runs exactly east-west).

Abstention is a ``None`` return, the executor's missing-value idiom: the
column is absent on that row and ``score_by`` treats it as the neutral.

Cells, never boxes: a road's bounding rectangle describes its empty corners
as confidently as the road.
"""

from __future__ import annotations

import numpy as np

from maps.search.analyze_scene import geom_compute as _gc

# Compass word -> unit vector in the engine's one frame (x = east, y = NORTH),
# derived from geom_compute's table. Only the four axis-aligned cardinals: a
# diagonal ("northeast of") has no single orthogonal axis to band on; a
# diagonal of a POINT is the conjunction of two cardinal cuts.
CARDINALS: dict[str, tuple[float, float]] = {
    w: _gc.COMPASS_UNIT[w] for w in sorted(_gc.CARDINALS)
}

# Half-width of the band, in metres: wide enough that a block-scale target
# still sees the road, narrow enough that a distant limb of the same name
# does not vote. A round number, not a fitted one.
DEFAULT_BAND_M = 150.0


def cardinal_offset(cells_xy, at_xy, direction: str,
                    band_m: float = DEFAULT_BAND_M) -> float | None:
    """Signed metres the point lies in ``direction`` from the reference.

    ``cells_xy`` is the reference's cells as world (x, y) metres -- the union
    over every member way, which is what a merged feature's cells already
    are. Positive means the point IS in that direction, and the magnitude is
    the clearance past the reference's near edge within the band, so the
    value sorts: a row further east of the road ranks above one just east
    of it.

    Returns ``None`` when the relation does not discriminate -- see the
    module docstring for the two cases. A caller must not read ``None`` as
    zero or as "no": it means the geometry declines to say.
    """
    u = CARDINALS.get(str(direction or "").strip().lower())
    if u is None:
        raise ValueError(
            f"cardinal_offset direction must be one of "
            f"{sorted(CARDINALS)}; got {direction!r}")
    pts = np.asarray(cells_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return None
    u = np.asarray(u, dtype=float)
    q = np.asarray(at_xy, dtype=float)

    # The band is on the axis ORTHOGONAL to the query, measured from the
    # point: "at my latitude" for an east/west question.
    orth = np.array([-u[1], u[0]], dtype=float)
    near = pts[np.abs((pts - q) @ orth) <= float(band_m)]
    if not len(near):
        return None                      # nothing of the reference abreast

    along = near @ u
    here = float(q @ u)
    lo, hi = float(along.min()), float(along.max())
    if lo < here < hi:
        return None                      # the reference straddles the point
    return here - hi if here >= hi else here - lo


def densify_segments(seg_a, seg_b, step: float = 2.0):
    """Segment endpoints -> a point cloud at the bake's own 2 m pitch.

    The executor measures on the exact ``mask`` segments; sampling them at
    the cell pitch gives the band rule points to see. A road running a
    kilometre north-south may be TWO vertices, and a band around a point
    halfway along would otherwise contain neither.
    """
    a = np.asarray(seg_a, dtype=float)
    b = np.asarray(seg_b, dtype=float)
    if not len(a):
        return np.empty((0, 2), dtype=float)
    out = [a, b]
    d = np.hypot(*(b - a).T)
    for k in range(1, int(d.max() // step) + 1):
        sel = d > k * step
        if not sel.any():
            break
        t = (k * step) / d[sel]
        out.append(a[sel] + (b[sel] - a[sel]) * t[:, None])
    return np.vstack(out)


def points_from_geoms(geoms):
    """One or more ``mask_geom`` dicts -> the sampled point cloud, or None.

    A merged feature arrives as a list of member geoms and contributes every
    member's segments, so "east of Neuralia Road" asks about all its ways.
    """
    out = []
    for g in geoms:
        if not isinstance(g, dict):
            continue
        if len(g.get("A", ())):
            out.append(densify_segments(g["A"], g["B"]))
        elif len(g.get("V", ())):
            # A POINT reference has vertices and no segments: "north of that
            # corner" is a band around one point.
            out.append(np.asarray(g["V"], dtype=float))
    return np.vstack(out) if out else None
