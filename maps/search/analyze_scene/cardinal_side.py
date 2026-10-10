"""Which side of an extended feature a point is on, in map-fixed cardinals.

"East of Neuralia Road" has to be decidable when the reference is a 64-way
merged feature with 8,849 cells spanning kilometres. The three cheap answers
all fail on this bake's real shapes:

* **Nearest-point bearing** (``brg_gap_<ref>``) reads the min-gap vector. It
  is local, so it survives curvature and multi-part merges, and it is right
  93.2% of the time on the corpus -- but it inverts on a self-approaching
  shape. A road runs north-south, you are west of it, and just past you it
  loops back east: the nearest piece is now the loop, which lies west of
  you, so min-gap says EAST where a person says WEST. Measured, its ten
  misses concentrate on exactly those shapes (three on a 15,492-cell
  concave lake, one on a golf course at a 0.9 m gap where the sign is
  noise).
* **A baked centerline** describes ONE way. A large road is many ways, so a
  centerline route would answer about the way the phrase happened to hit
  rather than about the road — passing on short single-way roads in testing
  and failing silently on the ones that matter.
* **A principal axis** collapses a bent road to one straight direction.
  Measured at 90.5%, beaten by the rule below.

THE RULE: a cardinal is an ABSOLUTE axis, so the only real question is how
to aggregate over the reference's extent. Restrict the reference to the
cells lying within a band of the query point on the axis ORTHOGONAL to the
cardinal, then compare along the cardinal. "East of the road" becomes "east
of the road WHERE THE ROAD IS, at my latitude" -- which is what a person
means, and which needs no centerline, no skeleton, and no bounding box.

It ABSTAINS rather than guessing, in the two cases where there is no side:

* the reference straddles the point inside the band (a loop closing around
  it, a cul-de-sac bulb, a road crossing the band twice), and
* the band holds no reference cells at all (asking "east of" a road that
  runs exactly east-west). That is genuine ambiguity, not imprecision.

Abstention is a ``None`` return, which is the executor's existing
missing-value idiom: the column is simply absent on that row and
``score_by`` treats it as the neutral. Nothing is dropped and no side is
invented.

Measured over the 147 cue compass-mentions that resolve to a bake feature
(``benchmarks.citynav.analysis.cardinal_frame``): correct on 129 of the 130
it decides, abstaining on 17. Stable from a +-50 m band to a +-1000 m one
(97.7-99.2%), so the width is not a tuned knob.

Cells, never boxes: a road's minimum bounding rectangle is 50% road at the
median and 2.9% at p10 on this bake, so the box describes the empty corner
as confidently as the road.
"""

from __future__ import annotations

import numpy as np

from maps.search.analyze_scene import geom_compute as _gc

# Compass word -> unit vector in the engine's one frame (x = east, y = NORTH;
# see the geom_compute docstring -- row 0 of every repo grid is the SOUTH
# edge). DERIVED from the one table rather than copied, so this module can
# never disagree with it about which way north points.
#
# Only the four axis-aligned cardinals: a diagonal ("northeast of") has no
# single orthogonal axis to band on, so it is not offered rather than
# answered badly. The corpus says that costs nothing -- of its 150
# conjunctive cues, 95 read "the <diag> corner of A and B" and 17 more are a
# diagonal of an intersection or of a street's end, every one a quadrant of a
# POINT, where the conjunction of two cardinal cuts is exact rather than an
# approximation. Only ~14 ask a diagonal of an extended feature.
CARDINALS: dict[str, tuple[float, float]] = {
    w: _gc.COMPASS_UNIT[w] for w in sorted(_gc.CARDINALS)
}

# Half-width of the band, in metres. Wide enough that a block-scale target
# still sees the road it is being related to, narrow enough that a distant
# limb of the same name does not vote. The sweep says the exact value does
# not matter (+-50 m through +-1000 m all land within 1.5 points), so this
# is a readable round number rather than a fitted one.
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

    The executor measures on the exact ``mask`` rather than on the cell
    raster (a cell-set intersection agreed with the mask on only 85% of the
    road pairs the model asked for), so the runtime column is computed from
    mask segments while the review measured cells. Sampling the segments at
    the cell pitch makes the two the SAME answer: over the 138 corpus
    mentions where both decide, the sign agrees 138/138 and the abstain sets
    are identical.

    Sampling is what makes that true. A road running a kilometre north-south
    may be TWO vertices, and a band around a point halfway along it would
    contain neither of them and abstain on a road that is plainly there.
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

    The op layer holds no numpy (geometry lives here and in geom_compute),
    so the assembly is on this side of the seam. A merged feature arrives as
    a list of member geoms and contributes every member's segments, which is
    what makes "east of Neuralia Road" a question about all 64 ways rather
    than about the largest one.
    """
    out = []
    for g in geoms:
        if not isinstance(g, dict):
            continue
        if len(g.get("A", ())):
            out.append(densify_segments(g["A"], g["B"]))
        elif len(g.get("V", ())):
            # A POINT reference has vertices and no segments, and "north of
            # that corner" is the simplest cardinal there is -- a band around
            # one point, compared along the axis. Without this the column was
            # never written and the model got a loud "nothing orders by
            # 'north of 5284_700'"; it wrote exactly that 9 times on 25713024
            # before the fallback existed.
            out.append(np.asarray(g["V"], dtype=float))
    return np.vstack(out) if out else None


def _demo() -> None:
    """Self-check on the five shapes the frame review has to answer."""
    def road_ns(x=0.0, y0=-500.0, y1=500.0, n=501):
        return np.column_stack([np.full(n, x), np.linspace(y0, y1, n)])

    # 1. A straight north-south road; the point is west of it. A positive
    #    return means the point IS in the direction asked about.
    road = road_ns()
    assert cardinal_offset(road, (-80.0, 0.0), "west") == 80.0
    assert cardinal_offset(road, (-80.0, 0.0), "east") == -80.0

    # 2. THE COUNTEREXAMPLE, as a U: the road runs north, turns, and comes
    #    back south on the far side. A point BETWEEN the legs has the west
    #    leg as its nearest piece, so min-gap reports it EAST of the road --
    #    confidently, and wrongly. The band sees road on both sides at that
    #    latitude and declines to name a side, which is the honest answer.
    u_road = np.vstack([road, road_ns(x=-200.0),
                           np.column_stack([np.linspace(-200, 0, 201),
                                            np.full(201, 500.0)])])
    inside = (-150.0, 0.0)
    nearest = u_road[np.hypot(u_road[:, 0] - inside[0],
                              u_road[:, 1] - inside[1]).argmin()]
    assert inside[0] - nearest[0] > 0, "min-gap must invert here (says EAST)"
    assert all(cardinal_offset(u_road, inside, d) is None
               for d in ("east", "west"))
    # A point west of the WHOLE U still decides, and decides west.
    assert cardinal_offset(u_road, (-400.0, 0.0), "west") == 200.0
    assert cardinal_offset(u_road, (-400.0, 0.0), "east") == -200.0

    # 3. A road running exactly east-west, asked "east of". No side exists.
    ew = np.column_stack([np.linspace(-500, 500, 501), np.zeros(501)])
    assert cardinal_offset(ew, (0.0, 200.0), "east") is None
    assert cardinal_offset(ew, (0.0, 200.0), "north") == 200.0

    # 4. A closed loop with the point INSIDE: inside/outside is the
    #    meaningful relation there, so every cardinal abstains.
    th = np.linspace(0, 2 * np.pi, 361)
    ring = np.column_stack([300 * np.cos(th), 300 * np.sin(th)])
    assert all(cardinal_offset(ring, (0.0, 0.0), d) is None for d in CARDINALS)
    # ...and a point outside it still decides.
    assert cardinal_offset(ring, (500.0, 0.0), "east") == 200.0

    # 5. An isotropic blob: no axis at all, and the band still decides.
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 40, size=(2000, 2))
    east = cardinal_offset(blob, (400.0, 0.0), "east")
    assert east is not None and 200.0 < east < 400.0

    # A cul-de-sac bulb wrapping the point abstains; the stem decides.
    bulb = np.vstack([
        np.column_stack([np.zeros(200), np.linspace(-400, 0, 200)]),
        np.column_stack([100 * np.cos(th), 100 * np.sin(th) + 100])])
    assert cardinal_offset(bulb, (0.0, 100.0), "east") is None
    assert cardinal_offset(bulb, (-300.0, -200.0), "west") == 300.0

    # Empty reference and a bad word are distinguishable failures.
    assert cardinal_offset(np.empty((0, 2)), (0.0, 0.0), "east") is None
    try:
        cardinal_offset(road, (0.0, 0.0), "northeast")
    except ValueError:
        pass
    else:
        raise AssertionError("a diagonal must refuse, not answer")
    print("cardinal_side: all five shapes behave")


if __name__ == "__main__":
    _demo()
