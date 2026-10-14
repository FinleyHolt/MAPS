"""The spatial math the geometry op program runs on: compass words, colour
columns, the composite sort, and exact-mask distance, containment and
crossing.

Pure functions of their arguments over world metres in ONE frame, x = east and
y = north (row 0 of every grid is the south edge); every compass word maps
through :data:`COMPASS_UNIT`.
"""

from __future__ import annotations

import math

import numpy as np

from maps.util.polyline import clip_t

# 8-point compass wedges over compass bearing (0=N, clockwise).
_COMPASS8 = (
    (22.5, "north"), (67.5, "northeast"), (112.5, "east"), (157.5, "southeast"),
    (202.5, "south"), (247.5, "southwest"), (292.5, "west"), (337.5, "northwest"),
)

# The compass-word unit vectors, world frame (x east, y north): the eight
# words :func:`_compass8` emits.
_SQRT_HALF = math.sqrt(0.5)
COMPASS_UNIT = {
    "north": (0.0, 1.0), "northeast": (_SQRT_HALF, _SQRT_HALF),
    "east": (1.0, 0.0), "southeast": (_SQRT_HALF, -_SQRT_HALF),
    "south": (0.0, -1.0), "southwest": (-_SQRT_HALF, -_SQRT_HALF),
    "west": (-1.0, 0.0), "northwest": (-_SQRT_HALF, _SQRT_HALF),
}

# The four axis-aligned winds: :func:`cardinal_offset` bands on the
# orthogonal axis, and a diagonal has none.
CARDINALS = frozenset({"north", "south", "east", "west"})


def cw(v):
    """Clockwise quarter-turn: a viewer's right is ``cw(forward)``."""
    return (v[1], -v[0])


def _xy(p):
    """``(x, y)`` floats of ``[x, y]``."""
    return float(p[0]), float(p[1])


def _compass8(bearing_deg: float) -> str:
    b = float(bearing_deg) % 360.0
    for hi, name in _COMPASS8:
        if b < hi:
            return name
    return "north"


def dist(a, b) -> float:
    """Euclidean distance in metres."""
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(bx - ax, by - ay)


def bearing(a, b) -> float:
    """Compass bearing of ``a -> b`` in degrees, 0 = north, clockwise."""
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0


def cardinal(a, b) -> str:
    """8-point compass direction of ``a -> b`` ('north', 'northeast', ...)."""
    return _compass8(bearing(a, b))


# Half-width of the cardinal band, in metres: wide enough that a block-scale
# target still sees the road, narrow enough that a distant limb of the same
# name does not vote.
BAND_M = 150.0


def cardinal_offset(cells_xy, at_xy, direction: str) -> float | None:
    """Signed metres the point lies in ``direction`` (one of
    :data:`CARDINALS`) from an extended reference, or ``None`` when there is
    no side to name.

    The reference's points (``cells_xy``, every member way of a merged
    feature) are cut to the band of :data:`BAND_M` around the point on the axis
    orthogonal to ``direction`` ("east of the road where the road is, at my
    latitude"), then compared along it. Positive: the clearance past the
    band's farthest reference point in that direction. ``None`` when the band holds no
    reference point, or the reference straddles the point inside it."""
    d = str(direction or "").strip().lower()
    if d not in CARDINALS:
        raise ValueError(
            f"cardinal_offset direction must be one of "
            f"{sorted(CARDINALS)}; got {direction!r}")
    pts = np.asarray(cells_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return None
    u = np.asarray(COMPASS_UNIT[d], dtype=float)
    q = np.asarray(at_xy, dtype=float)
    orth = np.array([-u[1], u[0]], dtype=float)
    near = pts[np.abs((pts - q) @ orth) <= BAND_M]
    if not len(near):
        return None
    along = near @ u
    here = float(q @ u)
    lo, hi = float(along.min()), float(along.max())
    if lo < here < hi:
        return None
    return here - hi if here >= hi else here - lo


def densify_segments(seg_a, seg_b):
    """Segment endpoints -> a point cloud at the bake's 2 m pitch, so a band
    around a point halfway along a two-vertex road still holds the road."""
    step = 2.0
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
    """The :func:`cardinal_offset` point cloud of one or more
    :func:`mask_geom` dicts (every member's densified segments, or its
    vertices when it has none), or ``None``."""
    out = []
    for g in geoms:
        if len(g.get("A", ())):
            out.append(densify_segments(g["A"], g["B"]))
        elif len(g.get("V", ())):
            out.append(np.asarray(g["V"], dtype=float))
    return np.vstack(out) if out else None


# Colour matches by FAMILY: perception's usual slip stays inside one (gray
# reads silver under glare, red reads maroon).
# Palette word -> family; key order is the order a label's colour is read in
# (geom_program._color_of).
COLOR_FAMILY = {
    "white": "achromatic", "black": "achromatic", "gray": "achromatic", "silver": "achromatic",
    "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
    "orange": "warm", "brown": "warm", "tan": "warm", "beige": "warm", "gold": "warm",
    "maroon": "red", "purple": "purple",
}


def _color_word_raw(a: str, b: str) -> float:
    """1.0 same word, 0.8 same family, 0.0 another family, 0.5 when either is
    absent; a word outside the families matches by substring."""
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    fa, fb = COLOR_FAMILY.get(a), COLOR_FAMILY.get(b)
    if fa is not None and fb is not None:
        return 0.8 if fa == fb else 0.0
    return 1.0 if (a in b or b in a) else 0.0  # unknown words: substring fallback


def _parse_color_mixture(s: str) -> "list[tuple[str, float]] | None":
    """``[(word, fraction), ...]`` of a mixture ("red 0.45 / black 0.30",
    ``|`` also separates), or ``None`` when ``s`` is not one."""
    parts = [p.strip() for p in (s or "").replace("|", "/").split("/")]
    comps: list[tuple[str, float]] = []
    for part in parts:
        toks = part.split()
        if len(toks) != 2 or not toks[0].isalpha():
            return None
        try:
            frac = float(toks[1])
        except ValueError:
            return None
        if not 0.0 <= frac <= 1.0:
            return None
        word = toks[0].lower()
        comps.append((word, frac))
    return comps or None


def color_match(cand_color: str, cue_color: str, conf: float | None = None) -> float:
    """The value of a ``color_<word>`` column: :func:`_color_word_raw`, each
    mixture component shrunk toward 0.5 by its fraction and the best kept,
    then shrunk toward 0.5 by ``conf`` when given."""
    b = (cue_color or "").strip().lower()
    mix = _parse_color_mixture(cand_color or "")
    if mix is not None:
        raw = max(0.5 + frac * (_color_word_raw(w, b) - 0.5) for w, frac in mix)
    else:
        a = (cand_color or "").strip().lower()
        raw = _color_word_raw(a, b)
    if conf is None:
        return raw
    c = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else float(conf))
    return 0.5 + c * (raw - 0.5)


# The closed palette the surface serves; :func:`normalize_color` maps an open
# word onto it.
COLOR_ENUM = ["white", "gray", "black", "red", "blue", "green", "yellow", "orange", "brown", "purple"]

_COLOR_SYNONYMS = {
    "silver": "gray", "charcoal": "gray", "slate": "gray",
    "maroon": "red", "crimson": "red", "scarlet": "red", "burgundy": "red", "pink": "red",
    "navy": "blue", "cyan": "blue", "teal": "blue", "turquoise": "blue", "aqua": "blue",
    "violet": "purple", "magenta": "purple", "lavender": "purple",
    "gold": "yellow", "golden": "yellow", "amber": "yellow",
    "cream": "white", "beige": "brown", "tan": "brown", "bronze": "brown", "copper": "brown",
    "olive": "green", "lime": "green",
}


def normalize_color(word: str) -> str:
    """``word`` on :data:`COLOR_ENUM` by synonym, or itself for the caller to
    refuse."""
    w = (word or "").strip().lower()
    if w in COLOR_ENUM:
        return w
    return _COLOR_SYNONYMS.get(w, w)


# ``color_<word>`` = ``color_match(row.color, word, row.color_conf)``; a row
# with no perceived colour omits it.
COLOR_COLUMNS = frozenset("color_" + w for w in COLOR_ENUM)


def compact_color_cell(color_str: str, *, shade: "str | None", uncertain: bool,
                       rich: bool) -> str:
    """The rendered colour cell ``[dk|lt ]word[/word2][?]`` of a mixture
    (``shade`` is ``dk``, ``lt`` or None): ``word2`` when ``rich`` and the
    second component crosses a family, ``?`` on an uncertain read."""
    s = (color_str or "").strip()
    if not s or s == "?":
        return "?"
    mix = _parse_color_mixture(s)
    if mix is None:  # a plain single word (not a mixture)
        w = s.lower()
        return f"{w}?" if uncertain else w
    word = mix[0][0]
    second = None
    if rich and len(mix) > 1 and mix[1][1] >= 0.20 and mix[1][0] != word:
        w2 = mix[1][0]
        if COLOR_FAMILY.get(word) != COLOR_FAMILY.get(w2):
            second = w2
    head = f"{shade} " if shade else ""
    body = f"{word}/{second}" if second else word
    return f"{head}{body}?" if uncertain else f"{head}{body}"


def arc_lengths(poly_en):
    out = [0.0]
    for i in range(len(poly_en) - 1):
        out.append(out[-1] + math.hypot(poly_en[i + 1][0] - poly_en[i][0],
                                        poly_en[i + 1][1] - poly_en[i][1]))
    return out


# Cap on the adaptive score stamp: composites still equal at this many
# decimals are float noise apart, a true tie.
_SCORE_MAX_DP = 10


def score_by(items, keys, weights) -> list[dict]:
    """Rows best-first by ``sum(w * term)``, each stamped with ``score``.

    A ``color_<word>`` term is the value centred on its 0.5 neutral (missing:
    0, the row has not been seen yet); any other key is min-max scaled over
    the rows carrying it (missing or constant: 0.5). A weight's sign is its
    direction (negative: lower is better). ``score`` is rounded to the fewest
    decimals (4 to ``_SCORE_MAX_DP``) that keep distinct composites distinct;
    ties keep input order."""
    spans = {}
    for k in keys:
        vals = [r[k] for r in items if isinstance(r.get(k), (int, float))]
        spans[k] = (min(vals), max(vals)) if vals else (0.0, 0.0)

    def term(r, k):
        v = r.get(k)
        if k in COLOR_COLUMNS:
            return v - 0.5 if isinstance(v, (int, float)) else 0.0
        lo, hi = spans[k]
        return (v - lo) / (hi - lo) if isinstance(v, (int, float)) and hi > lo else 0.5

    raws = [sum(float(w) * term(r, k) for w, k in zip(weights, keys)) for r in items]
    distinct = sorted(set(raws))
    dp = 4
    while dp < _SCORE_MAX_DP and any(
            round(a, dp) == round(b, dp) for a, b in zip(distinct, distinct[1:])):
        dp += 1
    # + 0.0 turns a signed -0.0 into 0.0.
    out = [{**r, "score": round(s, dp) + 0.0} for r, s in zip(items, raws)]
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


# --- exact-mask geometry -----------------------------------------------------
# A mask is the chains as ingested (a feature's contour, else its cell
# boundary loops, else its polyline, else its point), never simplified. Gap
# columns measure mask to mask (0.0 on containment, touch or crossing);
# position columns measure from the area centroid.


def point_in_polygon(pt, poly) -> bool:
    """Even-odd ray cast of ``pt`` against ``poly`` (closing edge implicit)."""
    if not poly or len(poly) < 3:
        return False
    x, y = _xy(pt)
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


# Memory bound on the vectorised (points x segments) blocks: batching only,
# every pair is still computed exactly.
_PAIR_CHUNK = 4_000_000


def _shoelace(loops):
    """Shoelace sums ``(sum cross, sum (x1+x2) cross, sum (y1+y2) cross)``
    over the CLOSED chains; hole loops wind opposite and subtract."""
    sa = sx = sy = 0.0
    for lp in (loops or []):
        if not is_closed_chain(lp):
            continue
        n = len(lp)
        for i in range(n):
            x1, y1 = float(lp[i][0]), float(lp[i][1])
            x2, y2 = float(lp[(i + 1) % n][0]), float(lp[(i + 1) % n][1])
            cr = x1 * y2 - x2 * y1
            sa += cr
            sx += (x1 + x2) * cr
            sy += (y1 + y2) * cr
    return sa, sx, sy


def loops_area_centroid(loops):
    """Area centroid ``[x, y]`` of a mask realised as boundary loops
    (:func:`_shoelace`), or the vertex mean when no closed chain encloses
    area, or ``None`` with no vertices."""
    sa, sx, sy = _shoelace(loops)
    if abs(sa) < 1e-12:
        pts = [p for lp in (loops or []) for p in lp]
        if not pts:
            return None
        return [sum(float(p[0]) for p in pts) / len(pts),
                sum(float(p[1]) for p in pts) / len(pts)]
    return [sx / (3.0 * sa), sy / (3.0 * sa)]


def loops_area(loops):
    """Enclosed area (m^2) of a mask realised as boundary loops
    (:func:`_shoelace`); 0.0 for a point or a way."""
    return abs(_shoelace(loops)[0]) / 2.0


def is_closed_chain(pts) -> bool:
    """A chain encloses iff it returns to its first vertex (every producer
    emits the closing vertex)."""
    return (len(pts) >= 4
            and float(pts[0][0]) == float(pts[-1][0])
            and float(pts[0][1]) == float(pts[-1][1]))


def clip_chains_to_box(chains, box) -> list:
    """``chains`` cut to ``box`` = ``(x0, y0, x1, y1)``, one chain per run
    inside it: a crossing segment is cut at the edge, and a cut ring stays
    open rather than gaining a wall along the tile edge."""
    x0, y0, x1, y1 = (float(box[0]), float(box[1]),
                      float(box[2]), float(box[3]))

    def inside(p):
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def at(a, b, t):
        # The endpoints come back verbatim: interpolation is not
        # bit-identical, and :func:`one_place` joins on exact vertex
        # identity.
        if t <= 0.0:
            return [a[0], a[1]]
        if t >= 1.0:
            return [b[0], b[1]]
        # Clamped: a cut is on the boundary, and the interpolation can land a
        # few ULPs past it.
        return [min(max(a[0] + (b[0] - a[0]) * t, x0), x1),
                min(max(a[1] + (b[1] - a[1]) * t, y0), y1)]

    out: list = []
    for ch in chains or ():
        pts = [[float(p[0]), float(p[1])] for p in ch]
        if len(pts) == 1:
            if inside(pts[0]):
                out.append(pts)
            continue
        run: list = []
        for a, b in zip(pts, pts[1:]):
            seg = clip_t(a, b, (x0, y0, x1, y1))
            if seg is None:
                if run:
                    out.append(run)
                    run = []
                continue
            t0, t1 = seg
            pa, pb = at(a, b, t0), at(a, b, t1)
            if not run:
                run = [pa]
            elif run[-1] != pa:
                # The previous segment left the box before this one entered.
                out.append(run)
                run = [pa]
            if pb != run[-1]:
                run.append(pb)
            if t1 < 1.0:
                out.append(run)
                run = []
        if run:
            out.append(run)
    return [c for c in out if c]


def one_place(chains) -> bool:
    """Whether a mask realisation is ONE place: its chains all join by a
    SHARED VERTEX (exact identity, no tolerance) or by one closed chain
    ENCLOSING another (a hole belongs to its ring). Two fragments of a
    speckled mask, or a way the tile boundary cut in two, are not."""
    chains = [ch for ch in (chains or []) if ch]
    n = len(chains)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def groups():
        return len({find(i) for i in range(n)})

    first: dict = {}
    for i, ch in enumerate(chains):
        for p in ch:
            parent[find(i)] = find(first.setdefault((float(p[0]), float(p[1])), i))
    if groups() > 1:
        for i, ch in enumerate(chains):
            if is_closed_chain(ch):
                for j in range(n):
                    if j != i and point_in_polygon(chains[j][0], ch):
                        parent[find(j)] = find(i)
    return groups() == 1


def mask_geom(chains):
    """The arrays of one mask's chains: ``V`` vertices, ``BB`` their box
    (``None`` when empty), ``A``/``B`` every segment (distance), ``CA``/``CB``
    the closed chains' segments (containment), and ``closed``."""
    V: list = []
    A: list = []
    B: list = []
    CA: list = []
    CB: list = []
    for ch in (chains or []):
        pts = [(float(p[0]), float(p[1])) for p in ch]
        if not pts:
            continue
        V.extend(pts)
        closed = is_closed_chain(pts)
        for i in range(len(pts) - 1):
            A.append(pts[i])
            B.append(pts[i + 1])
            if closed:
                CA.append(pts[i])
                CB.append(pts[i + 1])
    Va = np.asarray(V, dtype=float).reshape(-1, 2)
    return {"V": Va,
            "BB": (float(Va[:, 0].min()), float(Va[:, 1].min()),
                   float(Va[:, 0].max()), float(Va[:, 1].max())) if len(Va) else None,
            "A": np.asarray(A, dtype=float).reshape(-1, 2),
            "B": np.asarray(B, dtype=float).reshape(-1, 2),
            "CA": np.asarray(CA, dtype=float).reshape(-1, 2),
            "CB": np.asarray(CB, dtype=float).reshape(-1, 2),
            "closed": bool(CA)}


def bbox_gap(ga, gb) -> float:
    """A lower bound on :func:`gap_witness`'s gap: the separation of the two
    boxes, 0.0 on overlap or an empty geom."""
    a, b = ga["BB"], gb["BB"]
    if a is None or b is None:
        return 0.0
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


#: Slack on a bound comparison, in metres: the bound and the gap can disagree
#: by an ulp, and slack only ever keeps more.
BOUND_SLACK_M = 1e-6


def stack_bboxes(geoms) -> np.ndarray:
    """``(M, 4)`` of the members' boxes, ``nan`` where one has none."""
    out = np.full((len(geoms), 4), np.nan, dtype=float)
    for i, g in enumerate(geoms):
        if g["BB"] is not None:
            out[i] = g["BB"]
    return out


def bbox_bounds(stacked: np.ndarray, g) -> np.ndarray:
    """:func:`bbox_gap` from every row of ``stacked`` to ``g``; an unknown
    box reads 0.0."""
    if len(stacked) == 0:
        return np.zeros(0, dtype=float)
    bb = g["BB"]
    if bb is None:
        return np.zeros(len(stacked), dtype=float)
    dx = np.maximum(0.0, np.maximum(stacked[:, 0] - bb[2],
                                    bb[0] - stacked[:, 2]))
    dy = np.maximum(0.0, np.maximum(stacked[:, 1] - bb[3],
                                    bb[1] - stacked[:, 3]))
    return np.nan_to_num(np.hypot(dx, dy), nan=0.0)


def _inside_mask_arr(P, A, B):
    """Even-odd membership of points ``P`` over the closed-loop edges
    ``A``->``B``, all loops together. Points are tested in y-order, so each
    chunk reads only the edges whose y-span meets its own."""
    out = np.zeros(len(P), dtype=bool)
    if len(P) == 0 or len(A) == 0:
        return out
    # Only the points inside the loops' box: a point outside it crosses an
    # even number of edges.
    inbox = ((P[:, 0] >= min(float(A[:, 0].min()), float(B[:, 0].min())))
             & (P[:, 0] <= max(float(A[:, 0].max()), float(B[:, 0].max())))
             & (P[:, 1] >= min(float(A[:, 1].min()), float(B[:, 1].min())))
             & (P[:, 1] <= max(float(A[:, 1].max()), float(B[:, 1].max()))))
    if not inbox.any():
        return out
    ylo = np.minimum(A[:, 1], B[:, 1])
    yhi = np.maximum(A[:, 1], B[:, 1])
    order = np.flatnonzero(inbox)
    order = order[np.argsort(P[order, 1], kind="stable")]
    step = max(1, int(_PAIR_CHUNK // max(1, len(A))))
    for s in range(0, len(order), step):
        idx = order[s:s + step]
        p = P[idx]
        lo, hi = float(p[:, 1].min()), float(p[:, 1].max())
        sel = (ylo <= hi) & (yhi > lo)
        if not sel.any():
            continue
        a, b = A[sel], B[sel]
        Ax, Ay = a[:, 0][None, :], a[:, 1][None, :]
        Bx, By = b[:, 0][None, :], b[:, 1][None, :]
        px, py = p[:, 0][:, None], p[:, 1][:, None]
        cond = (Ay > py) != (By > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (Bx - Ax) * (py - Ay) / (By - Ay) + Ax
        cross = cond & (px < xin)
        out[idx] = (cross.sum(axis=1) & 1).astype(bool)
    return out


def mask_contains(geom, pts):
    """Which ``pts`` lie inside the closed chains of a :func:`mask_geom`."""
    P = np.asarray([[float(p[0]), float(p[1])] for p in pts],
                   dtype=float).reshape(-1, 2)
    return _inside_mask_arr(P, geom["CA"], geom["CB"])


# Past this many (point, segment) pairs the argmin is looked for on the pairs
# that can hold it (``_min_pt_to_segs_pruned``).
_PRUNE_PAIRS = 100_000


def _min_pt_to_segs(P, A, B):
    """``(d, i, foot)``: the least distance from points ``P`` to segments
    ``A``->``B``, its point index and its foot."""
    if len(P) * len(A) > _PRUNE_PAIRS:
        return _min_pt_to_segs_pruned(P, A, B)
    return _min_pt_to_segs_all(P, A, B)


def _min_pt_to_segs_pruned(P, A, B):
    """The full scan's triple over the points and segments whose box bound
    does not exceed an upper bound from the best-bounded of each; the
    minimal pairs survive in order, so the witness is the same pair."""
    sx0, sx1 = np.minimum(A[:, 0], B[:, 0]), np.maximum(A[:, 0], B[:, 0])
    sy0, sy1 = np.minimum(A[:, 1], B[:, 1]), np.maximum(A[:, 1], B[:, 1])
    px0, px1 = float(P[:, 0].min()), float(P[:, 0].max())
    py0, py1 = float(P[:, 1].min()), float(P[:, 1].max())
    bx0, bx1 = float(sx0.min()), float(sx1.max())
    by0, by1 = float(sy0.min()), float(sy1.max())
    bp = np.hypot(np.maximum(0.0, np.maximum(bx0 - P[:, 0], P[:, 0] - bx1)),
                  np.maximum(0.0, np.maximum(by0 - P[:, 1], P[:, 1] - by1)))
    bs = np.hypot(np.maximum(0.0, np.maximum(px0 - sx1, sx0 - px1)),
                  np.maximum(0.0, np.maximum(py0 - sy1, sy0 - py1)))
    s0, p0 = int(bs.argmin()), int(bp.argmin())
    ub = min(_min_pt_to_segs_all(P, A[s0:s0 + 1], B[s0:s0 + 1])[0],
             _min_pt_to_segs_all(P[p0:p0 + 1], A, B)[0])
    keep_p = np.flatnonzero(bp <= ub + BOUND_SLACK_M)
    keep_s = np.flatnonzero(bs <= ub + BOUND_SLACK_M)
    d, i, foot = _min_pt_to_segs_all(P[keep_p], A[keep_s], B[keep_s])
    return d, (int(keep_p[i]) if i >= 0 else -1), foot


def _min_pt_to_segs_all(P, A, B):
    """Every pair, in order: the scan ``_min_pt_to_segs`` is defined by."""
    if len(P) == 0 or len(A) == 0:
        return float("inf"), -1, None
    best_d2, best_i, best_foot = float("inf"), -1, None
    step = max(1, int(_PAIR_CHUNK // max(1, len(A))))
    for s in range(0, len(P), step):
        d2, F = _pt_to_segs_grid(P[s:s + step], A, B)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if float(d2[i, j]) < best_d2:
            best_d2 = float(d2[i, j])
            best_i = s + i
            best_foot = (float(F[i, j, 0]), float(F[i, j, 1]))
    return math.sqrt(best_d2), best_i, best_foot


def _pt_to_segs_grid(P, A, B):
    """``(d2, F)`` for EVERY (point, segment) pair: squared distances
    ``(N, M)`` and foot points ``(N, M, 2)`` from points ``P`` to segments
    ``A``->``B``."""
    E = B - A
    L2 = np.einsum("mj,mj->m", E, E)
    L2s = np.where(L2 > 0.0, L2, 1.0)
    W = P[:, None, :] - A[None, :, :]
    t = np.einsum("nmj,mj->nm", W, E) / L2s[None, :]
    t = np.clip(np.where(L2[None, :] > 0.0, t, 0.0), 0.0, 1.0)
    F = A[None, :, :] + t[..., None] * E[None, :, :]
    D = P[:, None, :] - F
    return np.einsum("nmj,nmj->nm", D, D), F


def _proper_crossings(a1, b1, A2, B2):
    """``(X, P)``: which segment pairs ``a1->b1`` (n) x ``A2->B2`` (m)
    PROPERLY cross (interiors intersect), and their ``(n, m, 2)`` contact
    points (``None`` when none cross)."""
    r1 = b1 - a1
    r2 = B2 - A2
    W1 = A2[None, :, :] - a1[:, None, :]
    W2 = B2[None, :, :] - a1[:, None, :]
    o1 = r1[:, None, 0] * W1[..., 1] - r1[:, None, 1] * W1[..., 0]
    o2 = r1[:, None, 0] * W2[..., 1] - r1[:, None, 1] * W2[..., 0]
    W3 = a1[:, None, :] - A2[None, :, :]
    W4 = b1[:, None, :] - A2[None, :, :]
    o3 = r2[None, :, 0] * W3[..., 1] - r2[None, :, 1] * W3[..., 0]
    o4 = r2[None, :, 0] * W4[..., 1] - r2[None, :, 1] * W4[..., 0]
    X = (o1 * o2 < 0.0) & (o3 * o4 < 0.0)
    if not X.any():
        return X, None
    rxs = r1[:, None, 0] * r2[None, :, 1] - r1[:, None, 1] * r2[None, :, 0]
    t = ((W1[..., 0] * r2[None, :, 1] - W1[..., 1] * r2[None, :, 0])
         / np.where(rxs != 0.0, rxs, 1.0))
    return X, a1[:, None, :] + t[..., None] * r1[:, None, :]


def _overlaps_box(A, B, C, D):
    """Which segments ``A``->``B`` overlap the bounding box of the segments
    ``C``->``D`` (inclusive on the boundary, so a touch is never dropped)."""
    xmin = min(float(C[:, 0].min()), float(D[:, 0].min()))
    xmax = max(float(C[:, 0].max()), float(D[:, 0].max()))
    ymin = min(float(C[:, 1].min()), float(D[:, 1].min()))
    ymax = max(float(C[:, 1].max()), float(D[:, 1].max()))
    return ((np.maximum(A[:, 0], B[:, 0]) >= xmin)
            & (np.minimum(A[:, 0], B[:, 0]) <= xmax)
            & (np.maximum(A[:, 1], B[:, 1]) >= ymin)
            & (np.minimum(A[:, 1], B[:, 1]) <= ymax))


def _segments_cross(A1, B1, A2, B2):
    """The first proper crossing between two segment sets, or None (a touch
    already reads 0 through the vertex-to-segment distance)."""
    if len(A1) == 0 or len(A2) == 0:
        return None
    # Only the segments whose boxes meet: each side keeps, in order, the
    # segments overlapping the other side's box, so the first hit in
    # row-major order is the same pair.
    k1 = _overlaps_box(A1, B1, A2, B2)
    k2 = _overlaps_box(A2, B2, A1, B1)
    if not (k1.all() and k2.all()):
        if not k1.any() or not k2.any():
            return None
        A1, B1, A2, B2 = A1[k1], B1[k1], A2[k2], B2[k2]
    step = max(1, int(_PAIR_CHUNK // max(1, len(A2))))
    for k in range(0, len(A1), step):
        X, P = _proper_crossings(A1[k:k + step], B1[k:k + step], A2, B2)
        if P is not None:
            i, j = np.unravel_index(int(np.argmax(X)), X.shape)
            return float(P[i, j, 0]), float(P[i, j, 1])
    return None


def gap_witness(ga, gb):
    """``(gap, witness_a, witness_b)``: the exact mask-to-mask distance
    between two :func:`mask_geom` and the points achieving it; 0.0 at the
    contact point on containment, touch or crossing; ``(inf, None, None)``
    when either is empty."""
    Va, Vb = ga["V"], gb["V"]
    if len(Va) == 0 or len(Vb) == 0:
        return float("inf"), None, None
    # Containment and crossing both need the two boxes to overlap, and
    # ``bbox_gap`` is 0.0 exactly on overlap (touching included).
    if bbox_gap(ga, gb) <= 0.0:
        if len(gb["CA"]):
            ins = _inside_mask_arr(Va, gb["CA"], gb["CB"])
            if ins.any():
                k = int(np.argmax(ins))
                p = (float(Va[k, 0]), float(Va[k, 1]))
                return 0.0, p, p
        if len(ga["CA"]):
            ins = _inside_mask_arr(Vb, ga["CA"], ga["CB"])
            if ins.any():
                k = int(np.argmax(ins))
                p = (float(Vb[k, 0]), float(Vb[k, 1]))
                return 0.0, p, p
        x = _segments_cross(ga["A"], ga["B"], gb["A"], gb["B"])
        if x is not None:
            return 0.0, x, x
    best = (float("inf"), None, None)
    d, i, foot = _min_pt_to_segs(Va, gb["A"], gb["B"])
    if d < best[0]:
        best = (d, (float(Va[i, 0]), float(Va[i, 1])), foot)
    d, i, foot = _min_pt_to_segs(Vb, ga["A"], ga["B"])
    if d < best[0]:
        best = (d, foot, (float(Vb[i, 0]), float(Vb[i, 1])))
    if not math.isfinite(best[0]):
        # neither side carries a segment (point-mask vs point-mask)
        D = Va[:, None, :] - Vb[None, :, :]
        d2 = np.einsum("nmj,nmj->nm", D, D)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        best = (math.sqrt(float(d2[i, j])),
                (float(Va[i, 0]), float(Va[i, 1])),
                (float(Vb[j, 0]), float(Vb[j, 1])))
    return best


def set_gap(geoms, bbs, item_geom):
    """``(gap, ref_witness, item_witness)`` from ``item_geom`` to the
    nearest of ``geoms`` (``bbs`` = :func:`stack_bboxes` of them): a set of
    places is its nearest member.

    The nearest box seeds an upper bound and members whose box is farther
    are skipped; members are still taken in order with a strict test, so a
    tie keeps the earliest member's witness. Contact ends the scan.
    """
    bounds = bbox_bounds(bbs, item_geom)
    ub = None
    seed_i, seed = -1, None
    if len(geoms) > 1 and len(bounds):
        seed_i = int(bounds.argmin())
        seed = gap_witness(geoms[seed_i], item_geom)
        ub = seed[0]
    d = wr = wi = None
    for gi, g in enumerate(geoms):
        if d is not None and d <= 0.0:
            break
        cut = d if d is not None else ub
        if cut is not None and bounds[gi] > cut + BOUND_SLACK_M:
            continue
        _d, _wr, _wi = seed if gi == seed_i else gap_witness(g, item_geom)
        if d is None or _d < d:
            d, wr, wi = _d, _wr, _wi
    return d, wr, wi


def _segment_arrays(polys):
    """``(A, B)`` segment arrays of polylines; a one-vertex polyline is a
    zero-length segment."""
    A: list = []
    B: list = []
    for p in polys or []:
        P = np.asarray([[float(q[0]), float(q[1])] for q in p],
                       dtype=float).reshape(-1, 2)
        # a non-finite vertex would poison every orientation cross-product
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) >= 2:
            A.append(P[:-1])
            B.append(P[1:])
        elif len(P) == 1:
            A.append(P)
            B.append(P)
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(A), np.concatenate(B)


def polyline_crossings(polys_a, polys_b):
    """``(points, nearest_m)``: every distinct place (2 dp) two bundles of
    chains touch at distance exactly zero (a shared vertex, a vertex on a
    segment, or a proper crossing), and their nearest approach."""
    A1, B1 = _segment_arrays(polys_a)
    A2, B2 = _segment_arrays(polys_b)
    if len(A1) == 0 or len(A2) == 0:
        return [], float("inf")
    hits: list = []
    nearest = float("inf")
    step = max(1, int(_PAIR_CHUNK // max(1, len(A2))))
    for k in range(0, len(A1), step):
        a1, b1 = A1[k:k + step], B1[k:k + step]
        d1, F1 = _pt_to_segs_grid(a1, A2, B2)
        d2, F2 = _pt_to_segs_grid(b1, A2, B2)
        d3, F3 = _pt_to_segs_grid(A2, a1, b1)
        d4, F4 = _pt_to_segs_grid(B2, a1, b1)
        d = np.sqrt(np.stack([d1, d2, d3.T, d4.T]))
        w = np.stack([
            (a1[:, None, :] + F1) * 0.5,
            (b1[:, None, :] + F2) * 0.5,
            (A2[None, :, :] + F3.transpose(1, 0, 2)) * 0.5,
            (B2[None, :, :] + F4.transpose(1, 0, 2)) * 0.5,
        ])
        pick = np.argmin(d, axis=0)
        dmin = np.take_along_axis(d, pick[None], 0)[0]
        wmin = np.take_along_axis(w, pick[None, ..., None], 0)[0]
        # A proper crossing is distance 0 at its exact contact, whatever
        # the vertex spacing.
        X, P = _proper_crossings(a1, b1, A2, B2)
        if P is not None:
            dmin = np.where(X, 0.0, dmin)
            wmin = np.where(X[..., None], P, wmin)
        nearest = min(nearest, float(dmin.min()))
        keep = dmin <= 0.0
        if keep.any():
            hits.append(wmin[keep])
    if not hits:
        return [], nearest
    pts = np.round(np.concatenate(hits), 2)
    return ([list(p) for p in sorted({(float(x), float(y)) for x, y in pts})],
            nearest)


def _collapse_collinear(corners):
    """Drop interior vertices of straight runs on a closed rectilinear loop."""
    n = len(corners)
    if n < 3:
        return list(corners)
    out = []
    for i in range(n):
        pr, pc = corners[i - 1]
        r, c = corners[i]
        nr, nc = corners[(i + 1) % n]
        # keep the vertex iff the incoming and outgoing directions differ
        if (r - pr, c - pc) != (nr - r, nc - c):
            out.append((r, c))
    return out or [corners[0]]


def cells_boundary_loops(cells, origin_xy, cell_m):
    """Every closed boundary loop of ``(row, col)`` cells (each fragment's
    outline and each hole, collinear runs collapsed) in world metres, or
    ``None`` when there are none. Holes wind opposite to outlines."""
    cs = {(int(rc[0]), int(rc[1])) for rc in (cells if cells is not None else [])}
    if not cs:
        return None
    # Each cell side with no 4-neighbour is one directed corner edge,
    # clockwise in (row, col), so loops chain head to tail. Every corner
    # has as many edges in as out, so a chain from ``start`` closes there.
    edges: dict[tuple, list] = {}
    for (r, c) in cs:
        if (r - 1, c) not in cs:
            edges.setdefault((r, c), []).append((r, c + 1))
        if (r, c + 1) not in cs:
            edges.setdefault((r, c + 1), []).append((r + 1, c + 1))
        if (r + 1, c) not in cs:
            edges.setdefault((r + 1, c + 1), []).append((r + 1, c))
        if (r, c - 1) not in cs:
            edges.setdefault((r + 1, c), []).append((r, c))
    loops: list = []
    while edges:
        start = min(edges)  # deterministic loop seed
        loop = [start]
        cur = edges[start].pop()
        if not edges[start]:
            del edges[start]
        while cur != start:
            loop.append(cur)
            nxt = edges[cur].pop()
            if not edges[cur]:
                del edges[cur]
            cur = nxt
        lp = _collapse_collinear(loop)
        if len(lp) >= 3:
            loops.append(lp)
    if not loops:
        return None
    tx, ty = float(origin_xy[0]), float(origin_xy[1])
    m = float(cell_m)
    return [[[tx + c * m, ty + r * m] for (r, c) in (*lp, lp[0])]
            for lp in loops]


def mask_shape_features(cells, cell_m):
    """``{"footprint_area_m2": #unique cells x cell_m**2}`` of a rasterized
    footprint (``(row, col)`` cells), or ``None`` when it has no cells."""
    cs = {(int(rc[0]), int(rc[1])) for rc in (cells if cells is not None else [])}
    if not cs:
        return None
    m = float(cell_m)
    return {"footprint_area_m2": round(len(cs) * m * m, 1)}


def contour_shape_features(chains):
    """``{"footprint_area_m2": ...}`` summed over a feature's CLOSED world
    chains (:func:`loops_area`), or ``None`` when no closed chain encloses
    area: a way's "area" would be the accident of closing it."""
    rings = [ch for ch in (chains or []) if is_closed_chain(ch)]
    area = loops_area(rings)
    if area <= 1e-9:
        return None
    return {"footprint_area_m2": round(area, 2)}
